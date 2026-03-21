from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Literal

import httpx

from g1.locomotion.kimodo_controller import walk_to_point_kimodo
from g1.stance.controller import stabilize_for_vision
from g1.state_machine.types import State, StateResult

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from g1.locomotion.sdk_controller import SdkLocomotionController
    from g1.transforms.odometry import OdometrySubscriber
    from g1.transforms.service import TransformService
    from g1.vision.camera import OakCamera
    from g1.vision.detector import ObjectDetector

logger = logging.getLogger(__name__)

KIMODO_URL = os.environ.get("KIMODO_URL", "http://localhost:8420")

_MAX_RETRIES_PER_STATE = 3


class TreasureHuntStateMachine:
    """Look-Move-Look-Act pipeline for finding and interacting with objects.

    Args:
        target_object: Natural language description of the target (e.g. "red box").
        camera: OakCamera instance for RGB+Depth capture.
        detector: ObjectDetector instance for zero-shot detection.
        transforms: TransformService for coordinate frame conversions.
        odometry: OdometrySubscriber for robot state.
        sdk_controller: SDK locomotion controller (walk_method="SDK").
        say: Callback to make the robot speak.
        walk_method: "SDK" for P-controller, "KIMODO" for trajectory.
    """

    def __init__(
        self,
        target_object: str,
        camera: OakCamera,
        detector: ObjectDetector,
        transforms: TransformService,
        odometry: OdometrySubscriber,
        sdk_controller: SdkLocomotionController,
        say: Callable[[str], None],
        walk_method: Literal["SDK", "KIMODO"] = "SDK",
    ) -> None:
        self._target = target_object
        self._camera = camera
        self._detector = detector
        self._tf = transforms
        self._odom = odometry
        self._sdk = sdk_controller
        self._say = say
        self._walk_method = walk_method

        # State shared between handlers
        self._target_world_xyz: np.ndarray | None = None
        self._target_local_xyz: np.ndarray | None = None

    def run(self) -> dict[str, object]:
        """Execute the full LMLA pipeline.

        Returns:
            Dict with final state, coordinates, and any trajectory metadata.
        """
        state = State.LOOK
        retries: dict[State, int] = {}
        result_payload: dict[str, object] = {}

        handlers: dict[State, Callable[[], StateResult]] = {
            State.LOOK: self._handle_look,
            State.MOVE: self._handle_move,
            State.LOOK_AGAIN: self._handle_look_again,
            State.ACT: self._handle_act,
        }

        while state not in (State.DONE, State.FAIL):
            handler = handlers.get(state)
            if handler is None:
                logger.error("No handler for state %s", state)
                state = State.FAIL
                break

            logger.info("Entering state: %s", state.name)
            try:
                result = handler()
            except Exception:
                logger.exception("Exception in state %s", state.name)
                result = StateResult(
                    status="fail",
                    next_state=State.FAIL,
                    message=f"Unhandled exception in {state.name}",
                )

            if result.status == "retry":
                count = retries.get(state, 0) + 1
                retries[state] = count
                if count >= _MAX_RETRIES_PER_STATE:
                    logger.warning(
                        "Max retries (%d) for state %s",
                        _MAX_RETRIES_PER_STATE,
                        state.name,
                    )
                    state = State.FAIL
                    break
                logger.info(
                    "Retrying state %s (%d/%d)",
                    state.name,
                    count,
                    _MAX_RETRIES_PER_STATE,
                )
                state = result.next_state
                continue

            if result.status == "fail":
                state = State.FAIL
                break

            result_payload.update(result.payload)
            state = result.next_state

        return {
            "final_state": state.name,
            "target_object": self._target,
            "target_world_xyz": (
                self._target_world_xyz.tolist()
                if self._target_world_xyz is not None
                else None
            ),
            "target_local_xyz": (
                self._target_local_xyz.tolist()
                if self._target_local_xyz is not None
                else None
            ),
            **result_payload,
        }

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    def _handle_look(self) -> StateResult:
        """LOOK: Stabilize, detect target, transform to world frame."""
        self._say(f"Looking for the {self._target}...")
        stabilize_for_vision(
            velocity_func=self._odom.velocity_magnitude,
        )

        frame = self._camera.capture()
        detections = self._detector.detect(frame, [self._target])

        if not detections:
            return StateResult(
                status="retry",
                next_state=State.LOOK,
                message=f"No {self._target} detected",
            )

        best = detections[0]
        if not best.depth_valid or best.point_camera is None:
            return StateResult(
                status="retry",
                next_state=State.LOOK,
                message="Detection found but depth is invalid",
            )

        # Transform camera point -> world (odom) frame
        world_xyz = self._tf.transform_point_between(
            best.point_camera, "camera", "world"
        )
        self._target_world_xyz = world_xyz
        self._say(f"Found the {self._target}! Walking closer.")

        return StateResult(
            status="ok",
            next_state=State.MOVE,
            payload={
                "look_detection": {
                    "label": best.label,
                    "confidence": best.confidence,
                    "world_xyz": world_xyz.tolist(),
                }
            },
        )

    def _handle_move(self) -> StateResult:
        """MOVE: Walk toward saved world coordinate, stopping 0.5m short."""
        if self._target_world_xyz is None:
            return StateResult(
                status="fail", next_state=State.FAIL, message="No target coordinate"
            )

        target_x = float(self._target_world_xyz[0])
        target_y = float(self._target_world_xyz[1])

        if self._walk_method == "SDK":
            success = self._sdk.walk_to_point(
                target_x=target_x,
                target_y=target_y,
                current_odom_func=self._odom.get_state,
                stop_short_m=0.5,
            )
            if not success:
                return StateResult(
                    status="fail",
                    next_state=State.FAIL,
                    message="Walk timed out",
                )
        else:
            odom = self._odom.get_state()
            result = walk_to_point_kimodo(
                target_x=target_x,
                target_y=target_y,
                current_x=odom.x,
                current_y=odom.y,
                stop_short_m=0.5,
            )
            if result.get("status") == "already_close":
                pass  # No need to walk
            # Trajectory is returned for the RL policy to execute externally.

        # Ensure fully stopped before LOOK_AGAIN
        stabilize_for_vision(velocity_func=self._odom.velocity_magnitude)

        return StateResult(
            status="ok",
            next_state=State.LOOK_AGAIN,
        )

    def _handle_look_again(self) -> StateResult:
        """LOOK_AGAIN: Precision vision update at close range."""
        self._say("Getting a closer look...")
        stabilize_for_vision(velocity_func=self._odom.velocity_magnitude)

        frame = self._camera.capture()
        detections = self._detector.detect(frame, [self._target])

        if not detections:
            return StateResult(
                status="retry",
                next_state=State.LOOK,
                message="Lost target at close range — restarting search",
            )

        best = detections[0]
        if not best.depth_valid or best.point_camera is None:
            # Too close — step backward
            self._say("Too close, stepping back a little.")
            self._sdk.step_backward(0.2)
            return StateResult(
                status="retry",
                next_state=State.LOOK_AGAIN,
                message="Depth invalid at close range, stepping back",
            )

        # Transform to base_link (local frame) for the ACT state
        local_xyz = self._tf.transform_point_between(
            best.point_camera, "camera", "base"
        )
        self._target_local_xyz = local_xyz
        self._say(f"Target locked! Preparing to act on the {self._target}.")

        return StateResult(
            status="ok",
            next_state=State.ACT,
            payload={
                "look_again_detection": {
                    "label": best.label,
                    "confidence": best.confidence,
                    "local_xyz": local_xyz.tolist(),
                }
            },
        )

    def _handle_act(self) -> StateResult:
        """ACT: Call Kimodo with foot constraint to step on the target."""
        if self._target_local_xyz is None:
            return StateResult(
                status="fail",
                next_state=State.FAIL,
                message="No local target coordinate",
            )

        self._say(f"Stepping on the {self._target}!")

        foot_xyz = self._target_local_xyz.tolist()
        payload = {
            "prompt": "A person steps forward and stomps with right foot",
            "duration": 3.0,
            "diffusion_steps": 50,
            "foot_target_xyz": foot_xyz,
        }

        response = httpx.post(
            f"{KIMODO_URL}/generate",
            json=payload,
            timeout=120.0,
        )
        response.raise_for_status()
        kimodo_result = response.json()

        return StateResult(
            status="ok",
            next_state=State.DONE,
            payload={
                "act_kimodo_result": kimodo_result,
                "foot_target_xyz": foot_xyz,
            },
            message="Trajectory generated — pass to RL policy for execution",
        )
