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

# Kimodo prompt and target field per action type.
_ACT_CONFIG: dict[str, tuple[str, str]] = {
    "step_on": (
        "A person steps forward and stomps with right foot",
        "foot_target_xyz",
    ),
    "pick_up": (
        "A person crouches and picks up an object with the right hand",
        "hand_target_xyz",
    ),
}

Action = Literal["locate", "walk_to", "step_on", "pick_up"]


class TreasureHuntStateMachine:
    """Look-Move-Look-Act pipeline for finding and interacting with objects.

    Args:
        target_object: Natural language description of the target (e.g. "red box").
        camera: OakCamera instance for RGB+Depth capture.
        detector: ObjectDetector instance for zero-shot detection.
        transforms: TransformService for coordinate frame conversions.
        odometry: Optional OdometrySubscriber. When None the pipeline operates
            entirely in base_link frame (no absolute world position needed).
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
        odometry: OdometrySubscriber | None,
        sdk_controller: SdkLocomotionController,
        say: Callable[[str], None],
        walk_method: Literal["SDK", "KIMODO"] = "SDK",
        action: Action = "step_on",
    ) -> None:
        self._target = target_object
        self._camera = camera
        self._detector = detector
        self._tf = transforms
        self._odom = odometry
        self._sdk = sdk_controller
        self._say = say
        self._walk_method = walk_method
        self._action = action

        # State shared between handlers
        self._target_world_xyz: np.ndarray | None = None
        self._target_local_xyz: np.ndarray | None = None

    def _stabilize(self) -> None:
        """Stabilize for vision — velocity-based if odometry available, else sleep."""
        stabilize_for_vision(
            velocity_func=self._odom.velocity_magnitude if self._odom else None,
        )

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
        """LOOK: Stabilize, detect target, record position in base_link frame.

        When odometry is available the target is also projected to world (odom)
        frame for absolute navigation. Without odometry we work entirely in
        base_link — the base_link coordinates are used directly as relative
        offsets for the MOVE state.
        """
        self._say(f"Looking for the {self._target}...")
        self._stabilize()

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

        # Always compute base_link position (no odometry needed)
        base_xyz = self._tf.transform_point_between(
            best.point_camera, "camera", "base"
        )
        self._target_local_xyz = base_xyz

        # Optionally compute world position when odometry is available
        if self._odom is not None:
            world_xyz = self._tf.transform_point_between(
                best.point_camera, "camera", "world"
            )
            self._target_world_xyz = world_xyz
        else:
            # Without odometry: treat base_link offset as the navigation target
            self._target_world_xyz = base_xyz

        detection_payload = {
            "look_detection": {
                "label": best.label,
                "confidence": best.confidence,
                "base_xyz": base_xyz.tolist(),
                "world_xyz": self._target_world_xyz.tolist(),
            }
        }

        if self._action == "locate":
            self._say(
                f"Found the {self._target}! "
                f"It is {base_xyz[0]:.2f} metres ahead of me."
            )
            return StateResult(
                status="ok",
                next_state=State.DONE,
                payload=detection_payload,
            )

        self._say(f"Found the {self._target}! Walking closer.")
        return StateResult(
            status="ok",
            next_state=State.MOVE,
            payload=detection_payload,
        )

    def _handle_move(self) -> StateResult:
        """MOVE: Walk toward saved coordinate, stopping 0.5m short.

        Three modes:
          SDK + odometry  — closed-loop P-controller to world coordinates.
          SDK, no odometry — open-loop: turn + timed walk using base_link offset.
          KIMODO          — trajectory generation via Kimodo API.
        """
        if self._target_local_xyz is None or self._target_world_xyz is None:
            return StateResult(
                status="fail", next_state=State.FAIL, message="No target coordinate"
            )

        if self._walk_method == "SDK":
            if self._odom is not None:
                # Closed-loop P-controller (full odometry available)
                success = self._sdk.walk_to_point(
                    target_x=float(self._target_world_xyz[0]),
                    target_y=float(self._target_world_xyz[1]),
                    current_odom_func=self._odom.get_state,
                    stop_short_m=0.5,
                )
                if not success:
                    return StateResult(
                        status="fail", next_state=State.FAIL, message="Walk timed out"
                    )
            else:
                # Open-loop: use base_link offset from camera
                import math  # noqa: PLC0415

                bx = float(self._target_local_xyz[0])
                by = float(self._target_local_xyz[1])
                distance = math.sqrt(bx * bx + by * by)
                yaw = math.atan2(by, bx)
                walk_dist = max(0.0, distance - 0.5)  # stop 0.5m short
                self._sdk.walk_forward_distance(walk_dist, yaw_rad=yaw)
        else:
            current_x = float(self._odom.get_state().x) if self._odom else 0.0
            current_y = float(self._odom.get_state().y) if self._odom else 0.0
            result = walk_to_point_kimodo(
                target_x=float(self._target_world_xyz[0]),
                target_y=float(self._target_world_xyz[1]),
                current_x=current_x,
                current_y=current_y,
                stop_short_m=0.5,
            )
            if result.get("status") == "already_close":
                pass  # No need to walk

        self._stabilize()

        if self._action == "walk_to":
            self._say(f"Arrived near the {self._target}.")
            return StateResult(status="ok", next_state=State.DONE)

        return StateResult(status="ok", next_state=State.LOOK_AGAIN)

    def _handle_look_again(self) -> StateResult:
        """LOOK_AGAIN: Precision vision update at close range."""
        self._say("Getting a closer look...")
        self._stabilize()

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

        prompt, target_field = _ACT_CONFIG.get(
            self._action, _ACT_CONFIG["step_on"]
        )
        self._say(f"{prompt.capitalize()} targeting the {self._target}!")

        target_xyz = self._target_local_xyz.tolist()
        payload = {
            "prompt": prompt,
            "duration": 3.0,
            "diffusion_steps": 50,
            target_field: target_xyz,
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
                target_field: target_xyz,
            },
            message="Trajectory generated — pass to RL policy for execution",
        )
