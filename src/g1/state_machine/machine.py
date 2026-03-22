from __future__ import annotations

import logging
import math
import os
from typing import TYPE_CHECKING, Literal

import httpx
import numpy as np

from g1.stance.controller import stabilize_for_vision
from g1.state_machine.types import State, StateResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from g1.locomotion.sdk_controller import SdkLocomotionController
    from g1.transforms.service import TransformService
    from g1.vision.camera import OakCamera
    from g1.vision.detector import ObjectDetector

logger = logging.getLogger(__name__)

KIMODO_URL = os.environ.get("KIMODO_URL", "http://localhost:8420")

_MAX_RETRIES_PER_STATE = 3

# Walk-to is open-loop, so we keep the first approach conservative, then allow
# one short corrective move if the close-range recheck still says we are far.
_WALK_TO_STOP_SHORT_M = 0.2
_DEFAULT_STOP_SHORT_M = 0.5
_WALK_TO_CONFIRM_DISTANCE_M = 0.3
_MAX_MOVE_PASSES = 2

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
        sdk_controller: SDK locomotion controller for open-loop walking.
        say: Callback to make the robot speak.
        action: One of "locate", "walk_to", "step_on", "pick_up".
    """

    def __init__(
        self,
        target_object: str,
        camera: OakCamera,
        detector: ObjectDetector,
        transforms: TransformService,
        sdk_controller: SdkLocomotionController | None,
        say: Callable[[str], None],
        action: Action = "step_on",
    ) -> None:
        self._target = target_object
        self._camera = camera
        self._detector = detector
        self._tf = transforms
        self._sdk = sdk_controller
        self._say = say
        self._action = action

        # State shared between handlers
        self._target_local_xyz: np.ndarray | None = None
        self._move_passes = 0

    def _stabilize(self) -> None:
        stabilize_for_vision(velocity_func=None)

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
        """LOOK: Stabilize, detect target, record position in base_link frame."""
        self._say(f"Looking for the {self._target}...")
        self._stabilize()

        points: list[tuple[float, float, float]] = []
        for _ in range(5):
            frame = self._camera.capture()
            detections = self._detector.detect(frame, [self._target])
            if (
                detections
                and detections[0].depth_valid
                and detections[0].point_camera is not None
            ):
                points.append(detections[0].point_camera)

        if not points:
            return StateResult(
                status="retry",
                next_state=State.LOOK,
                message=f"No valid {self._target} detection across 5 frames",
            )

        avg_point: tuple[float, float, float] = (
            float(np.median([p[0] for p in points])),
            float(np.median([p[1] for p in points])),
            float(np.median([p[2] for p in points])),
        )

        base_xyz = self._tf.transform_point_between(avg_point, "camera", "base")
        self._target_local_xyz = base_xyz

        detection_payload: dict[str, object] = {
            "look_detection": {
                "label": self._target,
                "frames_used": len(points),
                "base_xyz": base_xyz.tolist(),
            }
        }

        if self._action == "locate":
            self._say(
                f"Found the {self._target}! It is {base_xyz[0]:.2f} metres ahead of me."
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
        """MOVE: Open-loop walk toward target, stopping 0.5m short."""
        if self._target_local_xyz is None:
            return StateResult(
                status="fail", next_state=State.FAIL, message="No target coordinate"
            )
        if self._sdk is None:
            return StateResult(
                status="fail",
                next_state=State.FAIL,
                message="Unitree locomotion SDK is unavailable for move action",
            )

        bx = float(self._target_local_xyz[0])
        by = float(self._target_local_xyz[1])
        distance = math.sqrt(bx * bx + by * by)
        yaw = math.atan2(by, bx)
        stop_short_m = (
            _WALK_TO_STOP_SHORT_M
            if self._action == "walk_to"
            else _DEFAULT_STOP_SHORT_M
        )
        walk_dist = max(0.0, distance - stop_short_m)
        self._move_passes += 1
        self._sdk.walk_forward_distance(walk_dist, yaw_rad=yaw)

        self._stabilize()
        return StateResult(status="ok", next_state=State.LOOK_AGAIN)

    def _handle_look_again(self) -> StateResult:
        """LOOK_AGAIN: Precision vision update at close range."""
        self._say("Getting a closer look...")
        self._stabilize()

        points: list[tuple[float, float, float]] = []
        for _ in range(5):
            frame = self._camera.capture()
            detections = self._detector.detect(frame, [self._target])
            if (
                detections
                and detections[0].depth_valid
                and detections[0].point_camera is not None
            ):
                points.append(detections[0].point_camera)

        if not points:
            # Check if any detection at all (depth just invalid = too close)
            frame = self._camera.capture()
            detections = self._detector.detect(frame, [self._target])
            if detections and not detections[0].depth_valid:
                if self._sdk is None:
                    return StateResult(
                        status="fail",
                        next_state=State.FAIL,
                        message="Unitree locomotion SDK is unavailable for close-range recovery",
                    )
                self._say("Too close, stepping back a little.")
                self._sdk.step_backward(0.2)
                return StateResult(
                    status="retry",
                    next_state=State.LOOK_AGAIN,
                    message="Depth invalid at close range, stepping back",
                )
            return StateResult(
                status="retry",
                next_state=State.LOOK,
                message="Lost target at close range — restarting search",
            )

        avg_point: tuple[float, float, float] = (
            float(np.median([p[0] for p in points])),
            float(np.median([p[1] for p in points])),
            float(np.median([p[2] for p in points])),
        )

        local_xyz = self._tf.transform_point_between(avg_point, "camera", "base")
        self._target_local_xyz = local_xyz

        detection_payload: dict[str, object] = {
            "look_again_detection": {
                "label": self._target,
                "frames_used": len(points),
                "local_xyz": local_xyz.tolist(),
            }
        }

        if self._action == "walk_to":
            close_distance = math.sqrt(
                float(local_xyz[0]) * float(local_xyz[0])
                + float(local_xyz[1]) * float(local_xyz[1])
            )
            if (
                close_distance > _WALK_TO_CONFIRM_DISTANCE_M
                and self._move_passes < _MAX_MOVE_PASSES
            ):
                self._say("Still a little far. Taking one more small step.")
                return StateResult(
                    status="ok",
                    next_state=State.MOVE,
                    payload=detection_payload,
                )

            self._say(f"Arrived near the {self._target}. Target confirmed.")
            return StateResult(
                status="ok", next_state=State.DONE, payload=detection_payload
            )

        self._say(f"Target locked! Preparing to act on the {self._target}.")
        return StateResult(status="ok", next_state=State.ACT, payload=detection_payload)

    def _handle_act(self) -> StateResult:
        """ACT: Call Kimodo with foot/hand constraint to interact with the target."""
        if self._target_local_xyz is None:
            return StateResult(
                status="fail",
                next_state=State.FAIL,
                message="No local target coordinate",
            )

        prompt, target_field = _ACT_CONFIG.get(self._action, _ACT_CONFIG["step_on"])
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
