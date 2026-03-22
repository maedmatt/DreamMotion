from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np

from g1.stance.controller import stabilize_for_vision
from g1.state_machine.types import State, StateResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from g1.transforms.service import TransformService
    from g1.vision.camera import OakCamera
    from g1.vision.detector import ObjectDetector

logger = logging.getLogger(__name__)

_MAX_RETRIES_PER_STATE = 3
_DEFAULT_ROOT_HEIGHT = 0.75

Action = Literal["locate", "walk_to", "step_on", "pick_up", "point_at"]

_ACTION_PROMPTS: dict[Action, str] = {
    "walk_to": "Walk to the target object",
    "step_on": "Walk to the target object and stomp on it with the right foot",
    "pick_up": "Walk to the target object and pick it up with the right hand",
    "point_at": "Point at the target object with the right arm",
    "locate": "Locate the target object",
}


class TreasureHuntStateMachine:
    """Look-first FSM for object-directed constrained motion generation.

    The gg/vision branch used Unitree SDK locomotion for the MOVE state. On this
    branch we preserve the same high-level flow, but movement is generated
    through Kimodo so the existing deploy watcher can execute it.
    """

    def __init__(
        self,
        target_object: str,
        camera: OakCamera,
        detector: ObjectDetector,
        transforms: TransformService,
        motion_generator: Callable[..., dict[str, object]],
        say: Callable[[str], None],
        action: Action = "walk_to",
    ) -> None:
        self._target = target_object
        self._camera = camera
        self._detector = detector
        self._tf = transforms
        self._motion_generator = motion_generator
        self._say = say
        self._action = action
        self._target_local_xyz: np.ndarray | None = None

    def _stabilize(self) -> None:
        stabilize_for_vision(velocity_func=None)

    def run(self) -> dict[str, object]:
        state = State.LOOK
        retries: dict[State, int] = {}
        result_payload: dict[str, object] = {}

        handlers = {
            State.LOOK: self._handle_look,
            State.MOVE: self._handle_move,
            State.POINT: self._handle_point,
            State.ACT: self._handle_act,
        }

        while state not in (State.DONE, State.FAIL):
            handler = handlers.get(state)
            if handler is None:
                logger.error("No handler for state %s", state.name)
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
                state = result.next_state
                continue

            if result.status == "fail":
                logger.warning("State %s failed: %s", state.name, result.message)
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

    def _collect_points(self, frames: int = 5) -> list[tuple[float, float, float]]:
        points: list[tuple[float, float, float]] = []
        for _ in range(frames):
            frame = self._camera.capture()
            detections = self._detector.detect(frame, [self._target])
            if (
                detections
                and detections[0].depth_valid
                and detections[0].point_camera is not None
            ):
                points.append(detections[0].point_camera)
        return points

    def _detect_target(self) -> StateResult:
        self._stabilize()
        points = self._collect_points()
        if not points:
            return StateResult(
                status="retry",
                next_state=State.LOOK,
                message=f"No valid {self._target} detection across 5 frames",
            )

        avg_point = (
            float(np.median([p[0] for p in points])),
            float(np.median([p[1] for p in points])),
            float(np.median([p[2] for p in points])),
        )
        base_xyz = self._tf.transform_point_between(avg_point, "camera", "base")
        self._target_local_xyz = base_xyz
        return StateResult(
            status="ok",
            next_state=State.DONE,
            payload={
                "look_detection": {
                    "label": self._target,
                    "frames_used": len(points),
                    "base_xyz": base_xyz.tolist(),
                }
            },
        )

    def _target_final_root_pos(self) -> list[float]:
        if self._target_local_xyz is None:
            raise RuntimeError("No target coordinate available")
        return [
            float(self._target_local_xyz[0]),
            float(self._target_local_xyz[1]),
            _DEFAULT_ROOT_HEIGHT,
        ]

    def _run_motion_generation(
        self,
        *,
        description: str,
        final_root_pos: list[float],
    ) -> StateResult:
        try:
            result = self._motion_generator(
                description,
                final_root_pos=final_root_pos,
            )
        except Exception as exc:
            logger.exception("Constrained motion generation failed")
            return StateResult(
                status="fail",
                next_state=State.FAIL,
                message=f"Motion generation failed: {exc}",
            )

        payload: dict[str, object] = {
            "motion_request": {
                "description": description,
                "final_root_pos": list(final_root_pos),
            }
        }
        if isinstance(result, dict):
            payload.update(result)
        return StateResult(
            status="ok",
            next_state=State.DONE,
            payload=payload,
        )

    def _handle_look(self) -> StateResult:
        self._say(f"Looking for the {self._target}.")
        result = self._detect_target()
        if result.status != "ok":
            return result

        if self._target_local_xyz is None:
            return StateResult(
                status="fail",
                next_state=State.FAIL,
                message="Detection succeeded but target_local_xyz is missing",
            )

        x, y, _ = self._target_local_xyz.tolist()
        self._say(
            f"Found the {self._target}. It is about {x:.2f} metres ahead and {y:.2f} metres to the side."
        )

        if self._action == "locate":
            return StateResult(
                status="ok",
                next_state=State.DONE,
                payload=result.payload,
            )
        if self._action == "walk_to":
            return StateResult(
                status="ok",
                next_state=State.MOVE,
                payload=result.payload,
            )
        if self._action == "point_at":
            return StateResult(
                status="ok",
                next_state=State.POINT,
                payload=result.payload,
            )
        return StateResult(
            status="ok",
            next_state=State.ACT,
            payload=result.payload,
        )

    def _handle_move(self) -> StateResult:
        final_root_pos = self._target_final_root_pos()
        description = f"Walk to the {self._target}"
        self._say(f"Generating a walking motion to the {self._target} now.")
        return self._run_motion_generation(
            description=description,
            final_root_pos=final_root_pos,
        )

    def _handle_point(self) -> StateResult:
        description = _ACTION_PROMPTS["point_at"].replace("target object", self._target)
        self._say(f"Generating a pointing motion for the {self._target}.")
        return self._run_motion_generation(
            description=description,
            final_root_pos=[0.0, 0.0, _DEFAULT_ROOT_HEIGHT],
        )

    def _handle_act(self) -> StateResult:
        final_root_pos = self._target_final_root_pos()
        description = _ACTION_PROMPTS[self._action].replace("target object", self._target)
        self._say(f"Generating an interaction motion for the {self._target}.")
        return self._run_motion_generation(
            description=description,
            final_root_pos=final_root_pos,
        )
