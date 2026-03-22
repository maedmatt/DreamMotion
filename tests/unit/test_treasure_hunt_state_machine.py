from __future__ import annotations

import sys

import numpy as np

from g1.state_machine.machine import TreasureHuntStateMachine


def _pass(name: str) -> None:
    print(f"  ok - {name}")


def _fail(name: str, detail: str) -> None:
    print(f"  not ok - {name}: {detail}")
    sys.exit(1)


class _FakeCamera:
    def capture(self) -> object:
        return object()


class _Detection:
    def __init__(self, point: tuple[float, float, float]) -> None:
        self.depth_valid = True
        self.point_camera = point


class _FakeDetector:
    def __init__(self, point: tuple[float, float, float]) -> None:
        self._point = point

    def detect(self, frame: object, classes: list[str]) -> list[_Detection]:
        return [_Detection(self._point)]


class _FakeTransforms:
    def __init__(self, result: tuple[float, float, float]) -> None:
        self._result = np.array(result, dtype=float)

    def transform_point_between(
        self,
        point: tuple[float, float, float],
        from_frame: str,
        to_frame: str,
    ) -> np.ndarray:
        assert from_frame == "camera"
        assert to_frame == "base"
        return self._result.copy()


def test_walk_to_uses_detected_target_as_final_root_pos() -> None:
    calls: list[dict[str, object]] = []

    def motion_generator(description: str, **kwargs: object) -> dict[str, object]:
        calls.append({"description": description, **kwargs})
        return {"motions": [{"prompt": description, "duration": 2.0}]}

    machine = TreasureHuntStateMachine(
        target_object="bottle",
        camera=_FakeCamera(),
        detector=_FakeDetector((0.3, 0.1, 1.2)),
        transforms=_FakeTransforms((1.25, -0.3, 0.6)),
        motion_generator=motion_generator,
        say=lambda _text: None,
        action="walk_to",
    )
    result = machine.run()

    if result["final_state"] != "DONE":
        _fail("walk_to final state", str(result))
    if len(calls) != 1:
        _fail("walk_to motion count", f"expected 1 call, got {len(calls)}")

    call = calls[0]
    if call["description"] != "Walk to the bottle":
        _fail("walk_to description", str(call["description"]))
    if call.get("final_root_pos") != [1.25, -0.3, 0.75]:
        _fail("walk_to root constraint", str(call.get("final_root_pos")))

    _pass("walk_to uses detected target as final_root_pos")


def test_locate_does_not_generate_motion() -> None:
    calls: list[dict[str, object]] = []

    def motion_generator(description: str, **kwargs: object) -> dict[str, object]:
        calls.append({"description": description, **kwargs})
        return {}

    machine = TreasureHuntStateMachine(
        target_object="bottle",
        camera=_FakeCamera(),
        detector=_FakeDetector((0.3, 0.1, 1.2)),
        transforms=_FakeTransforms((0.8, 0.05, 0.6)),
        motion_generator=motion_generator,
        say=lambda _text: None,
        action="locate",
    )
    result = machine.run()

    if result["final_state"] != "DONE":
        _fail("locate final state", str(result))
    if calls:
        _fail("locate motion count", f"expected 0 calls, got {len(calls)}")

    _pass("locate stops after detection without generating motion")


if __name__ == "__main__":
    test_walk_to_uses_detected_target_as_final_root_pos()
    test_locate_does_not_generate_motion()
    print("all tests passed")
