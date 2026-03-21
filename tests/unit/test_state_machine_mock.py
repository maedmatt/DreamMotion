"""Unit tests for TreasureHuntStateMachine using mock dependencies.

No hardware required. Run with:

    uv run python tests/unit/test_state_machine_mock.py

Tests the full LOOK → MOVE → LOOK_AGAIN → ACT → DONE pipeline using
fake camera, detector, transforms, odometry, and SDK objects.
Also tests failure paths (object not found, depth invalid, walk timeout).
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    pass

from g1.state_machine.machine import TreasureHuntStateMachine
from g1.state_machine.types import State
from g1.transforms.odometry import OdomState
from g1.vision.camera import CameraFrame, Intrinsics
from g1.vision.detector import Detection

import numpy as np


def _pass(name: str) -> None:
    print(f"  ✓ {name}")


def _fail(name: str, detail: str) -> None:
    print(f"  ✗ {name}: {detail}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_INTRINSICS = Intrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480)
_BLANK_COLOR = np.zeros((480, 640, 3), dtype=np.uint8)
_BLANK_DEPTH = np.ones((480, 640), dtype=np.float32) * 1.5  # 1.5m everywhere


def _fake_frame() -> CameraFrame:
    return CameraFrame(color=_BLANK_COLOR, depth=_BLANK_DEPTH, intrinsics=_INTRINSICS)


def _fake_detection(
    depth_valid: bool = True,
    point: tuple[float, float, float] = (0.5, 0.0, 1.5),
) -> Detection:
    return Detection(
        label="box",
        confidence=0.92,
        bbox_xyxy=(100, 100, 200, 200),
        center_uv=(150, 150),
        point_camera=point if depth_valid else None,
        depth_valid=depth_valid,
    )


def _fake_odom_state(x: float = 0.0, y: float = 0.0) -> OdomState:
    return OdomState(x=x, y=y, z=0.0,
                     quat_x=0.0, quat_y=0.0, quat_z=0.0, quat_w=1.0,
                     vx=0.0, vy=0.0, vyaw=0.0)


def _fake_odometry() -> MagicMock:
    odom = MagicMock()
    odom.get_state.return_value = _fake_odom_state()
    odom.velocity_magnitude.return_value = 0.0
    return odom


def _build_machine(
    camera: object,
    detector: object,
    transforms: object,
    odometry: object,
    sdk: object,
    walk_method: str = "SDK",
) -> TreasureHuntStateMachine:
    spoken: list[str] = []
    return TreasureHuntStateMachine(
        target_object="box",
        camera=camera,  # type: ignore[arg-type]
        detector=detector,  # type: ignore[arg-type]
        transforms=transforms,  # type: ignore[arg-type]
        odometry=odometry,  # type: ignore[arg-type]
        sdk_controller=sdk,  # type: ignore[arg-type]
        say=spoken.append,
        walk_method=walk_method,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_happy_path_sdk() -> None:
    """LOOK → MOVE (SDK) → LOOK_AGAIN → ACT → DONE."""
    camera = MagicMock()
    camera.capture.return_value = _fake_frame()

    detector = MagicMock()
    detector.detect.return_value = [_fake_detection()]

    transforms = MagicMock()
    transforms.transform_point_between.side_effect = lambda p, *_: np.array(p)

    odometry = MagicMock()
    odometry.get_state.return_value = _fake_odom_state()
    odometry.velocity_magnitude.return_value = 0.0

    sdk = MagicMock()
    sdk.walk_to_point.return_value = True

    with patch("g1.state_machine.machine.httpx.post") as mock_post:
        mock_post.return_value.json.return_value = {"qpos": [[0.0] * 36]}
        mock_post.return_value.raise_for_status = MagicMock()

        machine = _build_machine(camera, detector, transforms, odometry, sdk)
        result = machine.run()

    if result["final_state"] != "DONE":
        _fail("happy_path_sdk", f"expected DONE, got {result['final_state']}")
    if result["target_world_xyz"] is None:
        _fail("happy_path_sdk", "target_world_xyz is None")
    if result["target_local_xyz"] is None:
        _fail("happy_path_sdk", "target_local_xyz is None")
    _pass("happy path SDK walk → DONE")


def test_happy_path_kimodo_walk() -> None:
    """Same as above but with walk_method='KIMODO'."""
    camera = MagicMock()
    camera.capture.return_value = _fake_frame()

    detector = MagicMock()
    detector.detect.return_value = [_fake_detection()]

    transforms = MagicMock()
    transforms.transform_point_between.side_effect = lambda p, *_: np.array(p)

    odometry = MagicMock()
    odometry.get_state.return_value = _fake_odom_state()
    odometry.velocity_magnitude.return_value = 0.0

    sdk = MagicMock()

    with patch("g1.state_machine.machine.walk_to_point_kimodo") as mock_walk, \
         patch("g1.state_machine.machine.httpx.post") as mock_post:
        mock_walk.return_value = {"status": "trajectory_generated"}
        mock_post.return_value.json.return_value = {"qpos": [[0.0] * 36]}
        mock_post.return_value.raise_for_status = MagicMock()

        machine = _build_machine(camera, detector, transforms, odometry, sdk, walk_method="KIMODO")
        result = machine.run()

    if result["final_state"] != "DONE":
        _fail("happy_path_kimodo", f"expected DONE, got {result['final_state']}")
    _pass("happy path KIMODO walk → DONE")


# ---------------------------------------------------------------------------
# LOOK failure paths
# ---------------------------------------------------------------------------


def test_look_no_detections_retries_then_fails() -> None:
    """LOOK returns no detections 3 times → FAIL."""
    camera = MagicMock()
    camera.capture.return_value = _fake_frame()

    detector = MagicMock()
    detector.detect.return_value = []  # always empty

    machine = _build_machine(camera, detector, MagicMock(), _fake_odometry(), MagicMock())
    result = machine.run()

    if result["final_state"] != "FAIL":
        _fail("look_no_detections", f"expected FAIL, got {result['final_state']}")
    if detector.detect.call_count != 3:
        _fail("look_no_detections", f"expected 3 retries, got {detector.detect.call_count}")
    _pass("LOOK: no detections → retries 3× → FAIL")


def test_look_invalid_depth_retries() -> None:
    """LOOK gets detection but depth is invalid → retry."""
    camera = MagicMock()
    camera.capture.return_value = _fake_frame()

    detector = MagicMock()
    detector.detect.return_value = [_fake_detection(depth_valid=False)]

    machine = _build_machine(camera, detector, MagicMock(), _fake_odometry(), MagicMock())
    result = machine.run()

    if result["final_state"] != "FAIL":
        _fail("look_invalid_depth", f"expected FAIL, got {result['final_state']}")
    _pass("LOOK: invalid depth → retries → FAIL")


# ---------------------------------------------------------------------------
# MOVE failure paths
# ---------------------------------------------------------------------------


def test_move_sdk_timeout_fails() -> None:
    """MOVE walk returns False (timeout) → FAIL."""
    camera = MagicMock()
    camera.capture.return_value = _fake_frame()

    detector = MagicMock()
    detector.detect.return_value = [_fake_detection()]

    transforms = MagicMock()
    transforms.transform_point_between.side_effect = lambda p, *_: np.array(p)

    odometry = MagicMock()
    odometry.get_state.return_value = _fake_odom_state()
    odometry.velocity_magnitude.return_value = 0.0

    sdk = MagicMock()
    sdk.walk_to_point.return_value = False  # timeout

    machine = _build_machine(camera, detector, transforms, odometry, sdk)
    result = machine.run()

    if result["final_state"] != "FAIL":
        _fail("move_timeout", f"expected FAIL, got {result['final_state']}")
    _pass("MOVE: SDK walk timeout → FAIL")


# ---------------------------------------------------------------------------
# LOOK_AGAIN paths
# ---------------------------------------------------------------------------


def test_look_again_depth_invalid_steps_back() -> None:
    """LOOK_AGAIN: depth invalid → step back → retry → eventually find."""
    camera = MagicMock()
    camera.capture.return_value = _fake_frame()

    call_count = {"n": 0}

    def detect_side_effect(frame: object, classes: object) -> list[Detection]:
        call_count["n"] += 1
        if call_count["n"] <= 1:
            # First LOOK call succeeds
            return [_fake_detection(depth_valid=True)]
        if call_count["n"] == 2:
            # First LOOK_AGAIN fails (too close)
            return [_fake_detection(depth_valid=False)]
        # Second LOOK_AGAIN succeeds
        return [_fake_detection(depth_valid=True)]

    detector = MagicMock()
    detector.detect.side_effect = detect_side_effect

    transforms = MagicMock()
    transforms.transform_point_between.side_effect = lambda p, *_: np.array(p)

    odometry = MagicMock()
    odometry.get_state.return_value = _fake_odom_state()
    odometry.velocity_magnitude.return_value = 0.0

    sdk = MagicMock()
    sdk.walk_to_point.return_value = True

    with patch("g1.state_machine.machine.httpx.post") as mock_post:
        mock_post.return_value.json.return_value = {"qpos": [[0.0] * 36]}
        mock_post.return_value.raise_for_status = MagicMock()

        machine = _build_machine(camera, detector, transforms, odometry, sdk)
        result = machine.run()

    if result["final_state"] != "DONE":
        _fail("look_again_step_back", f"expected DONE, got {result['final_state']}")
    sdk.step_backward.assert_called_once_with(0.2)
    _pass("LOOK_AGAIN: depth invalid → step_backward → retry → DONE")


def test_look_again_object_lost_restarts() -> None:
    """LOOK_AGAIN: object totally lost → retry goes back to LOOK."""
    camera = MagicMock()
    camera.capture.return_value = _fake_frame()

    call_count = {"n": 0}

    def detect_side_effect(frame: object, classes: object) -> list[Detection]:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return [_fake_detection()]  # LOOK succeeds
        return []  # LOOK_AGAIN always fails

    detector = MagicMock()
    detector.detect.side_effect = detect_side_effect

    transforms = MagicMock()
    transforms.transform_point_between.side_effect = lambda p, *_: np.array(p)

    odometry = MagicMock()
    odometry.get_state.return_value = _fake_odom_state()
    odometry.velocity_magnitude.return_value = 0.0

    sdk = MagicMock()
    sdk.walk_to_point.return_value = True

    machine = _build_machine(camera, detector, transforms, odometry, sdk)
    result = machine.run()

    # Should fail after exhausting retries across LOOK_AGAIN (3×) and LOOK re-runs
    if result["final_state"] != "FAIL":
        _fail("look_again_lost", f"expected FAIL, got {result['final_state']}")
    _pass("LOOK_AGAIN: object lost → restarts LOOK → eventually FAIL")


# ---------------------------------------------------------------------------
# State machine narration
# ---------------------------------------------------------------------------


def test_say_called_each_state() -> None:
    """The say callback should be called at least once per state."""
    camera = MagicMock()
    camera.capture.return_value = _fake_frame()

    detector = MagicMock()
    detector.detect.return_value = [_fake_detection()]

    transforms = MagicMock()
    transforms.transform_point_between.side_effect = lambda p, *_: np.array(p)

    odometry = MagicMock()
    odometry.get_state.return_value = _fake_odom_state()
    odometry.velocity_magnitude.return_value = 0.0

    sdk = MagicMock()
    sdk.walk_to_point.return_value = True

    spoken: list[str] = []

    with patch("g1.state_machine.machine.httpx.post") as mock_post:
        mock_post.return_value.json.return_value = {"qpos": [[0.0] * 36]}
        mock_post.return_value.raise_for_status = MagicMock()

        machine = TreasureHuntStateMachine(
            target_object="box",
            camera=camera,  # type: ignore[arg-type]
            detector=detector,  # type: ignore[arg-type]
            transforms=transforms,  # type: ignore[arg-type]
            odometry=odometry,  # type: ignore[arg-type]
            sdk_controller=sdk,  # type: ignore[arg-type]
            say=spoken.append,
        )
        machine.run()

    if len(spoken) < 4:
        _fail("say_called", f"expected ≥4 say calls, got {len(spoken)}: {spoken}")
    _pass(f"say callback called {len(spoken)} times across states")


if __name__ == "__main__":
    print("=== state_machine mock unit tests ===\n")

    test_happy_path_sdk()
    test_happy_path_kimodo_walk()
    test_look_no_detections_retries_then_fails()
    test_look_invalid_depth_retries()
    test_move_sdk_timeout_fails()
    test_look_again_depth_invalid_steps_back()
    test_look_again_object_lost_restarts()
    test_say_called_each_state()

    print("\nAll tests passed.")
