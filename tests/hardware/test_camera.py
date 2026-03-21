"""Hardware test: OAK-D camera capture and object detection.

Prerequisites:
  - OAK-D connected via USB to this machine
  - depthai and inference packages installed (uv sync)
  - A box (or any object) visible to the camera

Run with:

    uv run python tests/hardware/test_camera.py

Optional: point camera at a box for the detection test.

What this verifies:
  1. OAK-D pipeline starts and streams frames
  2. Aligned depth map has the expected shape and valid values
  3. Deprojection returns geometrically plausible 3D coordinates
  4. YOLO-World detects objects from a text prompt
  5. Min-depth guard trips correctly when object is < 0.35m away
"""

from __future__ import annotations

import sys
import time

import numpy as np


def _pass(name: str) -> None:
    print(f"  ✓ {name}")


def _fail(name: str, detail: str) -> None:
    print(f"  ✗ {name}: {detail}")
    sys.exit(1)


def _warn(name: str, detail: str) -> None:
    print(f"  ⚠ {name}: {detail}")


def test_camera_starts() -> None:
    print("Starting OAK-D pipeline (takes ~3s)...")
    from g1.vision.camera import get_camera

    cam = get_camera()
    _pass("OAK-D pipeline started")
    return cam


def test_frame_shapes(cam: object) -> None:
    from g1.vision.camera import OakCamera

    assert isinstance(cam, OakCamera)
    frame = cam.capture()

    if frame.color.ndim != 3 or frame.color.shape[2] != 3:
        _fail("frame_shapes", f"color shape: {frame.color.shape}")
    if frame.depth.ndim != 2:
        _fail("frame_shapes", f"depth shape: {frame.depth.shape}")
    if frame.color.shape[:2] != frame.depth.shape:
        _fail("frame_shapes",
              f"color {frame.color.shape[:2]} != depth {frame.depth.shape}")

    _pass(f"frame shapes: color={frame.color.shape}, depth={frame.depth.shape}")
    return frame


def test_depth_values(frame: object) -> None:
    from g1.vision.camera import CameraFrame

    assert isinstance(frame, CameraFrame)
    valid = frame.depth[frame.depth > 0]
    if len(valid) == 0:
        _fail("depth_values", "no valid depth pixels at all — check stereo baseline")

    d_min = float(valid.min())
    d_max = float(valid.max())
    d_median = float(np.median(valid))
    coverage = 100 * len(valid) / frame.depth.size

    print(f"    Depth stats: min={d_min:.2f}m  max={d_max:.2f}m  "
          f"median={d_median:.2f}m  coverage={coverage:.0f}%")

    if d_median < 0.3 or d_median > 10.0:
        _warn("depth_values", f"unusual median depth {d_median:.2f}m — scene may be empty")
    else:
        _pass("depth values in plausible range")
    return frame


def test_deproject(cam: object, frame: object) -> None:
    from g1.vision.camera import CameraFrame, OakCamera

    assert isinstance(cam, OakCamera)
    assert isinstance(frame, CameraFrame)

    cx = frame.intrinsics.width // 2
    cy = frame.intrinsics.height // 2
    d = float(frame.depth[cy, cx])

    if d < 0.01:
        _warn("deproject", "center pixel has no depth — skipping deproject check")
        return

    x, y, z = cam.deproject(cx, cy, d)
    print(f"    Center pixel ({cx},{cy}) depth={d:.3f}m → 3D=({x:.3f}, {y:.3f}, {z:.3f})")

    if abs(z - d) > 0.01:
        _fail("deproject", f"z={z:.3f} doesn't match depth={d:.3f}")
    _pass("deproject: Z matches depth")


def test_detection_with_object(cam: object) -> None:
    """Point the camera at a box before running this."""
    from g1.vision.camera import OakCamera
    from g1.vision.detector import get_detector

    assert isinstance(cam, OakCamera)
    detector = get_detector()

    print("    Capturing frame for detection...")
    frame = cam.capture()
    detections = detector.detect(frame, ["box"])

    if not detections:
        _warn("detection", "no 'box' detected — is a box in the camera view?")
        return

    best = detections[0]
    print(f"    Best detection: label={best.label}  conf={best.confidence:.2f}"
          f"  depth_valid={best.depth_valid}  3D={best.point_camera}")

    if best.confidence < 0.2:
        _warn("detection", f"low confidence {best.confidence:.2f}")
    else:
        _pass(f"detected '{best.label}' conf={best.confidence:.2f}")

    if best.depth_valid and best.point_camera is not None:
        x, y, z = best.point_camera
        if z <= 0 or z > 10:
            _fail("detection_3d", f"implausible Z={z:.3f}")
        _pass(f"3D position: ({x:.3f}, {y:.3f}, {z:.3f}) m")
    else:
        _warn("detection", "depth_valid=False — object may be < 0.35m from camera")


def test_min_depth_guard() -> None:
    """
    To test this manually:
      1. Hold the box/hand < 0.35m from the camera lens.
      2. Run this test.
      3. Verify depth_valid=False is printed.
    """
    from g1.vision.camera import get_camera
    from g1.vision.detector import get_detector

    cam = get_camera()
    detector = get_detector()

    print("    >>> Hold an object < 35cm from the lens, then press Enter <<<")
    input()
    frame = cam.capture()
    detections = detector.detect(frame, ["object", "hand", "box"])

    if not detections:
        _warn("min_depth_guard", "nothing detected at close range")
        return

    best = detections[0]
    print(f"    depth_valid={best.depth_valid}  3D={best.point_camera}")
    if not best.depth_valid:
        _pass("min_depth guard correctly set depth_valid=False at < 0.35m")
    else:
        _warn("min_depth_guard",
              f"depth_valid=True with Z={best.point_camera[2] if best.point_camera else '?'}m "
              f"— was object truly < 0.35m from lens?")


if __name__ == "__main__":
    print("=== hardware/test_camera.py ===")
    print("Prerequisites: OAK-D connected via USB\n")

    cam = test_camera_starts()
    frame = test_frame_shapes(cam)
    test_depth_values(frame)
    test_deproject(cam, frame)
    test_detection_with_object(cam)

    print("\nOptional interactive test:")
    run_min_depth = input("Run min-depth guard test? [y/N] ").strip().lower()
    if run_min_depth == "y":
        test_min_depth_guard()

    print("\nDone.")
