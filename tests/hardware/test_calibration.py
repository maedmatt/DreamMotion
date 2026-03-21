"""Hardware test: camera_to_base transform calibration verification.

Prerequisites:
  - OAK-D mounted on the robot (fixed downward angle)
  - Robot powered on and stationary
  - UNITREE_NETWORK_INTERFACE set (for odometry)
  - A box placed at a KNOWN position relative to the robot

Run with:

    UNITREE_NETWORK_INTERFACE=eth0 uv run python tests/hardware/test_calibration.py

What this verifies:
  1. camera_to_base() matrix is geometrically sane (determinant = 1)
  2. A point 1m in front of the camera maps to a plausible base_link coordinate
  3. Full chain camera → base → world gives a world coordinate that moves
     correctly as the robot moves (sanity check with manual displacement)

Calibration procedure if numbers are off:
  - Place a box at exactly (0.5m forward, 0m sideways) from the robot base
  - Run this test and observe base_link output
  - Adjust env vars until it matches:
      G1_CAMERA_X_OFFSET   (forward offset of camera from base, default 0.05)
      G1_CAMERA_Y_OFFSET   (lateral offset, default 0.0)
      G1_CAMERA_Z_OFFSET   (height of camera above base, default 0.45)
      G1_CAMERA_PITCH      (downward angle in radians, default -0.35 ≈ -20°)
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


def test_camera_to_base_matrix() -> None:
    from g1.transforms.static import camera_to_base

    T = camera_to_base()
    print(f"  camera_to_base:\n{np.round(T, 4)}\n")

    # Must be a valid rigid transform: det(R) == 1
    R = T[:3, :3]
    det = np.linalg.det(R)
    if abs(det - 1.0) > 1e-5:
        _fail("camera_to_base_det", f"det(R)={det:.6f}, expected 1.0")
    _pass(f"rotation matrix det = {det:.6f} ≈ 1.0")

    # Camera Z (forward) should have a positive X component in robot frame
    cam_z_in_base = T[:3, 2]
    if cam_z_in_base[0] <= 0:
        _warn("camera_forward", f"camera forward in base = {cam_z_in_base} — X should be positive")
    else:
        _pass(f"camera forward in base_link: {np.round(cam_z_in_base, 3)}")

    # Translation should show camera above base (positive Z in robot frame)
    height = T[2, 3]
    if height < 0.2 or height > 1.5:
        _warn("camera_height", f"camera height = {height:.3f}m — expected 0.2–1.5m")
    else:
        _pass(f"camera height above base_link: {height:.3f}m")


def test_known_point_in_base_frame() -> None:
    """Place box at exactly 0.5m forward, 0m lateral from robot base.

    The base_link output should read approximately [0.5, 0.0, 0.0].
    """
    from g1.transforms.service import get_transform_service
    from g1.vision.camera import get_camera
    from g1.vision.detector import get_detector

    print("\n  >>> Place a box at exactly 0.5m directly in front of the robot <<<")
    print("  >>> Press Enter when ready <<<")
    input()

    cam = get_camera()
    detector = get_detector()
    tf = get_transform_service()

    frame = cam.capture()
    detections = detector.detect(frame, ["box"])

    if not detections:
        _fail("known_point", "no box detected — is it in view?")

    best = detections[0]
    if not best.depth_valid or best.point_camera is None:
        _fail("known_point", f"depth_valid={best.depth_valid} — move box farther from camera")

    cam_xyz = best.point_camera
    base_xyz = tf.transform_point_between(cam_xyz, "camera", "base")

    print(f"\n  Camera frame: ({cam_xyz[0]:.3f}, {cam_xyz[1]:.3f}, {cam_xyz[2]:.3f})")
    print(f"  Base frame:   ({base_xyz[0]:.3f}, {base_xyz[1]:.3f}, {base_xyz[2]:.3f})")
    print(f"\n  Expected:     (~0.5, ~0.0, ~0.0)")

    x_err = abs(base_xyz[0] - 0.5)
    y_err = abs(base_xyz[1])
    if x_err > 0.15:
        _warn("known_point_x", f"X error = {x_err:.3f}m — adjust G1_CAMERA_X_OFFSET or G1_CAMERA_PITCH")
    else:
        _pass(f"X in base_link = {base_xyz[0]:.3f}m (error {x_err:.3f}m)")

    if y_err > 0.1:
        _warn("known_point_y", f"Y error = {y_err:.3f}m — adjust G1_CAMERA_Y_OFFSET")
    else:
        _pass(f"Y in base_link = {base_xyz[1]:.3f}m (error {y_err:.3f}m)")


def test_world_frame_consistent() -> None:
    """World coordinate should be stable while robot is stationary."""
    from g1.transforms.service import get_transform_service
    from g1.vision.camera import get_camera
    from g1.vision.detector import get_detector

    print("\n  >>> Keep a box stationary in view for 3 seconds <<<")
    print("  >>> Press Enter to start <<<")
    input()

    cam = get_camera()
    detector = get_detector()
    tf = get_transform_service()

    print("  Capturing 5 samples over 3 seconds...")
    world_points = []
    for i in range(5):
        frame = cam.capture()
        dets = detector.detect(frame, ["box"])
        if dets and dets[0].depth_valid and dets[0].point_camera:
            wp = tf.transform_point_between(dets[0].point_camera, "camera", "world")
            world_points.append(wp)
            print(f"    Sample {i+1}: world=({wp[0]:.3f}, {wp[1]:.3f}, {wp[2]:.3f})")
        time.sleep(0.6)

    if len(world_points) < 3:
        _fail("world_consistent", f"only {len(world_points)} valid samples")

    stacked = np.array(world_points)
    std = stacked.std(axis=0)
    print(f"  Std dev: ({std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}) m")

    if std.max() > 0.05:
        _warn("world_consistent", f"high jitter {std.max():.3f}m — check odometry or camera mount")
    else:
        _pass(f"world coordinate stable: max std = {std.max():.4f}m")


if __name__ == "__main__":
    print("=== hardware/test_calibration.py ===")
    print("Prerequisites: OAK-D on robot, UNITREE_NETWORK_INTERFACE set\n")

    test_camera_to_base_matrix()
    test_known_point_in_base_frame()
    test_world_frame_consistent()

    print("\nDone. Adjust env vars and re-run until numbers match reality.")
