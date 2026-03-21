from __future__ import annotations

import os

import numpy as np

from g1.transforms.math import mat_from_rpy


def _load_camera_offsets() -> tuple[float, float, float]:
    """Load camera mount offsets from env vars (meters)."""
    x = float(os.environ.get("G1_CAMERA_X_OFFSET", "0.05"))
    y = float(os.environ.get("G1_CAMERA_Y_OFFSET", "0.0"))
    z = float(os.environ.get("G1_CAMERA_Z_OFFSET", "0.45"))
    return x, y, z


def _load_camera_rpy() -> tuple[float, float, float]:
    """Load camera mount orientation from env vars (radians).

    The OAK-D is physically angled downward, so pitch will typically be
    negative (e.g. -0.35 for ~20 degrees down).
    """
    roll = float(os.environ.get("G1_CAMERA_ROLL", "0.0"))
    pitch = float(os.environ.get("G1_CAMERA_PITCH", "-0.35"))
    yaw = float(os.environ.get("G1_CAMERA_YAW", "0.0"))
    return roll, pitch, yaw


def _optical_to_robot_rotation() -> np.ndarray:
    """Rotation from camera optical frame to robot body frame.

    Camera optical frame: Z forward, X right, Y down.
    Robot base_link frame: X forward, Y left, Z up.

    Returns:
        3x3 rotation matrix.
    """
    return np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )


def camera_to_base() -> np.ndarray:
    """Return the static 4x4 transform from camera_link to base_link.

    The OAK-D is mounted rigidly on the robot with a fixed downward angle.
    The transform combines:
    1. Optical-to-robot frame rotation
    2. Physical mount orientation (roll, pitch, yaw from calibration)
    3. Translation offset from base_link origin to camera mount point

    All values are configurable via env vars or can be replaced with a
    calibration matrix from an OpenCV checkerboard procedure.
    """
    x, y, z = _load_camera_offsets()
    roll, pitch, yaw = _load_camera_rpy()

    # Mount transform: where the camera sits relative to base_link
    T_mount = mat_from_rpy([x, y, z], [roll, pitch, yaw])

    # Optical frame correction
    R_optical = _optical_to_robot_rotation()
    T_optical = np.eye(4)
    T_optical[:3, :3] = R_optical

    # Full chain: point_in_camera -> optical correction -> mount transform
    return T_mount @ T_optical
