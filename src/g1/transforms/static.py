from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from g1.transforms.math import mat_from_rpy

# Path to calibration file produced by the OpenCV checkerboard procedure.
# When this file exists it takes priority over env-var estimates.
_CALIB_FILE = (
    Path(__file__).parent.parent.parent.parent / "config" / "camera_to_base.json"
)


def _load_from_file() -> np.ndarray | None:
    """Load the 4x4 matrix from config/camera_to_base.json if it exists."""
    if not _CALIB_FILE.exists():
        return None
    with _CALIB_FILE.open() as f:
        data = json.load(f)
    T = np.array(data["camera_to_base"], dtype=np.float64)
    if T.shape != (4, 4):
        msg = f"camera_to_base.json must contain a 4x4 matrix, got {T.shape}"
        raise ValueError(msg)
    return T


def _estimate_from_env() -> np.ndarray:
    """Rough estimate from physical measurements via env vars.

    Used as fallback until a real checkerboard calibration is available.
    Override with env vars or (better) by placing config/camera_to_base.json.

        G1_CAMERA_X_OFFSET   forward offset from base_link (m), default 0.05
        G1_CAMERA_Y_OFFSET   lateral offset (m), default 0.0
        G1_CAMERA_Z_OFFSET   height above base_link (m), default 0.45
        G1_CAMERA_PITCH      downward tilt in radians, default -0.35 (~20 deg)
    """
    x = float(os.environ.get("G1_CAMERA_X_OFFSET", "0.05"))
    y = float(os.environ.get("G1_CAMERA_Y_OFFSET", "0.0"))
    z = float(os.environ.get("G1_CAMERA_Z_OFFSET", "0.45"))
    roll = float(os.environ.get("G1_CAMERA_ROLL", "0.0"))
    pitch = float(os.environ.get("G1_CAMERA_PITCH", "-0.35"))
    yaw = float(os.environ.get("G1_CAMERA_YAW", "0.0"))

    # Camera optical frame: Z forward, X right, Y down
    # Robot base_link frame: X forward, Y left, Z up
    R_optical = np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    T_optical = np.eye(4)
    T_optical[:3, :3] = R_optical
    T_mount = mat_from_rpy([x, y, z], [roll, pitch, yaw])
    return T_mount @ T_optical


def camera_to_base() -> np.ndarray:
    """Return the 4x4 transform from camera_link to base_link.

    Priority:
      1. config/camera_to_base.json  (from OpenCV checkerboard calibration)
      2. Env-var estimate            (rough physical measurement, fallback)
    """
    T = _load_from_file()
    if T is not None:
        return T
    return _estimate_from_env()
