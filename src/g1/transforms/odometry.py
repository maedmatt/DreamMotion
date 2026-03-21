from __future__ import annotations

import importlib
import threading
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from g1.transforms.math import mat_from_translation_quat
from g1.unitree_common import ensure_channel_initialized


@dataclass(frozen=True, slots=True)
class OdomState:
    """Raw odometry state from the Unitree SDK."""

    x: float
    y: float
    z: float
    quat_x: float
    quat_y: float
    quat_z: float
    quat_w: float
    vx: float
    vy: float
    vyaw: float


class OdometrySubscriber:
    """Subscribe to Unitree SportModeState for base_link -> odom transforms."""

    def __init__(self) -> None:
        ensure_channel_initialized()

        sport_module = importlib.import_module(
            "unitree_sdk2py.g1.sport.g1_sport_client"
        )
        self._lock = threading.Lock()
        self._latest: OdomState | None = None

        # Subscribe to sport mode state
        self._subscriber = sport_module.SportClient()
        self._subscriber.Init()
        self._subscriber.SetTimeout(5.0)

    def update(self, state: object) -> None:
        """Callback for DDS state updates.

        The Unitree SDK delivers SportModeState with fields:
          position [x, y, z], imu_state.quaternion [w, x, y, z],
          velocity [vx, vy, vyaw].
        """
        # Extract fields from SDK state object
        pos = state.position  # type: ignore[attr-defined]
        quat = state.imu_state.quaternion  # type: ignore[attr-defined]
        vel = state.velocity  # type: ignore[attr-defined]

        odom = OdomState(
            x=float(pos[0]),
            y=float(pos[1]),
            z=float(pos[2]),
            # Unitree uses [w, x, y, z] -> convert to scipy [x, y, z, w]
            quat_x=float(quat[1]),
            quat_y=float(quat[2]),
            quat_z=float(quat[3]),
            quat_w=float(quat[0]),
            vx=float(vel[0]),
            vy=float(vel[1]),
            vyaw=float(vel[2]),
        )
        with self._lock:
            self._latest = odom

    def get_state(self) -> OdomState:
        """Return the latest odometry state.

        Raises:
            RuntimeError: If no odometry has been received yet.
        """
        with self._lock:
            if self._latest is None:
                raise RuntimeError(
                    "No odometry received yet. Ensure the robot is powered on "
                    "and the SDK connection is active."
                )
            return self._latest

    def base_to_odom(self) -> np.ndarray:
        """Return the latest 4x4 transform from base_link to odom (world) frame."""
        state = self.get_state()
        return mat_from_translation_quat(
            translation=[state.x, state.y, state.z],
            quaternion=[state.quat_x, state.quat_y, state.quat_z, state.quat_w],
        )

    def velocity_magnitude(self) -> float:
        """Return the current linear + angular velocity magnitude."""
        state = self.get_state()
        return float(np.sqrt(state.vx**2 + state.vy**2 + state.vyaw**2))


@lru_cache(maxsize=1)
def get_odometry() -> OdometrySubscriber:
    """Return the singleton OdometrySubscriber."""
    return OdometrySubscriber()
