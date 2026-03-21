from __future__ import annotations

import importlib
import time
from functools import lru_cache

from g1.unitree_common import ensure_channel_initialized


class SdkLocomotionController:
    """Open-loop locomotion controller using Unitree LocoClient."""

    def __init__(self) -> None:
        ensure_channel_initialized()
        loco_module = importlib.import_module("unitree_sdk2py.g1.loco.g1_loco_client")
        self._client = loco_module.LocoClient()
        self._client.SetTimeout(5.0)
        self._client.Init()

    def walk_forward_distance(
        self,
        distance_m: float,
        yaw_rad: float = 0.0,
        speed: float = 0.25,
        yaw_speed: float = 0.3,
    ) -> None:
        """Open-loop walk: turn to face yaw_rad, then walk distance_m forward.

        No odometry required — uses elapsed time and fixed speeds.
        Suitable when position feedback is unavailable (e.g. Unitree G1).

        Args:
            distance_m: How far to walk (meters).
            yaw_rad: Heading offset to correct before walking (radians).
                Positive = left, negative = right (robot base_link convention).
            speed: Forward speed (m/s).
            yaw_speed: Rotation speed (rad/s).
        """
        # 1. Turn to face the target
        if abs(yaw_rad) > 0.05:
            turn_dir = 1.0 if yaw_rad > 0 else -1.0
            turn_duration = abs(yaw_rad) / yaw_speed
            deadline = time.monotonic() + turn_duration
            while time.monotonic() < deadline:
                self._client.Move(0.0, 0.0, turn_dir * yaw_speed)
                time.sleep(0.5)
            self.stop()
            time.sleep(0.3)  # settle

        # 2. Walk straight
        if distance_m > 0.05:
            walk_duration = distance_m / speed
            deadline = time.monotonic() + walk_duration
            while time.monotonic() < deadline:
                self._client.Move(speed, 0.0, 0.0)
                time.sleep(0.5)
            self.stop()

    def step_backward(self, distance_m: float) -> None:
        """Move backward by approximately *distance_m* meters."""
        walk_duration = distance_m / 0.3  # ~0.3 m/s backward
        deadline = time.monotonic() + walk_duration
        while time.monotonic() < deadline:
            self._client.Move(-0.3, 0.0, 0.0)
            time.sleep(0.5)
        self.stop()

    def stop(self) -> None:
        """Send zero velocity."""
        self._client.Move(0.0, 0.0, 0.0)


@lru_cache(maxsize=1)
def get_sdk_controller() -> SdkLocomotionController:
    """Return the singleton SDK locomotion controller."""
    return SdkLocomotionController()
