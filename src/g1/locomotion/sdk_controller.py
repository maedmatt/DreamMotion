from __future__ import annotations

import importlib
import math
import time
from functools import lru_cache
from typing import TYPE_CHECKING

from g1.unitree_common import ensure_channel_initialized

if TYPE_CHECKING:
    from collections.abc import Callable

    from g1.transforms.odometry import OdomState

# Proportional gains for the walk controller.
_KP_DIST = 0.5  # m/s per meter of distance
_KP_YAW = 1.0  # rad/s per radian of heading error
_MAX_VX = 0.3  # m/s forward speed cap
_MAX_VYAW = 0.5  # rad/s yaw rate cap
_CONTROL_HZ = 20  # control loop frequency


class SdkLocomotionController:
    """Velocity-based P-controller for walking using Unitree SportClient."""

    def __init__(self) -> None:
        ensure_channel_initialized()
        loco_module = importlib.import_module(
            "unitree_sdk2py.g1.loco.g1_loco_client"
        )
        self._client = loco_module.LocoClient()
        self._client.SetTimeout(5.0)
        self._client.Init()

    def walk_to_point(
        self,
        target_x: float,
        target_y: float,
        current_odom_func: Callable[[], OdomState],
        stop_short_m: float = 0.5,
    ) -> bool:
        """Walk toward a target in odom frame, stopping *stop_short_m* away.

        Uses a simple proportional controller:
          vx  = clamp(Kp_dist * distance, -MAX_VX, MAX_VX)
          vyaw = clamp(Kp_yaw * heading_error, -MAX_VYAW, MAX_VYAW)

        Args:
            target_x: Target X in odom frame (meters).
            target_y: Target Y in odom frame (meters).
            current_odom_func: Callable returning the latest OdomState.
            stop_short_m: Stop this far from the target (meters).

        Returns:
            True if the robot reached the stopping zone, False on timeout.
        """
        dt = 1.0 / _CONTROL_HZ
        timeout_s = 60.0
        elapsed = 0.0

        while elapsed < timeout_s:
            odom = current_odom_func()
            dx = target_x - odom.x
            dy = target_y - odom.y
            distance = math.sqrt(dx * dx + dy * dy)

            if distance <= stop_short_m:
                self.stop()
                return True

            # Desired heading to target
            desired_yaw = math.atan2(dy, dx)
            # Current yaw from quaternion
            from scipy.spatial.transform import Rotation

            current_yaw = Rotation.from_quat(
                [odom.quat_x, odom.quat_y, odom.quat_z, odom.quat_w]
            ).as_euler("xyz")[2]

            heading_error = _normalize_angle(desired_yaw - current_yaw)

            # Proportional control
            vx = _clamp(_KP_DIST * distance, -_MAX_VX, _MAX_VX)
            vyaw = _clamp(_KP_YAW * heading_error, -_MAX_VYAW, _MAX_VYAW)

            # Reduce forward speed when heading is off
            if abs(heading_error) > 0.3:
                vx *= 0.3

            self._client.Move(vx, 0.0, vyaw)
            time.sleep(dt)
            elapsed += dt

        self.stop()
        return False

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


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


@lru_cache(maxsize=1)
def get_sdk_controller() -> SdkLocomotionController:
    """Return the singleton SDK locomotion controller."""
    return SdkLocomotionController()
