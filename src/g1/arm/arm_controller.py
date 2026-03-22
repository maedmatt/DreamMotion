from __future__ import annotations

import importlib
import math
import time
from functools import lru_cache
from typing import Any

import numpy as np

from g1.unitree_common import ensure_channel_initialized

# Approximate right shoulder position in base_link frame (metres).
# base_link origin sits at the robot's hip. The right shoulder is roughly
# centred fore-aft, offset 0.15 m to the right, and 0.28 m above the hip.
_RIGHT_SHOULDER_BASE: np.ndarray = np.array([0.0, -0.15, 0.28])

# G1 right arm joint motor indices (from Unitree SDK g1_arm5_sdk_dds_example)
_R_SHOULDER_PITCH = 22  # 0 = hanging; positive = arm swings forward
_R_SHOULDER_ROLL = 23  # 0 = arm at side; negative = arm abducts to the right
_R_SHOULDER_YAW = 24
_R_ELBOW = 25
_R_WRIST_ROLL = 26

# Left arm indices (held at current position during pointing)
_LEFT_ARM_JOINTS = [15, 16, 17, 18, 19]
_RIGHT_ARM_JOINTS = [22, 23, 24, 25, 26]

# We intentionally leave waist yaw out of the pointing controller so a simple
# point gesture cannot twist the torso. The left arm is still published at its
# current pose to keep the arm SDK override stable while the right arm moves.
_CONTROLLED_JOINTS: list[int] = [
    *_LEFT_ARM_JOINTS,
    *_RIGHT_ARM_JOINTS,
]

# kNotUsedJoint — writing q=1 here enables the arm SDK override, q=0 releases it
_WEIGHT_JOINT = 29

_KP = 60.0
_KD = 1.5
_CONTROL_DT = 0.02  # 50 Hz publish rate

_POINT_ROLL_BIAS = -0.18
_POINT_ELBOW_EXTENSION = -0.12


class ArmPointController:
    """Point the G1 right arm at a 3-D target using the ``rt/arm_sdk`` DDS channel.

    The ``rt/arm_sdk`` topic overrides arm joint commands without interfering
    with the legged locomotion pipeline.  No sport-mode switching is required.

    The controller blends smoothly in three phases:

    1. Blend from the robot's current arm pose to the computed pointing pose.
    2. Hold the pointing pose.
    3. Return the arm to its sampled starting pose and release the arm SDK override.

    Usage::

        ctrl = ArmPointController()
        ctrl.point_at((1.0, 0.0, -0.5))  # target in base_link frame
    """

    def __init__(self) -> None:
        ensure_channel_initialized()

        channel_mod = importlib.import_module("unitree_sdk2py.core.channel")
        default_mod = importlib.import_module("unitree_sdk2py.idl.default")
        types_mod = importlib.import_module("unitree_sdk2py.idl.unitree_hg.msg.dds_")
        crc_mod = importlib.import_module("unitree_sdk2py.utils.crc")

        LowCmd_ = types_mod.LowCmd_
        LowState_ = types_mod.LowState_

        self._pub = channel_mod.ChannelPublisher("rt/arm_sdk", LowCmd_)
        self._pub.Init()

        self._sub = channel_mod.ChannelSubscriber("rt/lowstate", LowState_)
        self._low_state: Any | None = None
        self._sub.Init(self._on_lowstate, 10)

        self._cmd = default_mod.unitree_hg_msg_dds__LowCmd_()
        self._crc = crc_mod.CRC()

    def _on_lowstate(self, msg: Any) -> None:
        self._low_state = msg

    def _wait_for_state(self, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while self._low_state is None:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    "Timed out waiting for rt/lowstate — is the robot on?"
                )
            time.sleep(0.05)

    @staticmethod
    def _pointing_angles(
        target_xyz: tuple[float, float, float],
    ) -> dict[int, float] | None:
        """Compute right-arm joint angles to point toward *target_xyz* in base_link.

        Returns None if the target is too close to the shoulder (< 5 cm) to be
        meaningful.

        Joint sign conventions (G1 right arm):
          RightShoulderPitch (22): 0 = arm hanging; positive = arm swings forward.
          RightShoulderRoll  (23): 0 = arm at side; negative = arm abducts to the right.
          RightShoulderYaw   (24): used for horizontal aiming toward the target.
          RightElbow         (25): slightly extended for a straighter pointing pose.
          RightWristRoll     (26): held at 0 (neutral).
        """
        direction = np.asarray(target_xyz) - _RIGHT_SHOULDER_BASE
        dist = float(np.linalg.norm(direction))
        if dist < 0.05:
            return None
        d = direction / dist  # unit vector in base_link: x=fwd, y=left, z=up

        horizontal = max(1e-6, math.hypot(d[0], d[1]))

        # Elevation from the shoulder toward the target.
        pitch = math.atan2(-d[2], horizontal)
        pitch = max(-2.0, min(1.2, pitch))

        # Use shoulder yaw for left/right aiming instead of forcing roll to do all the work.
        yaw = math.atan2(d[1], d[0])
        yaw = max(-1.2, min(1.2, yaw))

        # Keep a small outward abduction bias so the upper arm clears the torso.
        roll = _POINT_ROLL_BIAS + math.asin(max(-1.0, min(1.0, -d[1]))) * 0.15
        roll = max(-0.45, min(0.2, roll))

        return {
            _R_SHOULDER_PITCH: pitch,
            _R_SHOULDER_ROLL: roll,
            _R_SHOULDER_YAW: yaw,
            _R_ELBOW: _POINT_ELBOW_EXTENSION,
            _R_WRIST_ROLL: 0.0,
        }

    def _publish(self, joint_angles: dict[int, float], weight: float) -> None:
        """Publish one arm SDK frame.

        *joint_angles* maps motor index → target q (rad).
        Joints not in the dict are held at their current robot state.
        *weight* = motor_cmd[29].q: 1.0 = full arm SDK control, 0.0 = released.
        """
        assert self._low_state is not None
        self._cmd.motor_cmd[_WEIGHT_JOINT].q = weight
        for idx in _CONTROLLED_JOINTS:
            q = joint_angles.get(idx, float(self._low_state.motor_state[idx].q))
            self._cmd.motor_cmd[idx].q = q
            self._cmd.motor_cmd[idx].dq = 0.0
            self._cmd.motor_cmd[idx].kp = _KP
            self._cmd.motor_cmd[idx].kd = _KD
            self._cmd.motor_cmd[idx].tau = 0.0
        self._cmd.crc = self._crc.Crc(self._cmd)
        self._pub.Write(self._cmd)

    def point_at(
        self,
        target_xyz: tuple[float, float, float],
        blend_secs: float = 1.0,
        hold_secs: float = 3.0,
    ) -> None:
        """Point the right arm at *target_xyz* (base_link frame) then release.

        Args:
            target_xyz: (x, y, z) in base_link frame (x=forward, y=left, z=up).
            blend_secs: Duration for smooth blend-in and blend-out.
            hold_secs: Duration to hold the pointing pose.
        """
        self._wait_for_state()

        targets = self._pointing_angles(target_xyz)
        if targets is None:
            return  # target too close to shoulder; skip silently

        blend_steps = max(1, round(blend_secs / _CONTROL_DT))
        hold_steps = max(1, round(hold_secs / _CONTROL_DT))
        release_steps = max(1, round(0.5 / _CONTROL_DT))

        # Snapshot the robot's current arm pose before we take control.
        assert self._low_state is not None
        initial_q: dict[int, float] = {
            idx: float(self._low_state.motor_state[idx].q) for idx in _CONTROLLED_JOINTS
        }

        # Phase 1: blend toward pointing targets (weight = 1 throughout).
        # Joints not in `targets` stay at initial_q (left arm + waist held in place).
        for i in range(blend_steps):
            alpha = (i + 1) / blend_steps
            blended: dict[int, float] = {}
            for idx in _CONTROLLED_JOINTS:
                t = targets.get(idx, initial_q[idx])
                blended[idx] = initial_q[idx] + alpha * (t - initial_q[idx])
            self._publish(blended, 1.0)
            time.sleep(_CONTROL_DT)

        # Phase 2: hold the pointing pose.
        pointing_q: dict[int, float] = {
            idx: targets.get(idx, initial_q[idx]) for idx in _CONTROLLED_JOINTS
        }
        for _ in range(hold_steps):
            self._publish(pointing_q, 1.0)
            time.sleep(_CONTROL_DT)

        # Phase 3: return to the sampled starting pose while weight stays 1.
        for i in range(blend_steps):
            alpha = (i + 1) / blend_steps
            returning: dict[int, float] = {}
            for idx in _CONTROLLED_JOINTS:
                returning[idx] = pointing_q[idx] + alpha * (initial_q[idx] - pointing_q[idx])
            self._publish(returning, 1.0)
            time.sleep(_CONTROL_DT)

        # Phase 4: release arm SDK (blend weight 1 → 0).
        for i in range(release_steps):
            self._cmd.motor_cmd[_WEIGHT_JOINT].q = 1.0 - (i + 1) / release_steps
            self._cmd.crc = self._crc.Crc(self._cmd)
            self._pub.Write(self._cmd)
            time.sleep(_CONTROL_DT)


@lru_cache(maxsize=1)
def get_arm_controller() -> ArmPointController:
    """Return the singleton arm point controller (initialised on first call)."""
    return ArmPointController()
