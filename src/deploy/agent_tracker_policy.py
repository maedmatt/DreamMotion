# pyright: reportMissingImports=false
"""BeyondMimic tracker policy with blend-in/blend-out on motion load.

When a new motion is loaded via load_motion():
  - Blend-in: interpolates from current robot dof_pos to motion frame 0
  - Blend-out: interpolates from motion last frame to default standing pose

Set BLEND_ENABLED = False below to disable blending for testing.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from robojudo.policy import policy_registry
from robojudo.policy.protomotions_bm_tracker_policy import (
    ProtoMotionsBMTrackerPolicy,
)
from robojudo.policy.protomotions_utils import MotionPlayer
from robojudo.utils.rotation import TransformAlignment

logger = logging.getLogger(__name__)

# ── Toggle this to enable/disable blend edges ──
BLEND_ENABLED = True
BLEND_DURATION_S = 0.5

# fmt: off
# Default standing joint angles (29 DOFs, same order as ONNX metadata)
DEFAULT_DOF_POS = np.array([
    -0.312,  0.0,    0.0,   0.669, -0.363,  0.0,    # left leg
    -0.312,  0.0,    0.0,   0.669, -0.363,  0.0,    # right leg
     0.0,    0.0,    0.0,                            # waist
     0.2,    0.2,    0.0,   0.6,   0.0, 0.0, 0.0,   # left arm
     0.2,   -0.2,    0.0,   0.6,   0.0, 0.0, 0.0,   # right arm
], dtype=np.float32)
# fmt: on


@policy_registry.register
class AgentTrackerPolicy(ProtoMotionsBMTrackerPolicy):
    """Tracker policy with motion loading and optional blend edges.

    Call load_motion() to swap in a new .pt file. If BLEND_ENABLED is
    True, blend frames are prepended/appended for smooth transitions.
    """

    def load_motion(
        self,
        motion_path: str,
        current_dof_pos: np.ndarray | None = None,
    ) -> None:
        """Load a new motion, optionally blend edges, reset tracking."""
        timing = self._meta["timing"]
        player = MotionPlayer(
            motion_path, motion_index=0, control_dt=timing["control_dt"]
        )

        _zero_align_motion(player, self._anchor_idx)

        if BLEND_ENABLED:
            blend_from = (
                current_dof_pos
                if current_dof_pos is not None
                else DEFAULT_DOF_POS.copy()
            )
            _blend_motion_edges(player, blend_from)

        self._player = player
        self.reset()

        logger.info(
            f"[AgentTracker] Loaded: {Path(motion_path).name} "
            f"({self._player.total_frames} frames, "
            f"{self._player.total_frames * timing['control_dt']:.1f}s)"
        )


# ── Blend utilities (from ZmqMotionPolicy) ──────────────────────────


def _zero_align_motion(player: MotionPlayer, anchor_idx: int) -> None:
    """Zero-align xy and yaw of the motion's anchor body at frame 0."""
    anchor_pos = player._body_pos[0, anchor_idx]
    anchor_quat = player._body_rot[0, anchor_idx]

    align = TransformAlignment(
        quat=anchor_quat, pos=anchor_pos, yaw_only=True, xy_only=True
    )

    shape = player._body_pos.shape
    player._body_pos = (
        align.align_pos(player._body_pos.reshape(-1, 3))
        .reshape(shape)
        .astype(np.float32)
    )

    shape = player._body_rot.shape
    player._body_rot = (
        align.align_quat(player._body_rot.reshape(-1, 4))
        .reshape(shape)
        .astype(np.float32)
    )

    shape = player._body_vel.shape
    player._body_vel = (
        align.align_xyz(player._body_vel.reshape(-1, 3))
        .reshape(shape)
        .astype(np.float32)
    )

    shape = player._body_ang_vel.shape
    player._body_ang_vel = (
        align.align_xyz(player._body_ang_vel.reshape(-1, 3))
        .reshape(shape)
        .astype(np.float32)
    )


def _blend_motion_edges(
    player: MotionPlayer,
    current_dof_pos: np.ndarray,
    blend_duration: float = BLEND_DURATION_S,
) -> None:
    """Prepend blend-in and append blend-out frames.

    Blend-in: current_dof_pos → motion frame 0
    Blend-out: motion last frame → DEFAULT_DOF_POS
    """
    from scipy.spatial.transform import Rotation as sRot
    from scipy.spatial.transform import Slerp

    ctrl_dt = player.control_dt
    n_blend = max(1, round(blend_duration / ctrl_dt))

    first = player.get_state_at_frame(0)
    last = player.get_state_at_frame(player.total_frames - 1)

    identity_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    num_bodies = player.num_bodies

    def _lerp(src: np.ndarray, dst: np.ndarray, n: int) -> np.ndarray:
        alphas = np.linspace(0.0, 1.0, n, dtype=np.float32)
        out = np.empty((n, *src.shape), dtype=np.float32)
        for i, a in enumerate(alphas):
            out[i] = (1.0 - a) * src + a * dst
        return out

    def _slerp_rot(src: np.ndarray, dst: np.ndarray, n: int) -> np.ndarray:
        out = np.empty((n, src.shape[0], 4), dtype=np.float32)
        alphas = np.linspace(0.0, 1.0, n, dtype=np.float64)
        for b in range(src.shape[0]):
            key_rots = sRot.from_quat(np.stack([src[b], dst[b]]).astype(np.float64))
            out[:, b, :] = (
                Slerp([0.0, 1.0], key_rots)(alphas).as_quat().astype(np.float32)
            )
        return out

    zero_dof_vel = np.zeros_like(first["dof_vel"])
    zero_body_vel = np.zeros_like(first["body_vel"])
    zero_ang_vel = np.zeros_like(first["body_ang_vel"])
    body_rot_id = np.tile(identity_quat, (num_bodies, 1))

    # Blend-in: current state → first motion frame
    bi = {
        "dof_pos": _lerp(current_dof_pos.astype(np.float32), first["dof_pos"], n_blend),
        "dof_vel": _lerp(zero_dof_vel, first["dof_vel"], n_blend),
        "body_rot": _slerp_rot(body_rot_id, first["body_rot"], n_blend),
        "body_pos": _lerp(first["body_pos"], first["body_pos"], n_blend),
        "body_vel": _lerp(zero_body_vel, first["body_vel"], n_blend),
        "body_ang_vel": _lerp(zero_ang_vel, first["body_ang_vel"], n_blend),
    }

    # Blend-out: last motion frame → default standing
    bo = {
        "dof_pos": _lerp(last["dof_pos"], DEFAULT_DOF_POS.copy(), n_blend),
        "dof_vel": _lerp(last["dof_vel"], zero_dof_vel, n_blend),
        "body_rot": _slerp_rot(last["body_rot"], body_rot_id, n_blend),
        "body_pos": _lerp(last["body_pos"], last["body_pos"], n_blend),
        "body_vel": _lerp(last["body_vel"], zero_body_vel, n_blend),
        "body_ang_vel": _lerp(last["body_ang_vel"], zero_ang_vel, n_blend),
    }

    # Concatenate: blend-in + original + blend-out
    for attr in (
        "_dof_pos",
        "_dof_vel",
        "_body_rot",
        "_body_pos",
        "_body_vel",
        "_body_ang_vel",
    ):
        key = attr[1:]  # strip leading underscore
        original = getattr(player, attr)
        setattr(player, attr, np.concatenate([bi[key], original, bo[key]], axis=0))

    player._num_frames = player._dof_pos.shape[0]

    logger.info(
        f"[AgentTracker] Blended: +{n_blend} in, +{n_blend} out "
        f"→ {player.total_frames} total frames"
    )
