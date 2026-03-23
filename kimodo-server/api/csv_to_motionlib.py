#!/usr/bin/env python3
"""
Standalone converter: Kimodo G1 CSV  -->  ProtoMotions .pt MotionLib

Dependencies: torch, numpy  (no mujoco / dm_control / scipy required)

Usage:
    python csv_to_motionlib.py --input output.csv --output motions.pt
    python csv_to_motionlib.py --input output.csv --output motions.pt --fps 30
    python csv_to_motionlib.py --input output.csv --output motions.pt --smooth 3 --speed 0.8
"""

import argparse, json, os, sys
from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import math

import numpy as np
import torch
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════════════════════
#  Rotation utilities  (extracted from protomotions/utils/rotations.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps).unsqueeze(-1)

def _sqrt_positive_part(x):
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def wxyz_to_xyzw(quat):
    shape = quat.shape
    flat = quat.reshape(-1, 4)[:, [1, 2, 3, 0]]
    return flat.reshape(shape)

def quat_mul(a, b, w_last: bool):
    assert a.shape == b.shape
    shape = a.shape; a = a.reshape(-1, 4); b = b.reshape(-1, 4)
    if w_last: x1,y1,z1,w1 = a[...,0],a[...,1],a[...,2],a[...,3]; x2,y2,z2,w2 = b[...,0],b[...,1],b[...,2],b[...,3]
    else:      w1,x1,y1,z1 = a[...,0],a[...,1],a[...,2],a[...,3]; w2,x2,y2,z2 = b[...,0],b[...,1],b[...,2],b[...,3]
    ww = (z1+x1)*(x2+y2); yy = (w1-y1)*(w2+z2); zz = (w1+y1)*(w2-z2)
    xx = ww+yy+zz; qq = 0.5*(xx+(z1-x1)*(x2-y2))
    w = qq-ww+(z1-y1)*(y2-z2); x = qq-xx+(x1+w1)*(x2+w2)
    y = qq-yy+(w1-x1)*(y2+z2); z = qq-zz+(z1+y1)*(w2-x2)
    if w_last: return torch.stack([x,y,z,w], dim=-1).reshape(shape)
    else:      return torch.stack([w,x,y,z], dim=-1).reshape(shape)

def quat_pos(x, w_last=True):
    w = x[...,3:] if w_last else x[...,0:1]
    return (1 - 2*(w<0).float()) * x

def quat_normalize(q):
    return _normalize(quat_pos(q))

def quat_mul_norm(x, y, w_last: bool):
    return quat_normalize(quat_mul(x, y, w_last))

def quat_conjugate(a, w_last: bool):
    shape = a.shape; a = a.reshape(-1, 4)
    if w_last: return torch.cat((-a[:,:3], a[:,-1:]), dim=-1).reshape(shape)
    else:      return torch.cat((a[:,0:1], -a[:,1:]), dim=-1).reshape(shape)

def quat_from_angle_axis(angle, axis, w_last: bool):
    theta = (angle / 2).unsqueeze(-1)
    xyz = _normalize(axis) * theta.sin()
    w = theta.cos()
    if w_last: return _normalize(torch.cat([xyz, w], dim=-1))
    else:      return _normalize(torch.cat([w, xyz], dim=-1))

def quat_angle_axis(x, w_last: bool):
    shape = x.shape[:-1]; quat = x.reshape(-1, 4)
    scalar_index = 3 if w_last else 0
    needs_flip = quat[..., scalar_index] < 0
    quat = torch.where(needs_flip.unsqueeze(-1), -quat, quat)
    w = quat[...,3] if w_last else quat[...,0]
    axis = quat[...,:3] if w_last else quat[...,1:]
    norm_axis = torch.norm(axis, p=2, dim=-1)
    angle = 2 * torch.atan2(norm_axis, w)
    axis_normalized = axis / norm_axis.unsqueeze(-1).clamp(min=1e-9)
    return angle.reshape(shape), axis_normalized.reshape(shape + (3,))

def quaternion_to_matrix(quaternions, w_last: bool):
    if w_last: i,j,k,r = torch.unbind(quaternions, -1)
    else:      r,i,j,k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack((
        1-two_s*(j*j+k*k), two_s*(i*j-k*r), two_s*(i*k+j*r),
        two_s*(i*j+k*r), 1-two_s*(i*i+k*k), two_s*(j*k-i*r),
        two_s*(i*k-j*r), two_s*(j*k+i*r), 1-two_s*(i*i+j*j),
    ), -1)
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def matrix_to_quaternion(matrix, w_last: bool):
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    batch_dim = matrix.shape[:-2]
    m00,m01,m02,m10,m11,m12,m20,m21,m22 = torch.unbind(matrix.reshape(batch_dim+(9,)), dim=-1)
    q_abs = _sqrt_positive_part(torch.stack([
        1.0+m00+m11+m22, 1.0+m00-m11-m22, 1.0-m00+m11-m22, 1.0-m00-m11+m22,
    ], dim=-1))
    quat_by_rijk = torch.stack([
        torch.stack([q_abs[...,0]**2, m21-m12, m02-m20, m10-m01], dim=-1),
        torch.stack([m21-m12, q_abs[...,1]**2, m10+m01, m02+m20], dim=-1),
        torch.stack([m02-m20, m10+m01, q_abs[...,2]**2, m12+m21], dim=-1),
        torch.stack([m10-m01, m20+m02, m21+m12, q_abs[...,3]**2], dim=-1),
    ], dim=-2)
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    quat_candidates = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    if w_last: quat_candidates = wxyz_to_xyzw(quat_candidates)
    return quat_candidates

def angle_from_matrix_axis(rot_mat, axis):
    axis = axis.to(rot_mat.device, rot_mat.dtype)
    cos_theta = (torch.einsum("bii->b", rot_mat) - 1.0) / 2.0
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    skew_sym = 0.5 * (rot_mat - rot_mat.transpose(-1, -2))
    sin_theta_axis = torch.stack([skew_sym[:,2,1], skew_sym[:,0,2], skew_sym[:,1,0]], dim=-1)
    sin_theta = torch.einsum("bi,i->b", sin_theta_axis, axis)
    return torch.atan2(sin_theta, cos_theta)


# ═══════════════════════════════════════════════════════════════════════════════
#  KinematicInfo & embedded G1 data
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class KinematicInfo:
    body_names: List[str]
    dof_names: List[str]
    parent_indices: List[int]
    local_pos: torch.Tensor
    local_rot_ref_mat: torch.Tensor
    hinge_axes_map: Dict[int, torch.Tensor]
    nq: int; nv: int; num_bodies: int; num_dofs: int
    dof_limits_lower: torch.Tensor; dof_limits_upper: torch.Tensor

    def to(self, device, dtype=None):
        kw = {"device": device}
        if dtype: kw["dtype"] = dtype
        self.local_pos = self.local_pos.to(**kw)
        self.local_rot_ref_mat = self.local_rot_ref_mat.to(**kw)
        self.dof_limits_lower = self.dof_limits_lower.to(**kw)
        self.dof_limits_upper = self.dof_limits_upper.to(**kw)
        self.hinge_axes_map = {k: v.to(**kw) for k, v in self.hinge_axes_map.items()}
        return self

_G1_KINEMATIC_JSON = r"""{"body_names":["pelvis","head","left_hip_pitch_link","left_hip_roll_link","left_hip_yaw_link","left_knee_link","left_ankle_pitch_link","left_ankle_roll_link","right_hip_pitch_link","right_hip_roll_link","right_hip_yaw_link","right_knee_link","right_ankle_pitch_link","right_ankle_roll_link","waist_yaw_link","waist_roll_link","torso_link","left_shoulder_pitch_link","left_shoulder_roll_link","left_shoulder_yaw_link","left_elbow_link","left_wrist_roll_link","left_wrist_pitch_link","left_wrist_yaw_link","left_rubber_hand","right_shoulder_pitch_link","right_shoulder_roll_link","right_shoulder_yaw_link","right_elbow_link","right_wrist_roll_link","right_wrist_pitch_link","right_wrist_yaw_link","right_rubber_hand"],"dof_names":["left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint","left_knee_joint","left_ankle_pitch_joint","left_ankle_roll_joint","right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint","right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint","waist_yaw_joint","waist_roll_joint","waist_pitch_joint","left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint","left_elbow_joint","left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint","right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint","right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint"],"parent_indices":[-1,0,0,2,3,4,5,6,0,8,9,10,11,12,0,14,15,16,17,18,19,20,21,22,23,16,25,26,27,28,29,30,31],"local_pos":[[0.0,0.0,0.7929999828338623],[0.0,0.0,0.4000000059604645],[0.0,0.06445199996232986,-0.10270000249147415],[0.0,0.052000001072883606,-0.03046499937772751],[0.025001000612974167,0.0,-0.12411999702453613],[-0.0782729983329773,0.0021488999482244253,-0.17734000086784363],[0.0,-9.444500028621405e-05,-0.30000999569892883],[0.0,0.0,-0.01755799911916256],[0.0,-0.06445199996232986,-0.10270000249147415],[0.0,-0.052000001072883606,-0.03046499937772751],[0.025001000612974167,0.0,-0.12411999702453613],[-0.0782729983329773,-0.0021488999482244253,-0.17734000086784363],[0.0,9.444500028621405e-05,-0.30000999569892883],[0.0,0.0,-0.01755799911916256],[0.0,0.0,0.0],[-0.003963499795645475,0.0,0.04399999976158142],[0.0,0.0,0.0],[0.00395630020648241,0.10022000223398209,0.2477799952030182],[0.0,0.03799999877810478,-0.013830999843776226],[0.0,0.006240000016987324,-0.10320000350475311],[0.015783000737428665,0.0,-0.08051799982786179],[0.10000000149011612,0.001887909951619804,-0.009999999776482582],[0.03799999877810478,0.0,0.0],[0.04600000008940697,0.0,0.0],[0.11999999731779099,0.0,0.0],[0.00395630020648241,-0.10021000355482101,0.2477799952030182],[0.0,-0.03799999877810478,-0.013830999843776226],[0.0,-0.006240000016987324,-0.10320000350475311],[0.015783000737428665,0.0,-0.08051799982786179],[0.10000000149011612,-0.001887909951619804,-0.009999999776482582],[0.03799999877810478,0.0,0.0],[0.04600000008940697,0.0,0.0],[0.11999999731779099,0.0,0.0]],"local_rot_ref_mat":[[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],[[0.9847439527511597,-0.0,-0.17400965094566345],[0.0,1.0,-0.0],[0.17400965094566345,0.0,0.9847439527511597]],[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],[[0.9847439527511597,0.0,0.17400965094566345],[0.0,1.0,0.0],[-0.17400965094566345,0.0,0.9847439527511597]],[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],[[0.9847439527511597,-0.0,-0.17400965094566345],[0.0,1.0,-0.0],[0.17400965094566345,0.0,0.9847439527511597]],[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],[[0.9847439527511597,0.0,0.17400965094566345],[0.0,1.0,0.0],[-0.17400965094566345,0.0,0.9847439527511597]],[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],[[1.0,0.0001993141049752012,-3.237802248534649e-10],[-0.00019159000657964498,0.961246132850647,-0.27569156885147095],[-5.494890137924813e-05,0.27569156885147095,0.961246132850647]],[[1.0,-0.0,0.0],[0.0,0.9612622857093811,0.2756352722644806],[-0.0,-0.2756352722644806,0.9612622857093811]],[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],[[1.0,-0.0001993141049752012,-3.237802248534649e-10],[0.00019159000657964498,0.961246132850647,0.27569156885147095],[-5.494890137924813e-05,-0.27569156885147095,0.961246132850647]],[[1.0,0.0,0.0],[0.0,0.9612622857093811,-0.2756352722644806],[0.0,0.2756352722644806,0.9612622857093811]],[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]],"hinge_axes_map":{"2":[[0.0,1.0,0.0]],"3":[[1.0,0.0,0.0]],"4":[[0.0,0.0,1.0]],"5":[[0.0,1.0,0.0]],"6":[[0.0,1.0,0.0]],"7":[[1.0,0.0,0.0]],"8":[[0.0,1.0,0.0]],"9":[[1.0,0.0,0.0]],"10":[[0.0,0.0,1.0]],"11":[[0.0,1.0,0.0]],"12":[[0.0,1.0,0.0]],"13":[[1.0,0.0,0.0]],"14":[[0.0,0.0,1.0]],"15":[[1.0,0.0,0.0]],"16":[[0.0,1.0,0.0]],"17":[[0.0,1.0,0.0]],"18":[[1.0,0.0,0.0]],"19":[[0.0,0.0,1.0]],"20":[[0.0,1.0,0.0]],"21":[[1.0,0.0,0.0]],"22":[[0.0,1.0,0.0]],"23":[[0.0,0.0,1.0]],"25":[[0.0,1.0,0.0]],"26":[[1.0,0.0,0.0]],"27":[[0.0,0.0,1.0]],"28":[[0.0,1.0,0.0]],"29":[[1.0,0.0,0.0]],"30":[[0.0,1.0,0.0]],"31":[[0.0,0.0,1.0]]},"nq":36,"nv":35,"num_bodies":33,"num_dofs":29}"""


def _load_g1_kinematic_info() -> KinematicInfo:
    d = json.loads(_G1_KINEMATIC_JSON)
    return KinematicInfo(
        body_names=d["body_names"],
        dof_names=d["dof_names"],
        parent_indices=d["parent_indices"],
        local_pos=torch.tensor(d["local_pos"], dtype=torch.float32),
        local_rot_ref_mat=torch.tensor(d["local_rot_ref_mat"], dtype=torch.float32),
        hinge_axes_map={int(k): torch.tensor(v, dtype=torch.float32) for k, v in d["hinge_axes_map"].items()},
        nq=d["nq"], nv=d["nv"], num_bodies=d["num_bodies"], num_dofs=d["num_dofs"],
        dof_limits_lower=torch.zeros(d["num_dofs"]),
        dof_limits_upper=torch.zeros(d["num_dofs"]),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Forward Kinematics  (extracted from protomotions/components/pose_lib.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_transforms_from_qpos(ki: KinematicInfo, qpos: torch.Tensor):
    """qpos: (B, nq) = [root_pos(3), root_quat_wxyz(4), joint_angles(29)]"""
    device, dtype = qpos.device, qpos.dtype
    B = qpos.shape[0]; Nb = ki.num_bodies
    root_pos = qpos[:, 0:3]
    root_quat = qpos[:, 3:7]
    root_quat = root_quat / torch.linalg.norm(root_quat, dim=-1, keepdim=True)
    root_rot_mat = quaternion_to_matrix(root_quat, w_last=False)

    joint_rot_mats = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(B, Nb, 3, 3).clone()
    joint_rot_mats[:, 0] = root_rot_mat

    joint_start = 0
    hinge_axes_map = {k: v.to(device, dtype) for k, v in ki.hinge_axes_map.items()}
    qpos_nr = qpos[:, 7:]
    for body_idx, axes in hinge_axes_map.items():
        n_dof = len(axes)
        angles = qpos_nr[:, joint_start:joint_start + n_dof]
        joint_start += n_dof
        body_mat = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, 3, 3).clone()
        for k in range(n_dof):
            q = quat_from_angle_axis(angles[:, k], axes[k:k+1].expand(B, -1), w_last=False)
            body_mat = torch.matmul(body_mat, quaternion_to_matrix(q, w_last=False))
        joint_rot_mats[:, body_idx] = body_mat

    return root_pos, joint_rot_mats


def _compute_fk(ki: KinematicInfo, root_pos: torch.Tensor, joint_rot_mats: torch.Tensor):
    """Forward kinematics -> world positions & rotation matrices."""
    device, dtype = root_pos.device, root_pos.dtype
    B = root_pos.shape[0]; Nb = ki.num_bodies
    parent_indices = ki.parent_indices
    local_pos = ki.local_pos.to(device, dtype).unsqueeze(0).expand(B, -1, -1)
    local_rot_ref = ki.local_rot_ref_mat.to(device, dtype).unsqueeze(0).expand(B, -1, -1, -1)

    world_pos = torch.zeros(B, Nb, 3, device=device, dtype=dtype)
    world_rot = torch.zeros(B, Nb, 3, 3, device=device, dtype=dtype)

    for i in range(Nb):
        if parent_indices[i] == -1:
            world_pos[:, i] = root_pos
            world_rot[:, i] = joint_rot_mats[:, 0]
        else:
            pidx = parent_indices[i]
            eff_local = torch.matmul(local_rot_ref[:, i], joint_rot_mats[:, i])
            world_rot[:, i] = torch.matmul(world_rot[:, pidx], eff_local)
            world_pos[:, i] = world_pos[:, pidx] + torch.matmul(
                world_rot[:, pidx], local_pos[:, i, :, None]
            ).squeeze(-1)
    return world_pos, world_rot


def _extract_qpos_from_transforms(ki, root_pos, joint_rot_mats):
    """Inverse: reconstruct qpos from root_pos + joint_rot_mats (for angle normalization)."""
    device, dtype = root_pos.device, root_pos.dtype
    B = root_pos.shape[0]
    qpos = torch.zeros(B, ki.nq, device=device, dtype=dtype)
    qpos[:, 0:3] = root_pos
    root_quat = matrix_to_quaternion(joint_rot_mats[:, 0], w_last=False)
    qpos[:, 3:7] = root_quat / torch.linalg.norm(root_quat, dim=-1, keepdim=True)
    hinge_axes_map = {k: v.to(device, dtype) for k, v in ki.hinge_axes_map.items()}
    js = 7
    for body_idx, axes in hinge_axes_map.items():
        n_dof = len(axes)
        if n_dof == 1:
            qpos[:, js] = angle_from_matrix_axis(joint_rot_mats[:, body_idx], axes[0])
        js += n_dof
    return qpos


# ═══════════════════════════════════════════════════════════════════════════════
#  Velocity computation
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_cartesian_velocity(pos, fps, max_horizon=3):
    T = pos.shape[0]
    if T < 2: return torch.zeros_like(pos)
    vels = []
    for h in range(1, max_horizon + 1):
        dt = h / fps; vel = torch.zeros_like(pos)
        if T > h:
            vel[:-h] = (pos[h:] - pos[:-h]) / dt
            vel[-h:] = vel[-h-1].unsqueeze(0).expand(h, -1, -1)
        else:
            vel[:-1] = (pos[1:] - pos[:-1]) * fps; vel[-1] = vel[-2]
        vels.append(vel)
    if max_horizon == 1: return vels[0]
    stacked = torch.stack(vels, dim=0)
    mag = torch.norm(stacked, dim=-1)
    idx = mag.argmin(dim=0).unsqueeze(-1).expand(-1, -1, 3)
    return torch.gather(stacked.permute(1,2,3,0), dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)


def _compute_angular_velocity(rot_mats, fps, max_horizon=3):
    T = rot_mats.shape[0]
    if T < 2: return torch.zeros(rot_mats.shape[:-2]+(3,), device=rot_mats.device, dtype=rot_mats.dtype)
    quats = matrix_to_quaternion(rot_mats, w_last=True)
    avels = []
    for h in range(1, max_horizon + 1):
        dt = h / fps
        av = torch.zeros(rot_mats.shape[:-2]+(3,), device=rot_mats.device, dtype=rot_mats.dtype)
        if T > h:
            q_t = quats[:-h]; q_th = quats[h:]
            diff = quat_mul_norm(q_th, quat_conjugate(q_t, True), True)
            angle, axis = quat_angle_axis(diff, True)
            valid = axis * angle.unsqueeze(-1) / dt
            av[1:T-h+1] = valid
            if h > 1 and T > h: av[T-h+1:] = valid[-1:].expand(h-1, -1, -1)
        avels.append(av)
    if max_horizon == 1: return avels[0]
    stacked = torch.stack(avels, dim=0)
    mag = torch.norm(stacked, dim=-1)
    idx = mag.argmin(dim=0).unsqueeze(-1).expand(-1, -1, 3)
    return torch.gather(stacked.permute(1,2,3,0), dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
#  Contact detection
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_contacts(positions, velocity, vel_thres=0.15, height_thresh=0.1):
    body_vel_mag = torch.linalg.norm(velocity, dim=-1)
    body_heights = positions[:, :, 2]
    return (body_vel_mag < vel_thres) & (body_heights < height_thresh)


# ═══════════════════════════════════════════════════════════════════════════════
#  Post-filter: smoothing & retiming
# ═══════════════════════════════════════════════════════════════════════════════

def _slerp(q0, q1, t):
    """Spherical linear interpolation. q0,q1: (...,4), t: (...,1) or scalar."""
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, device=q0.device, dtype=q0.dtype)
    if t.dim() == 0:
        t = t.unsqueeze(0)
    cos_half = (q0 * q1).sum(dim=-1, keepdim=True)
    # Ensure shortest path
    neg = cos_half < 0
    q1 = torch.where(neg, -q1, q1)
    cos_half = cos_half.abs()
    half_theta = torch.acos(cos_half.clamp(-1, 1))
    sin_half = torch.sqrt(1.0 - cos_half * cos_half).clamp(min=1e-8)
    t = t.unsqueeze(-1) if t.dim() < cos_half.dim() else t
    a = torch.sin((1 - t) * half_theta) / sin_half
    b = torch.sin(t * half_theta) / sin_half
    result = a * q0 + b * q1
    # Fallback to linear for near-identical quats
    linear = (1 - t) * q0 + t * q1
    close_mask = cos_half > 0.9995
    result = torch.where(close_mask, linear, result)
    return _normalize(result)


def _gaussian_kernel_1d(sigma: float, device) -> torch.Tensor:
    """Build a 1D Gaussian kernel. Kernel size = 2*ceil(3*sigma)+1."""
    radius = int(math.ceil(3.0 * sigma))
    if radius < 1:
        return torch.ones(1, device=device)
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def _smooth_1d(data: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Apply 1D convolution along dim-0. data: (T, ...), kernel: (K,)."""
    orig_shape = data.shape
    T = orig_shape[0]
    flat = data.reshape(T, -1).t().unsqueeze(1)        # (C, 1, T)
    K = kernel.shape[0]
    pad = K // 2
    padded = F.pad(flat, (pad, pad), mode="replicate")
    k = kernel.reshape(1, 1, -1)
    smoothed = F.conv1d(padded, k, padding=0)           # (C, 1, T)
    return smoothed.squeeze(1).t().reshape(orig_shape)


def smooth_qpos(qpos: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian-smooth qpos, handling quaternion normalization.

    Args:
        qpos: (T, nq) where cols 3..7 are root quaternion wxyz
        sigma: Gaussian sigma in frames (0 = no-op)

    Returns:
        Smoothed qpos (T, nq), root quaternion re-normalized.
    """
    if sigma <= 0:
        return qpos
    kernel = _gaussian_kernel_1d(sigma, qpos.device)
    out = _smooth_1d(qpos, kernel)
    # Re-normalize root quaternion after linear smoothing
    out[:, 3:7] = out[:, 3:7] / torch.linalg.norm(out[:, 3:7], dim=-1, keepdim=True)
    return out


def retime_qpos(qpos: torch.Tensor, speed: float) -> torch.Tensor:
    """Retime qpos by a speed factor using interpolation.

    Args:
        qpos: (T, nq) where cols 3..7 are root quaternion wxyz
        speed: >1 faster (fewer frames), <1 slower (more frames), 1 = no-op

    Returns:
        Retimed qpos (T', nq). Root quaternion interpolated with slerp.
    """
    if speed == 1.0:
        return qpos
    T = qpos.shape[0]
    T_new = max(2, int(round((T - 1) / speed)) + 1)
    # New timestamps mapped to original frame indices
    t_new = torch.linspace(0, T - 1, T_new, device=qpos.device)
    idx0 = t_new.long().clamp(max=T - 2)
    idx1 = (idx0 + 1).clamp(max=T - 1)
    frac = (t_new - idx0.float())                       # (T_new,)

    # Linear interp for position + joint angles (cols 0..3, 7..end)
    q0_all = qpos[idx0]; q1_all = qpos[idx1]
    out = q0_all + frac.unsqueeze(-1) * (q1_all - q0_all)

    # Slerp for root quaternion (cols 3..7)
    out[:, 3:7] = _slerp(q0_all[:, 3:7], q1_all[:, 3:7], frac)

    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  Minimal RobotState-compatible dict builder
# ═══════════════════════════════════════════════════════════════════════════════

class StateConversion(Enum):
    SIMULATOR = "simulator"
    COMMON = "common"


def _build_motion_dict(*, rigid_body_pos, rigid_body_rot, rigid_body_vel,
                        rigid_body_ang_vel, dof_pos, dof_vel,
                        rigid_body_contacts, fps):
    return {
        "state_conversion": StateConversion.COMMON,
        "fps": float(fps),
        "dof_pos": dof_pos,
        "dof_vel": dof_vel,
        "rigid_body_pos": rigid_body_pos,
        "rigid_body_rot": rigid_body_rot,
        "rigid_body_vel": rigid_body_vel,
        "rigid_body_ang_vel": rigid_body_ang_vel,
        "rigid_body_contacts": rigid_body_contacts,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MotionLib packer
# ═══════════════════════════════════════════════════════════════════════════════

_FIELD_MAP = {
    "gts": "rigid_body_pos",
    "grs": "rigid_body_rot",
    "gavs": "rigid_body_ang_vel",
    "gvs": "rigid_body_vel",
    "dvs": "dof_vel",
    "dps": "dof_pos",
}

def _pack_motionlib(motions: list, motion_files: list) -> dict:
    """Pack a list of motion dicts into a single MotionLib dict."""
    lib = {}
    for lib_key, motion_key in _FIELD_MAP.items():
        tensors = [m[motion_key] for m in motions]
        tp = torch.bool if tensors[0].dtype == torch.bool else torch.float32
        lib[lib_key] = torch.cat(tensors, dim=0).to(dtype=tp)

    contacts_list = [m["rigid_body_contacts"] for m in motions]
    tp = torch.bool if contacts_list[0].dtype == torch.bool else torch.float32
    lib["contacts"] = torch.cat(contacts_list, dim=0).to(dtype=tp)

    num_frames_list = [m["rigid_body_pos"].shape[0] for m in motions]
    fps_list = [m["fps"] for m in motions]

    lib["motion_num_frames"] = torch.tensor(num_frames_list, dtype=torch.long)
    shifted = lib["motion_num_frames"].roll(1); shifted[0] = 0
    lib["length_starts"] = shifted.cumsum(0)
    lib["motion_dt"] = torch.tensor([1.0/f for f in fps_list], dtype=torch.float32)
    lib["motion_lengths"] = torch.tensor(
        [(nf - 1) / f for nf, f in zip(num_frames_list, fps_list)], dtype=torch.float32
    )
    lib["motion_weights"] = torch.ones(len(motions), dtype=torch.float32)
    lib["motion_files"] = tuple(motion_files)
    return lib


# ═══════════════════════════════════════════════════════════════════════════════
#  Main conversion pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def convert_csv_to_motion(csv_path: str, fps: int = 30, device: str = "cpu",
                          smooth: float = 0.0, speed: float = 1.0):
    """Convert a single Kimodo G1 CSV to a motion dict.

    Args:
        csv_path: Path to CSV file.
        fps: Frames per second of the input.
        device: 'cpu' or 'cuda'.
        smooth: Gaussian smoothing sigma in frames (0 = off).
        speed: Playback speed factor (1.0 = original, 0.5 = half-speed / 2x frames,
               2.0 = double-speed / half frames).
    """
    data = np.loadtxt(csv_path, delimiter=",")
    root_pos_np = data[:, 0:3]
    root_rot_wxyz_np = data[:, 3:7]
    joint_angles_np = data[:, 7:]

    ki = _load_g1_kinematic_info().to(device)

    expected = ki.num_dofs
    actual = joint_angles_np.shape[1]
    if actual != expected:
        raise ValueError(f"CSV has {actual} joint columns, G1 expects {expected}")

    root_pos = torch.tensor(root_pos_np, device=device, dtype=torch.float32)
    root_rot_wxyz = torch.tensor(root_rot_wxyz_np, device=device, dtype=torch.float32)
    joint_angles = torch.tensor(joint_angles_np, device=device, dtype=torch.float32)

    qpos = torch.cat([root_pos, root_rot_wxyz, joint_angles], dim=-1)

    # ── Post-filter: retime then smooth (applied to qpos before FK) ──
    if speed != 1.0:
        n_before = qpos.shape[0]
        qpos = retime_qpos(qpos, speed)
        print(f"    Retimed: {n_before} -> {qpos.shape[0]} frames (speed={speed}x)")
    if smooth > 0:
        qpos = smooth_qpos(qpos, smooth)
        print(f"    Smoothed: sigma={smooth} frames (kernel={2*int(math.ceil(3*smooth))+1})")
    # ─────────────────────────────────────────────────────────────────

    root_pos_fk, joint_rot_mats = _extract_transforms_from_qpos(ki, qpos)

    world_pos, world_rot_mat = _compute_fk(ki, root_pos_fk, joint_rot_mats)
    world_quat = matrix_to_quaternion(world_rot_mat, w_last=True)  # xyzw

    # Re-extract normalized joint angles
    qpos_re = _extract_qpos_from_transforms(ki, root_pos_fk, joint_rot_mats)
    dof_pos = qpos_re[:, 7:]

    # Velocities (computed from smoothed/retimed FK results)
    lin_vel = _compute_cartesian_velocity(world_pos, fps, max_horizon=3)
    ang_vel = _compute_angular_velocity(world_rot_mat, fps, max_horizon=3)
    dof_vel = _compute_cartesian_velocity(dof_pos.unsqueeze(1), fps, max_horizon=1).squeeze(1)

    # Height fix per frame
    body_heights = world_pos[..., 2]
    min_h = body_heights.min(dim=1)[0]
    lift = torch.clamp(0.02 - min_h, min=-0.02)
    trans = torch.zeros(lift.shape[0], 3, device=device)
    trans[:, 2] = lift
    world_pos = world_pos + trans.unsqueeze(1)
    vel_delta = torch.zeros_like(trans)
    vel_delta[:-1] = (trans[1:] - trans[:-1]) * fps
    lin_vel = lin_vel + vel_delta.unsqueeze(1)

    # Global height fix
    global_min = world_pos[..., 2].min()
    world_pos[..., 2] += -global_min + 0.04

    contacts = _compute_contacts(world_pos, lin_vel)

    motion = _build_motion_dict(
        rigid_body_pos=world_pos, rigid_body_rot=world_quat,
        rigid_body_vel=lin_vel, rigid_body_ang_vel=ang_vel,
        dof_pos=dof_pos, dof_vel=dof_vel,
        rigid_body_contacts=contacts, fps=fps,
    )
    return motion


def main():
    parser = argparse.ArgumentParser(
        description="Convert Kimodo G1 CSV(s) to ProtoMotions .pt MotionLib"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to a CSV file or a directory of CSV files")
    parser.add_argument("--output", type=str, default="motions.pt",
                        help="Output .pt file path")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second of the input CSV")
    parser.add_argument("--smooth", type=float, default=1.0,
                        help="Gaussian smoothing sigma in frames (0=off, try 1~5)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Speed factor (0.5=half-speed/2x frames, 2.0=double-speed)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        csv_files = sorted(input_path.glob("*.csv"))
    else:
        csv_files = [input_path]

    if not csv_files:
        print(f"No CSV files found at {args.input}")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV file(s)")
    if args.smooth > 0:
        print(f"  Smoothing: sigma={args.smooth}")
    if args.speed != 1.0:
        print(f"  Speed: {args.speed}x")
    motions = []
    motion_files_list = []

    for csv_file in csv_files:
        print(f"  Converting {csv_file.name} ...")
        motion = convert_csv_to_motion(str(csv_file), fps=args.fps,
                                       smooth=args.smooth, speed=args.speed)
        nf = motion["rigid_body_pos"].shape[0]
        print(f"    {nf} frames, {nf/args.fps:.2f}s")
        motions.append(motion)
        motion_files_list.append(str(csv_file))

    lib = _pack_motionlib(motions, motion_files_list)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(lib, str(out))

    total_frames = sum(m["rigid_body_pos"].shape[0] for m in motions)
    total_time = sum((m["rigid_body_pos"].shape[0] - 1) / m["fps"] for m in motions)
    print(f"\nDone! Saved {len(motions)} motion(s) ({total_frames} frames, {total_time:.2f}s) -> {out}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
