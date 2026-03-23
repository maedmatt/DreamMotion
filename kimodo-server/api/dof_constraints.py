"""
Constraint helpers for the Kimodo API.
Converts convenience fields (dof_pos) and raw constraint JSON into
constraint_lst for the Kimodo model.

Supported constraint types (raw pass-through):
  - root2d: 2D root waypoints
  - fullbody: full-body keyframe poses
  - left-hand / right-hand / left-foot / right-foot: end-effector
  - end-effector: generic end-effector with joint_names

Convenience fields:
  - initial_dof_pos: 29 DOF angles -> fullbody constraint on frame 0
  - final_dof_pos: 29 DOF angles -> fullbody constraint on last frame
"""
from typing import List, Optional

import torch

from kimodo.geometry import matrix_to_axis_angle


def mujoco_root_quat_to_kimodo_aa(quat_wxyz: List[float]) -> List[float]:
    """Convert a MuJoCo root quaternion (w,x,y,z) to Kimodo-space axis-angle.

    MuJoCo is Z-up/X-forward; Kimodo is Y-up/Z-forward.
    R_kimodo = M2K @ R_mujoco @ M2K^T where M2K is the coordinate transform.
    """
    from kimodo.geometry import quaternion_to_matrix

    q = [float(v) for v in quat_wxyz]
    R_muj = quaternion_to_matrix(torch.tensor([q], dtype=torch.float32)).squeeze(0)
    m2k = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    R_kim = m2k @ R_muj @ m2k.T
    aa = matrix_to_axis_angle(R_kim.unsqueeze(0)).squeeze(0)
    return aa.tolist()


def mujoco_root_pos_to_kimodo(pos_xyz: List[float]) -> List[float]:
    """Convert MuJoCo root position (x,y,z) Z-up to Kimodo (x,y,z) Y-up."""
    x, y, z = [float(v) for v in pos_xyz]
    return [y, z, x]


def dof_to_axis_angle_34(dof_pos: List[float], skeleton=None, converter=None) -> List[List[float]]:
    """Convert 29 MuJoCo DOF angles (radians) to 34-joint axis-angle representation.

    Uses MujocoQposConverter for correct coordinate-space axis transforms
    and body quaternion offsets from the MuJoCo XML.

    Args:
        dof_pos: 29 MuJoCo hinge-joint angles in radians (0 = rest/T-pose).
        skeleton: G1Skeleton34 instance (created if None).
        converter: MujocoQposConverter instance (created if None).

    Returns:
        34-element list of [ax, ay, az] axis-angle vectors in Kimodo space.
    """
    assert len(dof_pos) == 29, f"Expected 29 DOF values, got {len(dof_pos)}"

    if skeleton is None:
        from kimodo.skeleton import G1Skeleton34
        skeleton = G1Skeleton34()
    if converter is None:
        from kimodo.exports.mujoco import MujocoQposConverter
        converter = MujocoQposConverter(skeleton)

    mujoco_dofs = torch.tensor(dof_pos, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    rest_dofs = converter._rest_dofs_axis_angle.unsqueeze(0).unsqueeze(0)
    kimodo_dofs = mujoco_dofs + rest_dofs

    nb_joints = skeleton.nbjoints
    identity = torch.eye(3, dtype=torch.float32).unsqueeze(0).expand(nb_joints, -1, -1)
    local_rot_mats = identity.unsqueeze(0).unsqueeze(0)

    local_rot_mats = converter._joint_dofs_to_local_rot_mats(
        kimodo_dofs, local_rot_mats,
        device=torch.device("cpu"), dtype=torch.float32,
        use_relative=False,
    )

    aa = matrix_to_axis_angle(local_rot_mats[0, 0])
    return aa.tolist()


def _build_fullbody_constraint(
    frame_idx: int,
    dof_pos: List[float],
    skeleton,
    converter,
    root_pos: Optional[List[float]] = None,
    root_quat: Optional[List[float]] = None,
    root_height: float = 0.75,
) -> dict:
    """Build a single fullbody constraint dict for one keyframe.

    Args:
        frame_idx: Target frame index.
        dof_pos: 29 MuJoCo DOF angles (radians).
        skeleton: G1Skeleton34 instance.
        converter: MujocoQposConverter instance.
        root_pos: MuJoCo root position [x,y,z] (Z-up). Converted to Kimodo Y-up.
        root_quat: MuJoCo root quaternion [w,x,y,z]. Converted to Kimodo axis-angle.
        root_height: Fallback root height in Kimodo Y-up when root_pos is not given.
    """
    aa_34 = dof_to_axis_angle_34(dof_pos, skeleton, converter)

    if root_quat is not None:
        aa_34[0] = mujoco_root_quat_to_kimodo_aa(root_quat)

    if root_pos is not None:
        kim_pos = mujoco_root_pos_to_kimodo(root_pos)
    else:
        kim_pos = [0.0, root_height, 0.0]

    return {
        "type": "fullbody",
        "frame_indices": [frame_idx],
        "local_joints_rot": [aa_34],
        "root_positions": [kim_pos],
    }


def build_constraint_list(
    num_frames: int,
    initial_dof_pos: Optional[List[float]] = None,
    final_dof_pos: Optional[List[float]] = None,
    constraints: Optional[List[dict]] = None,
    root_height: float = 0.75,
    skeleton=None,
    converter=None,
    initial_root_pos: Optional[List[float]] = None,
    initial_root_quat: Optional[List[float]] = None,
    final_root_pos: Optional[List[float]] = None,
    final_root_quat: Optional[List[float]] = None,
) -> list:
    """Build a combined list of constraint dicts for Kimodo.

    Merges convenience fields (initial/final_dof_pos) with raw constraints.
    Raw constraints are passed through as-is to load_constraints_lst.
    """
    result = []

    if initial_dof_pos is not None:
        result.append(_build_fullbody_constraint(
            0, initial_dof_pos, skeleton, converter,
            root_pos=initial_root_pos, root_quat=initial_root_quat,
            root_height=root_height,
        ))

    if final_dof_pos is not None:
        result.append(_build_fullbody_constraint(
            num_frames - 1, final_dof_pos, skeleton, converter,
            root_pos=final_root_pos, root_quat=final_root_quat,
            root_height=root_height,
        ))

    if constraints is not None:
        result.extend(constraints)

    return result


def load_all_constraints(
    skeleton, num_frames,
    initial_dof_pos=None, final_dof_pos=None,
    constraints=None,
    initial_root_pos=None, initial_root_quat=None,
    final_root_pos=None, final_root_quat=None,
):
    """Build and load all constraints, returning constraint_lst for model()."""
    from kimodo.constraints import load_constraints_lst
    from kimodo.exports.mujoco import MujocoQposConverter

    converter = MujocoQposConverter(skeleton)
    cjson = build_constraint_list(
        num_frames, initial_dof_pos, final_dof_pos, constraints,
        skeleton=skeleton, converter=converter,
        initial_root_pos=initial_root_pos, initial_root_quat=initial_root_quat,
        final_root_pos=final_root_pos, final_root_quat=final_root_quat,
    )
    if not cjson:
        return []

    return load_constraints_lst(cjson, skeleton)
