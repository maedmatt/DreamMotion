from __future__ import annotations

import logging
import math
import os
import re
from datetime import UTC, datetime
from pathlib import Path

import httpx
from strands import tool

from agent.prompt_refiner import refine_prompt

log = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")

# fmt: off
# Default standing joint angles (29 DOFs) — shared with AgentTrackerPolicy
DEFAULT_DOF_POS: list[float] = [
    -0.312,  0.0,    0.0,   0.669, -0.363,  0.0,    # left leg
    -0.312,  0.0,    0.0,   0.669, -0.363,  0.0,    # right leg
     0.0,    0.0,    0.0,                            # waist
     0.2,    0.2,    0.0,   0.6,   0.0, 0.0, 0.0,   # left arm
     0.2,   -0.2,    0.0,   0.6,   0.0, 0.0, 0.0,   # right arm
]
# fmt: on

KIMODO_FPS = 30

# fmt: off
# Standing pose in Kimodo axis-angle format (34 joints x 3).
# Pre-computed from DEFAULT_DOF_POS via MujocoQposConverter.
_STANDING_LOCAL_JOINTS_ROT: list[list[float]] = [
    [+0.000000, +0.000000, +0.000000],  # pelvis
    [-0.312000, +0.000000, +0.000000],  # left_hip_pitch
    [-0.174900, +0.000000, +0.000000],  # left_hip_roll
    [+0.000000, +0.000000, +0.000000],  # left_hip_yaw
    [+0.843899, +0.000000, +0.000000],  # left_knee
    [-0.363000, +0.000000, +0.000000],  # left_ankle_pitch
    [+0.000000, +0.000000, +0.000000],  # left_ankle_roll
    [+0.000000, +0.000000, +0.000000],  # left_toe_base
    [-0.312000, +0.000000, +0.000000],  # right_hip_pitch
    [-0.174900, +0.000000, +0.000000],  # right_hip_roll
    [+0.000000, +0.000000, +0.000000],  # right_hip_yaw
    [+0.843899, +0.000000, +0.000000],  # right_knee
    [-0.363000, +0.000000, +0.000000],  # right_ankle_pitch
    [+0.000000, +0.000000, +0.000000],  # right_ankle_roll
    [+0.000000, +0.000000, +0.000000],  # right_toe_base
    [+0.000000, +0.000000, +0.000000],  # waist_yaw
    [+0.000000, +0.000000, +0.000000],  # waist_roll
    [+0.000000, +0.000000, +0.000000],  # waist_pitch
    [+0.198723, -0.028128, +0.278355],  # left_shoulder_pitch
    [+0.000000, +0.000000, -0.079252],  # left_shoulder_roll
    [+0.000000, +0.000000, +0.000000],  # left_shoulder_yaw
    [+0.600000, +0.000000, +0.000000],  # left_elbow
    [+0.000000, +0.000000, +0.000000],  # left_wrist_roll
    [+0.000000, +0.000000, +0.000000],  # left_wrist_pitch
    [+0.000000, +0.000000, +0.000000],  # left_wrist_yaw
    [+0.000000, +0.000000, +0.000000],  # left_hand_roll
    [+0.198723, +0.028128, -0.278355],  # right_shoulder_pitch
    [+0.000000, +0.000000, +0.079252],  # right_shoulder_roll
    [+0.000000, +0.000000, +0.000000],  # right_shoulder_yaw
    [+0.600000, +0.000000, +0.000000],  # right_elbow
    [+0.000000, +0.000000, +0.000000],  # right_wrist_roll
    [+0.000000, +0.000000, +0.000000],  # right_wrist_pitch
    [+0.000000, +0.000000, +0.000000],  # right_wrist_yaw
    [+0.000000, +0.000000, +0.000000],  # right_hand_roll
]
# fmt: on

_STANDING_ROOT_HEIGHT_KIMODO = 0.7559

# Direction → (dx, dy) unit vectors in MuJoCo ground plane (X=forward, Y=left)
_DIRECTION_VECTORS: dict[str, tuple[float, float]] = {
    "forward": (1.0, 0.0),
    "backward": (-1.0, 0.0),
    "left": (0.0, 1.0),
    "right": (0.0, -1.0),
    "forward-left": (0.707, 0.707),
    "forward-right": (0.707, -0.707),
    "backward-left": (-0.707, 0.707),
    "backward-right": (-0.707, -0.707),
}

# Direction → yaw angle (radians) in Kimodo Y-up coords (rotation around Y).
# Kimodo default forward is +Z; yaw>0 rotates toward +X (= MuJoCo left).
_DIRECTION_YAWS: dict[str, float] = {
    "forward": 0.0,
    "backward": math.pi,
    "left": math.pi / 2,
    "right": -math.pi / 2,
    "forward-left": math.pi / 4,
    "forward-right": -math.pi / 4,
    "backward-left": 3 * math.pi / 4,
    "backward-right": -3 * math.pi / 4,
}


_DISTANCE_RE = re.compile(r"(\d+\.?\d*)\s*(?:m(?:eter)?s?\b|米)", re.IGNORECASE)


def extract_motion_params(
    prompt: str,
) -> tuple[str, float]:
    """Extract (move_direction, move_distance) from a free-text prompt.

    Returns ("", 0.5) when no locomotion direction is detected.
    Checks compound directions first (e.g. "forward-left") then single.
    """
    text = prompt.lower()

    direction = ""
    for d in sorted(_DIRECTION_VECTORS, key=len, reverse=True):
        if d in text:
            direction = d
            break

    distance = 0.5
    m = _DISTANCE_RE.search(text)
    if m:
        distance = float(m.group(1))

    return direction, distance


def _mujoco_to_kimodo_2d(mujoco_x: float, mujoco_y: float) -> list[float]:
    """MuJoCo ground plane (X=fwd, Y=left) → Kimodo smooth_root_2d.

    Kimodo is Y-up, Z-forward. The mujoco_to_kimodo matrix is:
      kimodo_x = mujoco_y, kimodo_y = mujoco_z, kimodo_z = mujoco_x.
    smooth_root_2d = [x_kimodo, z_kimodo] = [mujoco_y, mujoco_x].
    """
    return [mujoco_y, mujoco_x]


def _build_root2d_constraints(
    direction: str,
    distance: float,
    duration: float,
    waypoint_interval: int = 10,
) -> list[dict] | None:
    """Build root2d constraint dicts for the ``constraints`` API field.

    Creates dense waypoints every *waypoint_interval* frames along a
    linear trajectory from origin to the target position.
    """
    vec = _DIRECTION_VECTORS.get(direction.lower().strip())
    if vec is None:
        return None

    dx, dy = vec
    total_x, total_y = dx * distance, dy * distance

    num_frames = int(duration * KIMODO_FPS)
    last = max(num_frames - 1, 1)

    frame_indices: list[int] = list(range(0, last, waypoint_interval))
    if frame_indices[-1] != last:
        frame_indices.append(last)

    smooth_root_2d: list[list[float]] = []
    for f in frame_indices:
        t = f / last
        smooth_root_2d.append(_mujoco_to_kimodo_2d(total_x * t, total_y * t))

    constraint: dict = {
        "type": "root2d",
        "frame_indices": frame_indices,
        "smooth_root_2d": smooth_root_2d,
    }
    log.info(
        "root2d constraint: %d waypoints, start=%s, end=%s",
        len(frame_indices),
        smooth_root_2d[0],
        smooth_root_2d[-1],
    )
    return [constraint]


def _build_standing_fullbody_constraint(
    last_frame: int,
    mujoco_target_x: float = 0.0,
    mujoco_target_y: float = 0.0,
    yaw: float = 0.0,
) -> dict:
    """Build a fullbody constraint on *last_frame* for the default standing pose.

    Unlike ``final_dof_pos`` (which internally derives root from FK at origin),
    this explicitly sets ``root_positions`` and ``smooth_root_2d`` to the
    expected final base location, avoiding the root-pulled-to-origin conflict.

    ``yaw`` (radians, Kimodo Y-up) rotates the pelvis so the robot faces the
    direction of travel at the end of locomotion.
    """
    kimodo_x = mujoco_target_y
    kimodo_z = mujoco_target_x
    kimodo_y = _STANDING_ROOT_HEIGHT_KIMODO

    joints_rot = [row[:] for row in _STANDING_LOCAL_JOINTS_ROT]
    if abs(yaw) > 1e-6:
        joints_rot[0] = [0.0, yaw, 0.0]

    constraint: dict = {
        "type": "fullbody",
        "frame_indices": [last_frame],
        "local_joints_rot": [joints_rot],
        "root_positions": [[kimodo_x, kimodo_y, kimodo_z]],
        "smooth_root_2d": [[kimodo_x, kimodo_z]],
    }
    log.info(
        "fullbody standing: frame=%d, root=[%.3f,%.3f,%.3f], yaw=%.2f°",
        last_frame,
        kimodo_x,
        kimodo_y,
        kimodo_z,
        math.degrees(yaw),
    )
    return constraint


def _kimodo_url() -> str:
    return os.environ.get("KIMODO_URL", "http://localhost:8420")


KIMODO_HEADERS = {"ngrok-skip-browser-warning": "true"}


def _call_kimodo(
    prompt: str,
    duration: float,
    diffusion_steps: int,
    initial_dof_pos: list[float] | None = None,
    final_dof_pos: list[float] | None = None,
    num_samples: int = 1,
    num_transition_frames: int = 5,
    cfg_type: str = "regular",
    cfg_weight: list[float] | None = None,
    constraints: list[dict] | None = None,
) -> tuple[Path, Path | None, bytes | None]:
    """Call Kimodo API and save results as CSV and .pt.

    Supported constraint params (per Kimodo API):
      - initial_dof_pos / final_dof_pos: soft joint-angle guidance
      - constraints: raw Kimodo constraint dicts (root2d, etc.)
    """
    body: dict = {
        "prompt": prompt,
        "duration": duration,
        "diffusion_steps": diffusion_steps,
    }
    if initial_dof_pos is not None:
        body["initial_dof_pos"] = initial_dof_pos
    if final_dof_pos is not None:
        body["final_dof_pos"] = final_dof_pos
    if num_samples != 1:
        body["num_samples"] = num_samples
    if num_transition_frames != 5:
        body["num_transition_frames"] = num_transition_frames
    if cfg_type != "regular":
        body["cfg_type"] = cfg_type
    if cfg_weight is not None:
        body["cfg_weight"] = cfg_weight
    if constraints:
        body["constraints"] = constraints

    constraint_keys = [
        k for k in ("initial_dof_pos", "final_dof_pos", "constraints") if k in body
    ]
    log.info("Kimodo request constraints: %s", constraint_keys)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    url = _kimodo_url()

    csv_response = httpx.post(
        f"{url}/generate/csv",
        headers=KIMODO_HEADERS,
        json=body,
        timeout=120.0,
    )
    csv_response.raise_for_status()
    csv_path = OUTPUT_DIR / f"qpos_{timestamp}.csv"
    csv_path.write_bytes(csv_response.content)

    pt_path = None
    pt_bytes = None
    try:
        pt_response = httpx.post(
            f"{url}/generate/pt",
            headers=KIMODO_HEADERS,
            json=body,
            timeout=120.0,
        )
        pt_response.raise_for_status()
        pt_bytes = pt_response.content
        pt_path = OUTPUT_DIR / f"qpos_{timestamp}.pt"
        pt_path.write_bytes(pt_bytes)
    except httpx.HTTPError:
        log.warning("Failed to fetch .pt from Kimodo, skipping")

    return csv_path, pt_path, pt_bytes


@tool
def generate_motion(
    description: str,
    diffusion_steps: int = 50,
    return_to_standing: bool = True,
    move_direction: str = "",
    move_distance: float = 0.5,
) -> dict:
    """Generate humanoid robot motion from a natural language description.

    Automatically refines the description into optimized Kimodo prompt(s) and
    calls Kimodo to produce G1 qpos trajectories.

    Each frame has 36 values: root xyz, root quaternion wxyz, 29 joint angles.
    Use this whenever the user asks for a robot motion, pose, or movement.

    Args:
        description: Natural language description of the desired motion or sequence.
        diffusion_steps: Number of diffusion steps for generation quality (default 50).
        return_to_standing: If True (default), constrain the last frame to the G1
            default standing pose using a fullbody constraint that explicitly sets
            both joint angles AND the expected final base position (destination for
            locomotion, origin for in-place). Set to False only when the user
            explicitly wants to hold a non-standing final pose (e.g. sitting,
            kneeling, arms raised).
        move_direction: Locomotion direction for root position constraints. One of:
            "forward", "backward", "left", "right", "forward-left", "forward-right",
            "backward-left", "backward-right". Leave empty for in-place motions.
        move_distance: Total locomotion distance in meters (default 0.5). Only used
            when move_direction is set. Keep conservative (0.3-0.5 m) unless the
            user specifies an exact distance.

    Returns:
        Dictionary with a list of generated motions (qpos_path, prompt, duration,
        num_frames), plus an optional warning if unsupported behavior was approximated.
    """
    return generate_motion_impl(
        description,
        diffusion_steps=diffusion_steps,
        return_to_standing=return_to_standing,
        move_direction=move_direction,
        move_distance=move_distance,
    )


def generate_motion_impl(
    description: str,
    diffusion_steps: int = 50,
    return_to_standing: bool = True,
    move_direction: str = "",
    move_distance: float = 0.5,
) -> dict:
    """Shared implementation for motion generation across CLI and web agent flows."""
    refined = refine_prompt(description)
    prompts = refined["prompts"]
    durations = refined["durations"]
    warning = refined.get("warning")

    constraints_applied: list[str] = []
    if return_to_standing:
        constraints_applied.append("return_to_standing")

    has_locomotion = bool(move_direction and move_direction.strip())
    if has_locomotion:
        constraints_applied.append(
            f"locomotion:{move_direction.strip()}:{move_distance:.1f}m"
        )

    log.info("Constraints: %s", constraints_applied or "none")

    results = []
    for prompt, duration in zip(prompts, durations, strict=True):
        all_constraints: list[dict] = []
        num_frames = int(duration * KIMODO_FPS)
        last_frame = max(num_frames - 1, 1)

        if has_locomotion:
            root2d = _build_root2d_constraints(move_direction, move_distance, duration)
            if root2d:
                all_constraints.extend(root2d)

        if return_to_standing:
            dir_key = move_direction.lower().strip()
            vec = (
                _DIRECTION_VECTORS.get(dir_key, (0.0, 0.0))
                if has_locomotion
                else (0.0, 0.0)
            )
            target_x = vec[0] * move_distance if has_locomotion else 0.0
            target_y = vec[1] * move_distance if has_locomotion else 0.0
            yaw = _DIRECTION_YAWS.get(dir_key, 0.0) if has_locomotion else 0.0
            all_constraints.append(
                _build_standing_fullbody_constraint(
                    last_frame, target_x, target_y, yaw=yaw
                )
            )

        try:
            qpos_path, pt_path, _pt_bytes = _call_kimodo(
                prompt,
                duration,
                diffusion_steps,
                constraints=all_constraints or None,
            )
        except httpx.HTTPError:
            log.warning(
                "Kimodo call failed for prompt: %s",
                prompt,
                exc_info=True,
            )
            results.append(
                {
                    "prompt": prompt,
                    "duration": duration,
                    "error": "Kimodo generation failed",
                }
            )
            continue

        motion: dict[str, str | float] = {
            "qpos_path": str(qpos_path),
            "prompt": prompt,
            "duration": duration,
        }
        if pt_path:
            motion["pt_path"] = str(pt_path)
        results.append(motion)

    output: dict = {"motions": results}
    if constraints_applied:
        output["constraints_applied"] = constraints_applied
    if warning:
        output["warning"] = warning
    return output
