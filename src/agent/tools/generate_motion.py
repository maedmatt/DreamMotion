from __future__ import annotations

import logging
import os
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
            default standing pose so the robot returns to a safe idle position.
            Set to False only when the user explicitly wants to hold a final pose.
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
        root2d = (
            _build_root2d_constraints(move_direction, move_distance, duration)
            if has_locomotion
            else None
        )
        use_final_dof = return_to_standing and not has_locomotion
        try:
            qpos_path, pt_path, _pt_bytes = _call_kimodo(
                prompt,
                duration,
                diffusion_steps,
                final_dof_pos=(DEFAULT_DOF_POS if use_final_dof else None),
                constraints=root2d,
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
