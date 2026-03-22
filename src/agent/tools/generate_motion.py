from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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

DEFAULT_ROOT_HEIGHT = 0.75
KIMODO_FPS = 30
KIMODO_HEADERS = {"ngrok-skip-browser-warning": "true"}

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
    """MuJoCo ground plane (X=fwd, Y=left) → Kimodo smooth_root_2d."""
    return [mujoco_y, mujoco_x]


def _build_final_root_pos_from_xy(
    final_root_x: float | None,
    final_root_y: float | None,
    *,
    fallback: list[float] | None = None,
) -> list[float] | None:
    if final_root_x is None and final_root_y is None:
        return fallback

    base_x = float(fallback[0]) if fallback is not None else 0.0
    base_y = float(fallback[1]) if fallback is not None else 0.0
    return [
        float(final_root_x) if final_root_x is not None else base_x,
        float(final_root_y) if final_root_y is not None else base_y,
        DEFAULT_ROOT_HEIGHT,
    ]


def _build_root2d_constraints(
    direction: str,
    distance: float,
    duration: float,
    waypoint_interval: int = 10,
) -> list[dict[str, Any]] | None:
    """Build dense locomotion root2d constraints for Kimodo."""
    vec = _DIRECTION_VECTORS.get(direction.lower().strip())
    if vec is None:
        return None

    dx, dy = vec
    total_x, total_y = dx * distance, dy * distance

    num_frames = int(duration * KIMODO_FPS)
    last = max(num_frames - 1, 1)

    frame_indices = list(range(0, last, waypoint_interval))
    if not frame_indices or frame_indices[-1] != last:
        frame_indices.append(last)

    smooth_root_2d = []
    for frame_idx in frame_indices:
        t = frame_idx / last
        smooth_root_2d.append(_mujoco_to_kimodo_2d(total_x * t, total_y * t))

    constraint: dict[str, Any] = {
        "type": "root2d",
        "frame_indices": frame_indices,
        "smooth_root_2d": smooth_root_2d,
    }
    log.info(
        "root2d locomotion constraint: %d waypoints, start=%s, end=%s",
        len(frame_indices),
        smooth_root_2d[0],
        smooth_root_2d[-1],
    )
    return [constraint]


def _build_final_root2d_constraint(
    final_root_pos: list[float],
    duration: float,
) -> list[dict[str, Any]]:
    """Constrain only the final base-frame XY position at the last timestep."""
    num_frames = int(duration * KIMODO_FPS)
    last = max(num_frames - 1, 1)
    target = _mujoco_to_kimodo_2d(final_root_pos[0], final_root_pos[1])
    constraint: dict[str, Any] = {
        "type": "root2d",
        "frame_indices": [last],
        "smooth_root_2d": [target],
    }
    log.info(
        "root2d final target constraint: frame=%d, target=%s",
        last,
        target,
    )
    return [constraint]


def _kimodo_url() -> str:
    return os.environ.get("KIMODO_URL", "http://localhost:8420")


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
    constraints: list[dict[str, Any]] | None = None,
) -> tuple[Path, Path | None, bytes | None]:
    """Call Kimodo API and save results as CSV and .pt."""
    body: dict[str, Any] = {
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
        key
        for key in ("initial_dof_pos", "final_dof_pos", "constraints")
        if key in body
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
    final_root_x: float | None = None,
    final_root_y: float | None = None,
) -> dict[str, object]:
    """Generate humanoid robot motion from a natural language description.

    Automatically refines the description into optimized Kimodo prompt(s) and
    calls Kimodo to produce G1 qpos trajectories.

    Each frame has 36 values: root xyz, root quaternion wxyz, 29 joint angles.
    Use this whenever the user asks for a robot motion, pose, or movement.
    """
    return generate_motion_impl(
        description,
        diffusion_steps=diffusion_steps,
        return_to_standing=return_to_standing,
        move_direction=move_direction,
        move_distance=move_distance,
        final_root_x=final_root_x,
        final_root_y=final_root_y,
    )


def generate_motion_impl(
    description: str,
    diffusion_steps: int = 50,
    *,
    return_to_standing: bool = True,
    move_direction: str = "",
    move_distance: float = 0.5,
    final_root_x: float | None = None,
    final_root_y: float | None = None,
    final_root_pos: list[float] | None = None,
    initial_dof_pos: list[float] | None = None,
    final_dof_pos: list[float] | None = None,
    num_samples: int = 1,
    num_transition_frames: int = 5,
    cfg_type: str = "regular",
    cfg_weight: list[float] | None = None,
    constraints: list[dict[str, Any]] | None = None,
) -> dict[str, object]:
    """Shared implementation for motion generation across CLI and web flows."""
    refined = refine_prompt(description)
    prompts = refined["prompts"]
    durations = refined["durations"]
    warning = refined.get("warning")

    resolved_final_root_pos = _build_final_root_pos_from_xy(
        final_root_x,
        final_root_y,
        fallback=final_root_pos,
    )

    has_explicit_final_root = resolved_final_root_pos is not None
    has_locomotion = bool(move_direction and move_direction.strip()) and not has_explicit_final_root

    constraints_applied: list[str] = []
    if has_locomotion:
        constraints_applied.append(
            f"locomotion:{move_direction.strip()}:{move_distance:.1f}m"
        )
    if has_explicit_final_root:
        constraints_applied.append(
            "final_root_xy:"
            f"{resolved_final_root_pos[0]:.2f}:"
            f"{resolved_final_root_pos[1]:.2f}"
        )

    use_final_dof = return_to_standing and not has_locomotion and not has_explicit_final_root
    resolved_final_dof_pos = final_dof_pos
    if use_final_dof and resolved_final_dof_pos is None:
        resolved_final_dof_pos = DEFAULT_DOF_POS
        constraints_applied.append("return_to_standing")

    log.info(
        "Constraints: %s | final_dof=%s | root_target=%s",
        constraints_applied or "none",
        "standing" if resolved_final_dof_pos is not None else "free",
        resolved_final_root_pos,
    )

    results: list[dict[str, object]] = []
    for prompt, duration in zip(prompts, durations, strict=True):
        request_constraints = list(constraints or [])
        if has_explicit_final_root and resolved_final_root_pos is not None:
            request_constraints.extend(
                _build_final_root2d_constraint(resolved_final_root_pos, duration)
            )
        elif has_locomotion:
            root2d = _build_root2d_constraints(move_direction, move_distance, duration)
            if root2d is not None:
                request_constraints.extend(root2d)

        try:
            qpos_path, pt_path, _pt_bytes = _call_kimodo(
                prompt,
                duration,
                diffusion_steps,
                initial_dof_pos=initial_dof_pos,
                final_dof_pos=resolved_final_dof_pos,
                num_samples=num_samples,
                num_transition_frames=num_transition_frames,
                cfg_type=cfg_type,
                cfg_weight=cfg_weight,
                constraints=request_constraints or None,
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

        motion: dict[str, object] = {
            "qpos_path": str(qpos_path),
            "prompt": prompt,
            "duration": duration,
        }
        if pt_path:
            motion["pt_path"] = str(pt_path)
        results.append(motion)

    output: dict[str, object] = {"motions": results}
    if constraints_applied:
        output["constraints_applied"] = constraints_applied
    if warning:
        output["warning"] = warning
    return output
