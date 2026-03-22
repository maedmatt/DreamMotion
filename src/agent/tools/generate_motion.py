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
# Default standing joint angles (29 DOFs) - shared with AgentTrackerPolicy
DEFAULT_DOF_POS = [
    -0.312,  0.0,    0.0,   0.669, -0.363,  0.0,    # left leg
    -0.312,  0.0,    0.0,   0.669, -0.363,  0.0,    # right leg
     0.0,    0.0,    0.0,                            # waist
     0.2,    0.2,    0.0,   0.6,   0.0, 0.0, 0.0,   # left arm
     0.2,   -0.2,    0.0,   0.6,   0.0, 0.0, 0.0,   # right arm
]
# fmt: on

# Default root: standing at origin, Z-up (MuJoCo convention)
DEFAULT_ROOT_HEIGHT = 0.75
INITIAL_ROOT_POS = [0.0, 0.0, DEFAULT_ROOT_HEIGHT]
FINAL_ROOT_POS = [2.0, 0.0, DEFAULT_ROOT_HEIGHT]
DEFAULT_ROOT_QUAT = [1.0, 0.0, 0.0, 0.0]  # identity wxyz


KIMODO_HEADERS = {"ngrok-skip-browser-warning": "true"}


def _kimodo_url() -> str:
    return os.environ.get("KIMODO_URL", "http://localhost:8420")


def _call_kimodo(
    prompt: str,
    duration: float,
    diffusion_steps: int,
    initial_dof_pos: list[float] | None = None,
    final_dof_pos: list[float] | None = None,
    initial_root_pos: list[float] | None = None,
    initial_root_quat: list[float] | None = None,
    final_root_pos: list[float] | None = None,
    final_root_quat: list[float] | None = None,
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
    if initial_root_pos is not None:
        body["initial_root_pos"] = initial_root_pos
    if initial_root_quat is not None:
        body["initial_root_quat"] = initial_root_quat
    if final_root_pos is not None:
        body["final_root_pos"] = final_root_pos
    if final_root_quat is not None:
        body["final_root_quat"] = final_root_quat
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
def generate_motion(description: str, diffusion_steps: int = 50) -> dict[str, object]:
    """Generate humanoid robot motion from a natural language description."""
    return generate_motion_impl(description, diffusion_steps=diffusion_steps)


def generate_motion_impl(
    description: str,
    diffusion_steps: int = 50,
    *,
    initial_dof_pos: list[float] | None = None,
    final_dof_pos: list[float] | None = None,
    initial_root_pos: list[float] | None = None,
    initial_root_quat: list[float] | None = None,
    final_root_pos: list[float] | None = None,
    final_root_quat: list[float] | None = None,
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

    resolved_initial_dof_pos = DEFAULT_DOF_POS if initial_dof_pos is None else initial_dof_pos
    resolved_final_dof_pos = DEFAULT_DOF_POS if final_dof_pos is None else final_dof_pos
    resolved_initial_root_pos = INITIAL_ROOT_POS if initial_root_pos is None else initial_root_pos
    resolved_initial_root_quat = DEFAULT_ROOT_QUAT if initial_root_quat is None else initial_root_quat
    resolved_final_root_pos = FINAL_ROOT_POS if final_root_pos is None else final_root_pos
    resolved_final_root_quat = DEFAULT_ROOT_QUAT if final_root_quat is None else final_root_quat

    results: list[dict[str, object]] = []
    for prompt, duration in zip(prompts, durations, strict=True):
        try:
            qpos_path, pt_path, _pt_bytes = _call_kimodo(
                prompt,
                duration,
                diffusion_steps,
                initial_dof_pos=resolved_initial_dof_pos,
                final_dof_pos=resolved_final_dof_pos,
                initial_root_pos=resolved_initial_root_pos,
                initial_root_quat=resolved_initial_root_quat,
                final_root_pos=resolved_final_root_pos,
                final_root_quat=resolved_final_root_quat,
                num_samples=num_samples,
                num_transition_frames=num_transition_frames,
                cfg_type=cfg_type,
                cfg_weight=cfg_weight,
                constraints=constraints,
            )
        except httpx.HTTPError:
            log.warning("Kimodo call failed for prompt: %s", prompt, exc_info=True)
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
    if warning:
        output["warning"] = warning
    return output
