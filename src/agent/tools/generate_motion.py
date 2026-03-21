from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from pathlib import Path

import httpx
from strands import tool

from agent.prompt_refiner import refine_prompt
from g1.publisher import publish_motion

log = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")

# fmt: off
# Default standing joint angles (29 DOFs) — shared with AgentTrackerPolicy
DEFAULT_DOF_POS = [
    -0.312,  0.0,    0.0,   0.669, -0.363,  0.0,    # left leg
    -0.312,  0.0,    0.0,   0.669, -0.363,  0.0,    # right leg
     0.0,    0.0,    0.0,                            # waist
     0.2,    0.2,    0.0,   0.6,   0.0, 0.0, 0.0,   # left arm
     0.2,   -0.2,    0.0,   0.6,   0.0, 0.0, 0.0,   # right arm
]
# fmt: on


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
    constraints: list[dict] | None = None,
) -> tuple[Path, Path | None, bytes | None]:
    """Call Kimodo API and save results as CSV and .pt."""
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
    if constraints:
        body["constraints"] = constraints

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    url = _kimodo_url()

    # CSV endpoint → human-readable qpos record
    csv_response = httpx.post(
        f"{url}/generate/csv",
        headers=KIMODO_HEADERS,
        json=body,
        timeout=120.0,
    )
    csv_response.raise_for_status()
    csv_path = OUTPUT_DIR / f"qpos_{timestamp}.csv"
    csv_path.write_bytes(csv_response.content)

    # PT endpoint → MotionLib tensor for sim and ZMQ (best-effort)
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
def generate_motion(description: str, diffusion_steps: int = 50) -> dict:
    """Generate humanoid robot motion from a natural language description.

    Automatically refines the description into optimized Kimodo prompt(s) and
    calls Kimodo to produce G1 qpos trajectories. Handles multi-step sequences
    by splitting them into separate motion clips.

    Each frame has 36 values: root xyz, root quaternion wxyz, 29 joint angles.
    Use this whenever the user asks for a robot motion, pose, or movement.

    Args:
        description: Natural language description of the desired motion or sequence.
        diffusion_steps: Number of diffusion steps for generation quality (default 50).

    Returns:
        Dictionary with a list of generated motions (qpos_path, prompt, duration,
        num_frames), plus an optional warning if unsupported behavior was approximated.
    """
    refined = refine_prompt(description)
    prompts = refined["prompts"]
    durations = refined["durations"]
    warning = refined.get("warning")

    results = []
    for prompt, duration in zip(prompts, durations, strict=True):
        try:
            qpos_path, pt_path, pt_bytes = _call_kimodo(
                prompt,
                duration,
                diffusion_steps,
                # initial_dof_pos=DEFAULT_DOF_POS,
                # final_dof_pos=DEFAULT_DOF_POS,
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

        motion: dict[str, str | float] = {
            "qpos_path": str(qpos_path),
            "prompt": prompt,
            "duration": duration,
        }
        if pt_path:
            motion["pt_path"] = str(pt_path)
        if pt_bytes:
            publish_motion(
                metadata={"prompt": prompt, "duration": duration},
                pt_bytes=pt_bytes,
            )
        results.append(motion)

    output: dict = {"motions": results}
    if warning:
        output["warning"] = warning
    return output
