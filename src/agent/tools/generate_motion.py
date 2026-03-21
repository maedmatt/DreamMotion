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

KIMODO_URL = os.environ.get("KIMODO_URL", "http://localhost:8420")
OUTPUT_DIR = Path("output")

# 36 values per frame: root position (3), root quaternion wxyz (4), joint angles (29)
CSV_HEADER = "root_x,root_y,root_z,quat_w,quat_x,quat_y,quat_z," + ",".join(
    f"joint_{i}" for i in range(29)
)


def _call_kimodo(
    prompt: str, duration: float, diffusion_steps: int
) -> tuple[Path, Path | None, bytes | None]:
    """Call Kimodo API for a single prompt and save qpos as CSV and .pt."""
    body = {"prompt": prompt, "duration": duration, "diffusion_steps": diffusion_steps}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")

    # JSON endpoint → CSV
    response = httpx.post(
        f"{KIMODO_URL}/generate",
        json=body,
        timeout=120.0,
    )
    response.raise_for_status()
    data = response.json()

    qpos = data["qpos"]
    qpos_path = OUTPUT_DIR / f"qpos_{timestamp}.csv"
    lines = [CSV_HEADER]
    lines.extend(",".join(str(v) for v in frame) for frame in qpos)
    qpos_path.write_text("\n".join(lines))

    # PT endpoint → raw tensor (best-effort)
    pt_path = None
    pt_bytes = None
    try:
        pt_response = httpx.post(
            f"{KIMODO_URL}/generate/pt",
            json=body,
            timeout=120.0,
        )
        pt_response.raise_for_status()
        pt_bytes = pt_response.content
        pt_path = OUTPUT_DIR / f"qpos_{timestamp}.pt"
        pt_path.write_bytes(pt_bytes)
    except httpx.HTTPError:
        log.warning("Failed to fetch .pt from Kimodo, skipping")

    return qpos_path, pt_path, pt_bytes


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
                prompt, duration, diffusion_steps
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
