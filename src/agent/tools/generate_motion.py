from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import httpx
from strands import tool

KIMODO_URL = os.environ.get("KIMODO_URL", "http://localhost:8420")
OUTPUT_DIR = Path("output")

# 36 values per frame: root position (3), root quaternion wxyz (4), joint angles (29)
CSV_HEADER = "root_x,root_y,root_z,quat_w,quat_x,quat_y,quat_z," + ",".join(
    f"joint_{i}" for i in range(29)
)


@tool
def generate_motion(
    prompt: str, duration: float = 2.0, diffusion_steps: int = 50
) -> dict:
    """Generate humanoid robot motion from a text description.

    Calls Kimodo to produce a G1 qpos trajectory.
    Each frame has 36 values: root xyz, root quaternion wxyz, 29 joint angles.
    Use this whenever the user asks for a robot motion, pose, or movement.

    Args:
        prompt: Natural language description of the desired motion.
        duration: Length of the motion in seconds (default 2.0).
        diffusion_steps: Number of diffusion steps for generation quality (default 50).

    Returns:
        Dictionary with qpos_path (saved CSV), num_frames, and duration.
    """
    response = httpx.post(
        f"{KIMODO_URL}/generate",
        json={
            "prompt": prompt,
            "duration": duration,
            "diffusion_steps": diffusion_steps,
        },
        timeout=120.0,
    )
    response.raise_for_status()
    data = response.json()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    qpos = data["qpos"]
    qpos_path = OUTPUT_DIR / f"qpos_{timestamp}.csv"
    lines = [CSV_HEADER]
    lines.extend(",".join(str(v) for v in frame) for frame in qpos)
    qpos_path.write_text("\n".join(lines))

    return {
        "qpos_path": str(qpos_path),
        "num_frames": len(qpos),
        "duration": duration,
    }
