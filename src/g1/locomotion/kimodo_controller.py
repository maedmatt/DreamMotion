from __future__ import annotations

import math
import os

import httpx

KIMODO_URL = os.environ.get("KIMODO_URL", "http://localhost:8420")

# Average walking speed for duration estimation (m/s).
_AVG_WALK_SPEED = 0.5


def walk_to_point_kimodo(
    target_x: float,
    target_y: float,
    current_x: float,
    current_y: float,
    stop_short_m: float = 0.5,
    diffusion_steps: int = 50,
) -> dict[str, object]:
    """Generate a walking trajectory to the target via Kimodo.

    Calculates a stopping point *stop_short_m* short of the target and
    sends a trajectory generation request to the Kimodo API with a
    ``root_target_xy`` constraint.

    Args:
        target_x: Target X in odom frame (meters).
        target_y: Target Y in odom frame (meters).
        current_x: Current robot X in odom frame.
        current_y: Current robot Y in odom frame.
        stop_short_m: Stop this far from the target.
        diffusion_steps: Kimodo diffusion quality parameter.

    Returns:
        Dict with ``qpos`` trajectory and metadata from Kimodo.
    """
    dx = target_x - current_x
    dy = target_y - current_y
    distance = math.sqrt(dx * dx + dy * dy)

    walk_distance = max(0.0, distance - stop_short_m)
    if walk_distance < 0.01:
        return {"status": "already_close", "distance": distance}

    # Compute stop point along the line to target
    ratio = walk_distance / distance if distance > 0 else 0.0
    stop_x = current_x + dx * ratio
    stop_y = current_y + dy * ratio

    duration = max(2.0, min(10.0, walk_distance / _AVG_WALK_SPEED))

    payload = {
        "prompt": "A person walks forward steadily",
        "duration": duration,
        "diffusion_steps": diffusion_steps,
        "root_target_xy": [stop_x, stop_y],
    }

    response = httpx.post(
        f"{KIMODO_URL}/generate",
        json=payload,
        timeout=120.0,
    )
    response.raise_for_status()
    result = response.json()

    return {
        "status": "trajectory_generated",
        "stop_point": [stop_x, stop_y],
        "distance": walk_distance,
        "duration": duration,
        "qpos": result.get("qpos"),
    }
