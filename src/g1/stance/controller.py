from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


def stabilize_for_vision(
    velocity_func: Callable[[], float] | None = None,
    timeout_s: float = 5.0,
    threshold: float = 0.01,
) -> None:
    """Wait until the robot is stable enough for a vision capture.

    With the OAK-D mounted at a fixed downward angle, no posture change
    is needed. This function simply waits for the robot to stop moving.

    Args:
        velocity_func: Optional callable returning current velocity magnitude.
            If provided, polls until velocity < threshold. If None, falls
            back to a fixed 0.5s sleep.
        timeout_s: Maximum time to wait for stabilization (seconds).
        threshold: Velocity magnitude threshold to consider "stopped".
    """
    if velocity_func is None:
        time.sleep(0.5)
        return

    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        vel = velocity_func()
        if vel < threshold:
            return
        time.sleep(0.05)

    # Timed out — proceed anyway (best-effort).
