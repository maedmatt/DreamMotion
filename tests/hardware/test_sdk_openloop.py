"""Hardware test: open-loop SDK walk — no odometry, no camera required.

Tests that the G1 SportClient connects and responds to velocity commands.
Robot will move — keep a safe distance.

Prerequisites:
  - Robot powered on and standing
  - UNITREE_NETWORK_INTERFACE set (e.g. eth0)

Run with:

    UNITREE_NETWORK_INTERFACE=eth0 uv run python tests/hardware/test_sdk_openloop.py

Tests run in order, each prompts before moving.
"""

from __future__ import annotations

import sys
import time


def _pass(name: str) -> None:
    print(f"  ✓ {name}")


def _fail(name: str, detail: str) -> None:
    print(f"  ✗ {name}: {detail}")
    sys.exit(1)


def test_sdk_connects() -> None:
    from g1.locomotion.sdk_controller import get_sdk_controller

    print("  Connecting to SportClient...")
    sdk = get_sdk_controller()
    _pass("SportClient connected")
    return sdk


def test_stop() -> None:
    """Send a zero-velocity command — safe to run any time."""
    from g1.locomotion.sdk_controller import get_sdk_controller

    sdk = get_sdk_controller()
    sdk.stop()
    _pass("stop() sent zero velocity")


def test_walk_forward(sdk: object) -> None:
    """Walk forward 0.5m."""
    from g1.locomotion.sdk_controller import SdkLocomotionController

    assert isinstance(sdk, SdkLocomotionController)

    print("\n  Will walk ~0.5m forward.")
    print("  >>> Ensure 1m clear space ahead — press Enter to start <<<")
    input()

    sdk.walk_forward_distance(0.5, yaw_rad=0.0)
    _pass("walked forward ~0.5m (open-loop)")


def test_walk_with_turn(sdk: object) -> None:
    """Turn 45° left then walk 0.5m forward."""
    import math

    from g1.locomotion.sdk_controller import SdkLocomotionController

    assert isinstance(sdk, SdkLocomotionController)

    print("\n  Will turn 45° left then walk ~0.5m forward.")
    print("  >>> Ensure space in that direction — press Enter <<<")
    input()

    sdk.walk_forward_distance(0.5, yaw_rad=math.radians(45))
    _pass("turn 45° + walk 0.5m completed")


def test_step_backward(sdk: object) -> None:
    """Step backward 0.3m."""
    from g1.locomotion.sdk_controller import SdkLocomotionController

    assert isinstance(sdk, SdkLocomotionController)

    print("\n  Will step backward ~0.3m.")
    print("  >>> Press Enter <<<")
    input()

    sdk.step_backward(0.3)
    _pass("step_backward ~0.3m completed")


if __name__ == "__main__":
    print("=== hardware/test_sdk_openloop.py ===")
    print("Prerequisites: robot standing, UNITREE_NETWORK_INTERFACE set")
    print("WARNING: Robot will move. Keep a safe distance.\n")

    sdk = test_sdk_connects()
    test_stop()
    test_walk_forward(sdk)
    test_walk_with_turn(sdk)
    test_step_backward(sdk)

    print("\nDone.")
