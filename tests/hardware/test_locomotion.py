"""Hardware test: SDK locomotion controller and odometry.

Prerequisites:
  - Robot powered on and in sport mode
  - UNITREE_NETWORK_INTERFACE set (e.g. eth0)
  - At least 3m of clear floor space in front of the robot
  - Someone watching the robot — it will move!

Run with:

    UNITREE_NETWORK_INTERFACE=eth0 uv run python tests/hardware/test_locomotion.py

Tests run in order. Each prompts before moving so you can confirm the robot
has space. Press Ctrl+C at any time to abort.

What this verifies:
  1. Odometry subscriber connects and receives data
  2. Velocity magnitude reads 0 when stationary
  3. Robot walks 2m forward and stops ~0.5m short (i.e. ~1.5m from start)
  4. step_backward moves the robot ~0.2m back
  5. stop() halts all motion immediately
"""

from __future__ import annotations

import sys
import time

import numpy as np


def _pass(name: str) -> None:
    print(f"  ✓ {name}")


def _fail(name: str, detail: str) -> None:
    print(f"  ✗ {name}: {detail}")
    sys.exit(1)


def _warn(name: str, detail: str) -> None:
    print(f"  ⚠ {name}: {detail}")


def test_odometry_connects() -> None:
    from g1.transforms.odometry import get_odometry

    print("  Connecting to odometry subscriber (waiting 2s for first message)...")
    odom = get_odometry()
    time.sleep(2.0)

    try:
        state = odom.get_state()
    except RuntimeError as e:
        _fail("odometry_connects", str(e))

    print(f"    Position: ({state.x:.3f}, {state.y:.3f}, {state.z:.3f})")
    print(f"    Quaternion: ({state.quat_x:.3f}, {state.quat_y:.3f}, "
          f"{state.quat_z:.3f}, {state.quat_w:.3f})")
    _pass("odometry subscriber receiving data")
    return odom


def test_velocity_zero_when_still(odom: object) -> None:
    from g1.transforms.odometry import OdometrySubscriber

    assert isinstance(odom, OdometrySubscriber)
    vel = odom.velocity_magnitude()
    print(f"    Velocity magnitude: {vel:.4f} m/s")
    if vel > 0.05:
        _warn("velocity_still", f"v={vel:.4f} m/s — robot may still be settling")
    else:
        _pass(f"velocity ≈ 0 when stationary ({vel:.4f} m/s)")


def test_walk_to_point(odom: object) -> None:
    """Walk 2m forward, stop 0.5m short → should end ~1.5m from start."""
    from g1.locomotion.sdk_controller import get_sdk_controller
    from g1.transforms.odometry import OdometrySubscriber

    assert isinstance(odom, OdometrySubscriber)
    sdk = get_sdk_controller()

    start = odom.get_state()
    target_x = start.x + 2.0
    target_y = start.y

    print(f"\n  Starting at ({start.x:.3f}, {start.y:.3f})")
    print(f"  Target:    ({target_x:.3f}, {target_y:.3f})")
    print("  Robot will walk ~1.5m forward (stopping 0.5m short of target)")
    print("  >>> Ensure 2m of clear space ahead — press Enter to start <<<")
    input()

    print("  Walking...")
    success = sdk.walk_to_point(
        target_x=target_x,
        target_y=target_y,
        current_odom_func=odom.get_state,
        stop_short_m=0.5,
    )

    end = odom.get_state()
    actual_dist = float(np.sqrt((end.x - start.x)**2 + (end.y - start.y)**2))
    print(f"  Ended at  ({end.x:.3f}, {end.y:.3f})")
    print(f"  Distance travelled: {actual_dist:.3f}m  (expected ~1.5m)")

    if not success:
        _fail("walk_to_point", "walk returned False (timed out)")

    dist_err = abs(actual_dist - 1.5)
    if dist_err > 0.3:
        _warn("walk_to_point", f"distance error {dist_err:.3f}m — check Kp gains or odometry")
    else:
        _pass(f"walked {actual_dist:.3f}m (error {dist_err:.3f}m)")

    return sdk, odom


def test_step_backward(sdk: object, odom: object) -> None:
    from g1.locomotion.sdk_controller import SdkLocomotionController
    from g1.transforms.odometry import OdometrySubscriber

    assert isinstance(sdk, SdkLocomotionController)
    assert isinstance(odom, OdometrySubscriber)

    before = odom.get_state()
    print("\n  Stepping backward 0.2m...")
    print("  >>> Press Enter to step back <<<")
    input()

    sdk.step_backward(0.2)
    time.sleep(0.5)  # settle
    after = odom.get_state()

    dist = float(np.sqrt((after.x - before.x)**2 + (after.y - before.y)**2))
    print(f"  Moved {dist:.3f}m backward (expected ~0.2m)")

    if abs(dist - 0.2) > 0.1:
        _warn("step_backward", f"distance {dist:.3f}m vs expected 0.2m")
    else:
        _pass(f"step_backward: moved {dist:.3f}m")


def test_stop() -> None:
    from g1.locomotion.sdk_controller import get_sdk_controller
    from g1.transforms.odometry import get_odometry

    sdk = get_sdk_controller()
    odom = get_odometry()

    print("\n  Testing stop() — will start moving then stop immediately")
    print("  >>> Press Enter <<<")
    input()

    sdk._client.Move(0.2, 0.0, 0.0)
    time.sleep(0.5)
    sdk.stop()
    time.sleep(0.3)

    vel = odom.velocity_magnitude()
    print(f"  Velocity after stop(): {vel:.4f} m/s")
    if vel > 0.05:
        _warn("stop", f"v={vel:.4f} m/s — robot still moving slightly")
    else:
        _pass("stop() halts motion")


if __name__ == "__main__":
    print("=== hardware/test_locomotion.py ===")
    print("Prerequisites: robot in sport mode, UNITREE_NETWORK_INTERFACE set")
    print("WARNING: Robot will move. Keep a safe distance.\n")

    odom = test_odometry_connects()
    test_velocity_zero_when_still(odom)
    sdk, odom = test_walk_to_point(odom)
    test_step_backward(sdk, odom)
    test_stop()

    print("\nDone.")
