"""End-to-end pipeline test: full Look-Move-Look-Act run.

Prerequisites:
  - OAK-D mounted on robot with fixed downward angle
  - Robot powered on in sport mode
  - Kimodo SSH tunnel active (ssh -L 8420:localhost:8420 <aws-host>)
  - UNITREE_NETWORK_INTERFACE set
  - OPENAI_API_KEY set (for the agent narration, if enabled)
  - A box placed ~2–3m in front of the robot, clearly visible to the camera
  - At least 3m clear floor space

Run with:

    UNITREE_NETWORK_INTERFACE=eth0 OPENAI_API_KEY=sk-... \\
        uv run python tests/hardware/test_e2e.py

To test individual states in isolation use the --state flag:

    uv run python tests/hardware/test_e2e.py --state look
    uv run python tests/hardware/test_e2e.py --state move
    uv run python tests/hardware/test_e2e.py --state full

The 'full' run goes through all 4 states. The RL trajectory returned in ACT
is printed but NOT automatically executed — you pass it to your RL policy.
"""

from __future__ import annotations

import argparse
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


# ---------------------------------------------------------------------------
# Individual state tests
# ---------------------------------------------------------------------------


def run_look_only() -> None:
    """Run only the LOOK state: detect box and print world coordinates."""
    from g1.stance.controller import stabilize_for_vision
    from g1.transforms.odometry import get_odometry
    from g1.transforms.service import get_transform_service
    from g1.vision.camera import get_camera
    from g1.vision.detector import get_detector

    target = input("  Object to find [box]: ").strip() or "box"

    cam = get_camera()
    detector = get_detector()
    tf = get_transform_service()
    odom = get_odometry()

    print(f"\n  Waiting for odometry... ", end="", flush=True)
    time.sleep(2.0)
    odom.get_state()  # raises if no data
    print("OK")

    print("  Stabilizing...")
    stabilize_for_vision(velocity_func=odom.velocity_magnitude)

    print(f"  Detecting '{target}'...")
    frame = cam.capture()
    detections = detector.detect(frame, [target])

    if not detections:
        _fail("look", f"no '{target}' detected — is it in view?")

    best = detections[0]
    print(f"\n  Detection:")
    print(f"    label={best.label}  conf={best.confidence:.2f}  depth_valid={best.depth_valid}")
    print(f"    camera XYZ: {best.point_camera}")

    if not best.depth_valid or best.point_camera is None:
        _fail("look", "depth invalid — move object farther from camera (> 0.35m)")

    base_xyz = tf.transform_point_between(best.point_camera, "camera", "base")
    world_xyz = tf.transform_point_between(best.point_camera, "camera", "world")

    print(f"    base_link:  ({base_xyz[0]:.3f}, {base_xyz[1]:.3f}, {base_xyz[2]:.3f})")
    print(f"    world/odom: ({world_xyz[0]:.3f}, {world_xyz[1]:.3f}, {world_xyz[2]:.3f})")
    _pass("LOOK state completed")


def run_move_only() -> None:
    """Command the robot to walk to a manually entered world coordinate."""
    from g1.locomotion.sdk_controller import get_sdk_controller
    from g1.stance.controller import stabilize_for_vision
    from g1.transforms.odometry import get_odometry

    odom = get_odometry()
    time.sleep(2.0)
    state = odom.get_state()

    print(f"  Current position: ({state.x:.3f}, {state.y:.3f})")
    tx = float(input("  Target X (odom frame): "))
    ty = float(input("  Target Y (odom frame): "))
    stop_short = float(input("  Stop short (m) [0.5]: ").strip() or "0.5")

    print(f"\n  Walking to ({tx:.3f}, {ty:.3f}), stopping {stop_short}m short...")
    print("  >>> Press Enter to start (robot will move!) <<<")
    input()

    sdk = get_sdk_controller()
    success = sdk.walk_to_point(tx, ty, odom.get_state, stop_short_m=stop_short)

    stabilize_for_vision(velocity_func=odom.velocity_magnitude)
    end = odom.get_state()
    dist = float(np.sqrt((end.x - state.x)**2 + (end.y - state.y)**2))

    print(f"  Ended at ({end.x:.3f}, {end.y:.3f}), walked {dist:.3f}m")
    if success:
        _pass("MOVE state completed")
    else:
        _fail("move", "walk timed out")


def run_full_pipeline(action: str = "step_on") -> None:
    """Run the pipeline with the given action.

    action:
      locate   — LOOK only, report position, no movement
      walk_to  — LOOK + MOVE, stop near object
      step_on  — LOOK + MOVE + LOOK_AGAIN + ACT (foot, needs Kimodo)
      pick_up  — LOOK + MOVE + LOOK_AGAIN + ACT (hand, needs Kimodo)
    """
    from g1.locomotion.sdk_controller import get_sdk_controller
    from g1.state_machine.machine import TreasureHuntStateMachine
    from g1.transforms.service import get_transform_service
    from g1.vision.camera import get_camera
    from g1.vision.detector import get_detector

    target = input("  Object to find [box]: ").strip() or "box"
    walk_method = input("  Walk method [SDK/kimodo]: ").strip().upper() or "SDK"

    needs_kimodo = action in ("step_on", "pick_up")
    if needs_kimodo:
        print(
            f"\n  Action '{action}' will call Kimodo for the ACT step.\n"
            "  If Kimodo is unavailable the pipeline will error at ACT.\n"
            "  Use --action walk_to to test movement only without Kimodo."
        )

    print("\n  Initializing subsystems...")
    time.sleep(2.0)

    camera = get_camera()
    detector = get_detector()
    transforms = get_transform_service()
    sdk = get_sdk_controller()

    spoken: list[str] = []

    def say(text: str) -> None:
        spoken.append(text)
        print(f"  [ROBOT SAYS] {text}")

    machine = TreasureHuntStateMachine(
        target_object=target,
        camera=camera,
        detector=detector,
        transforms=transforms,
        odometry=None,
        sdk_controller=sdk,
        say=say,
        walk_method=walk_method,  # type: ignore[arg-type]
        action=action,  # type: ignore[arg-type]
    )

    print(f"\n  action='{action}'  target='{target}'  walk_method={walk_method}")
    print("  >>> Ensure object is visible and 2–3m ahead — press Enter to start <<<")
    input()

    result = machine.run()

    print("\n  === Pipeline result ===")
    print(f"  Final state:      {result['final_state']}")
    print(f"  Target base XYZ:  {result.get('target_local_xyz')}")

    if "look_detection" in result:
        d = result["look_detection"]
        print(f"  LOOK:             conf={d['confidence']:.2f}  base={d['base_xyz']}")
    if "look_again_detection" in result:
        d = result["look_again_detection"]
        print(f"  LOOK_AGAIN:       conf={d['confidence']:.2f}  local={d['local_xyz']}")
    if "act_kimodo_result" in result:
        qpos = result["act_kimodo_result"].get("qpos", [])
        print(f"  Kimodo traj:      {len(qpos)} frames")
        print("  >>> Pass qpos to your RL policy for execution <<<")

    if result["final_state"] == "DONE":
        _pass(f"Pipeline '{action}' completed successfully")
    else:
        _fail("full_pipeline", f"ended in state {result['final_state']}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end pipeline test")
    parser.add_argument(
        "--state",
        choices=["look", "move", "full"],
        default="full",
        help="Low-level state to run in isolation (default: full)",
    )
    parser.add_argument(
        "--action",
        choices=["locate", "walk_to", "step_on", "pick_up"],
        default=None,
        help=(
            "Action to run via the state machine. Overrides --state when set. "
            "locate/walk_to need no Kimodo; step_on/pick_up require Kimodo tunnel."
        ),
    )
    args = parser.parse_args()

    print("=== hardware/test_e2e.py ===")
    print("Prerequisites: robot on, OAK-D mounted\n")

    if args.action is not None:
        run_full_pipeline(action=args.action)
    elif args.state == "look":
        run_look_only()
    elif args.state == "move":
        run_move_only()
    else:
        run_full_pipeline()
