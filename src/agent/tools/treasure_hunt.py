from __future__ import annotations

from typing import Literal

from strands import tool

from g1.locomotion.sdk_controller import get_sdk_controller
from g1.state_machine.machine import TreasureHuntStateMachine
from g1.transforms.odometry import get_odometry
from g1.transforms.service import get_transform_service
from g1.vision.camera import get_camera
from g1.vision.detector import get_detector


def _make_say_callback() -> object:
    """Build a narration callback using the audio service (best-effort)."""
    try:
        from g1.audio.client import get_unitree_audio_service

        svc = get_unitree_audio_service()
        return lambda text: svc.say_text(text=text, speaker_id=0)
    except Exception:
        # Fall back to console-only narration if audio is unavailable.
        return lambda text: print(f"[SAY] {text}")


@tool
def treasure_hunt(
    target_object: str,
    walk_method: str = "SDK",
) -> dict[str, object]:
    """Find a target object using vision and walk to it to interact.

    Runs the Look-Move-Look-Act state machine:
    1. LOOK — detect the object using the OAK-D camera and YOLO-World.
    2. MOVE — walk toward the object, stopping 0.5m short.
    3. LOOK_AGAIN — precision re-detection at close range.
    4. ACT — generate a Kimodo trajectory with foot constraint to step on it.

    Args:
        target_object: Natural language description of the object
                       to find (e.g. "red box", "yellow ball").
        walk_method: Walking strategy — "SDK" for closed-loop P-controller
                     or "KIMODO" for Kimodo trajectory generation.

    Returns:
        Dictionary with final state, coordinates found, and any
        generated trajectory metadata.
    """
    camera = get_camera()
    detector = get_detector()
    transforms = get_transform_service()
    odometry = get_odometry()
    sdk = get_sdk_controller()
    say = _make_say_callback()

    method: Literal["SDK", "KIMODO"] = (
        "SDK" if walk_method.upper() == "SDK" else "KIMODO"
    )

    machine = TreasureHuntStateMachine(
        target_object=target_object,
        camera=camera,
        detector=detector,
        transforms=transforms,
        odometry=odometry,
        sdk_controller=sdk,
        say=say,  # type: ignore[arg-type]
        walk_method=method,
    )
    return machine.run()
