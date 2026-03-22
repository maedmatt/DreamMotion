from __future__ import annotations

from strands import tool

from g1.locomotion.sdk_controller import get_sdk_controller
from g1.state_machine.machine import Action, TreasureHuntStateMachine
from g1.transforms.service import get_transform_service
from g1.vision.camera import get_camera
from g1.vision.detector import get_detector


def _make_say_callback() -> object:
    """Build a narration callback using the audio service (best-effort)."""
    try:
        from g1.audio.client import get_unitree_audio_service

        svc = get_unitree_audio_service()
        return lambda text: svc.say_text(text=text, speaker_id=1)
    except Exception:
        return lambda text: print(f"[SAY] {text}")


@tool
def treasure_hunt(
    target_object: str,
    action: str = "step_on",
) -> dict[str, object]:
    """Find a real-world object with the camera and optionally interact with it.

    Use this tool whenever the user asks to find, locate, go to, pick up,
    or interact with a physical object.

    Actions:
      locate   — look for the object and report where it is. No movement.
                 Use for: "where is the X?", "can you see the X?",
                          "find the X", "is there a X nearby?"
      walk_to  — walk close to the object and stop. No interaction.
                 Use for: "go to the X", "approach the X",
                          "move towards the X"
      step_on  — walk to the object and step on it with the right foot.
                 Use for: "step on the X", "stomp on the X",
                          "stand on the X"
      pick_up  — walk to the object and pick it up with the right hand.
                 Use for: "pick up the X", "grab the X",
                          "get the X", "bring me the X"
      point_at — detect the object and point at it with the right arm.
                 No walking — points in place from wherever the robot stands.
                 Use for: "point at the X", "show me the X",
                          "indicate the X", "where is the X, point to it"

    Args:
        target_object: Natural language description of the object
                       (e.g. "red box", "yellow ball", "water bottle").
        action: One of "locate", "walk_to", "step_on", "pick_up", "point_at".

    Returns:
        Dictionary with final_state, coordinates found, and any
        trajectory metadata from Kimodo.
    """
    camera = get_camera()
    detector = get_detector()
    transforms = get_transform_service()
    say = _make_say_callback()

    act: Action = (
        action
        if action in ("locate", "walk_to", "step_on", "pick_up", "point_at")
        else "step_on"
    )  # type: ignore[assignment]

    sdk = get_sdk_controller() if act != "locate" else None

    arm = None
    if act == "point_at":
        from g1.arm.arm_controller import get_arm_controller

        arm = get_arm_controller()

    machine = TreasureHuntStateMachine(
        target_object=target_object,
        camera=camera,
        detector=detector,
        transforms=transforms,
        sdk_controller=sdk,
        say=say,  # type: ignore[arg-type]
        arm_controller=arm,
        action=act,
    )
    return machine.run()
