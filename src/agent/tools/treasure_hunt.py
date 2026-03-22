from __future__ import annotations

from typing import TYPE_CHECKING

from strands import tool

from g1.state_machine.machine import Action, TreasureHuntStateMachine

if TYPE_CHECKING:
    from collections.abc import Callable


_VALID_ACTIONS: tuple[Action, ...] = (
    "locate",
    "walk_to",
    "step_on",
    "pick_up",
    "point_at",
)


def _make_say_callback() -> Callable[[str], None]:
    """Build a best-effort narration callback."""

    def _say(text: str) -> None:
        message = text.strip()
        if not message:
            return
        try:
            from g1.audio.tool import say_text_impl

            result = say_text_impl(message)
            if result.get("status") == "failed":
                print(f"[SAY] {message}")
        except Exception:
            print(f"[SAY] {message}")

    return _say


def _default_motion_generator(
    description: str,
    **kwargs: object,
) -> dict[str, object]:
    from agent.tools.generate_motion import generate_motion_impl

    return generate_motion_impl(description, **kwargs)


def treasure_hunt_impl(
    target_object: str,
    action: str = "walk_to",
    *,
    say: Callable[[str], None] | None = None,
    motion_generator: Callable[..., dict[str, object]] | None = None,
    camera: object | None = None,
    detector: object | None = None,
    transforms: object | None = None,
) -> dict[str, object]:
    """Shared treasure-hunt implementation for CLI and web flows."""
    target = target_object.strip()
    if not target:
        raise ValueError("target_object must not be empty")

    from g1.transforms.service import get_transform_service
    from g1.vision.camera import get_camera
    from g1.vision.detector import get_detector

    act: Action = action if action in _VALID_ACTIONS else "walk_to"

    machine = TreasureHuntStateMachine(
        target_object=target,
        camera=(camera or get_camera()),
        detector=(detector or get_detector()),
        transforms=(transforms or get_transform_service()),
        motion_generator=motion_generator or _default_motion_generator,
        say=say or _make_say_callback(),
        action=act,
    )
    return machine.run()


@tool
def treasure_hunt(
    target_object: str,
    action: str = "walk_to",
) -> dict[str, object]:
    """Find a real-world object and optionally generate a constrained motion.

    Use this tool whenever the user asks about a physical object in the room,
    for example locating it, walking to it, pointing at it, stepping on it, or
    picking it up.

    Actions:
      locate   - only detect and report the object position
      walk_to  - detect then generate a constrained walk motion to the object
      step_on  - detect then generate a constrained stomp motion
      pick_up  - detect then generate a constrained pickup motion
      point_at - detect then generate a pointing motion from the current pose
    """
    return treasure_hunt_impl(target_object=target_object, action=action)
