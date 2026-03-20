from __future__ import annotations

from strands import tool

from g1.audio.client import get_unitree_audio_service


@tool
def say_text(text: str, speaker_id: int = 0) -> dict[str, object]:
    """Make the Unitree G1 speak text through its onboard speaker.

    Use this whenever the robot should say something out loud.

    Args:
        text: The exact line the robot should speak.
        speaker_id: Unitree TTS speaker profile. The official examples use `0`
            for the default voice and `1` for English TTS.

    Returns:
        Dictionary with the spoken text and Unitree SDK status information.
    """
    service = get_unitree_audio_service()
    return service.say_text(text=text, speaker_id=speaker_id)
