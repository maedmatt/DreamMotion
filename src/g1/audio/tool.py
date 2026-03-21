from __future__ import annotations

import logging

from strands import tool

from g1.audio.client import get_unitree_audio_service

log = logging.getLogger(__name__)


@tool
def say_text(text: str, speaker_id: int = 1) -> dict[str, object]:
    """Make the Unitree G1 speak text through its onboard speaker.

    Use this whenever the robot should say something out loud.

    Args:
        text: The exact line the robot should speak.
        speaker_id: Unitree TTS speaker profile (0=default, 1=English).

    Returns:
        Dictionary with the spoken text and status information.
    """
    try:
        service = get_unitree_audio_service()
        return service.say_text(text=text, speaker_id=speaker_id)
    except Exception:
        log.warning("TTS failed — robot may be disconnected")
        return {
            "text": text,
            "speaker_id": speaker_id,
            "status": "failed",
            "error": "Could not reach robot speaker",
        }
