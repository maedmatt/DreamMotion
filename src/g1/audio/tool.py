from __future__ import annotations

import logging

from strands import tool

from g1.audio.client import get_unitree_audio_service

log = logging.getLogger(__name__)


def say_text_impl(
    text: str,
    speaker_id: int = 1,
    voice: str = 'alloy',
) -> dict[str, object]:
    """Shared implementation for robot speech across CLI and web agent flows."""
    del voice  # Robot speech follows the direct Unitree TTS path; speaker_id=1 is the English profile on this robot.

    message = text.strip()
    if not message:
        return {
            'text': text,
            'speaker_id': speaker_id,
            'status': 'failed',
            'error': 'Text to speak must not be empty.',
        }

    try:
        service = get_unitree_audio_service()
        result = service.say_text(text=message, speaker_id=speaker_id)
        log.info(
            'Robot TTS queued (speaker_id=%s, status=%s)',
            speaker_id,
            result.get('status'),
        )
        return {
            'text': message,
            'speaker_id': speaker_id,
            'source': 'unitree_tts',
            **result,
        }
    except Exception:
        log.warning('Robot TTS failed for speaker_id=%s', speaker_id, exc_info=True)
        return {
            'text': message,
            'speaker_id': speaker_id,
            'status': 'failed',
            'error': 'Could not reach robot speaker',
        }


@tool
def say_text(text: str, speaker_id: int = 1, voice: str = 'alloy') -> dict[str, object]:
    """Make the Unitree G1 speak text through its speaker.

    This follows the same direct TTS path as the working Unitree example:
    initialize the audio client and send the agent's text to TtsMaker.
    """
    return say_text_impl(text=text, speaker_id=speaker_id, voice=voice)
