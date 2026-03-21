from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SpeechInputConfig:
    sample_rate: int
    channels: int
    microphone_device: str | int | None
    transcribe_model: str
    transcribe_language: str = "en"


def _parse_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer.") from exc


def _parse_device(raw: str | None) -> str | int | None:
    if raw is None or not raw.strip():
        return None
    raw = raw.strip()
    return int(raw) if raw.isdigit() else raw


def load_speech_input_config() -> SpeechInputConfig:
    return SpeechInputConfig(
        sample_rate=_parse_int_env("VOICE_INPUT_SAMPLE_RATE", 48000),
        channels=_parse_int_env("VOICE_INPUT_CHANNELS", 5),
        microphone_device=_parse_device(os.environ.get("VOICE_INPUT_DEVICE")),
        transcribe_model=os.environ.get(
            "VOICE_INPUT_TRANSCRIBE_MODEL",
            "gpt-4o-mini-transcribe",
        ),
        transcribe_language="en",
    )
