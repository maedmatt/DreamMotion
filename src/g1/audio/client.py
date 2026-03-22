from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from functools import lru_cache

from g1.unitree_common import ensure_channel_initialized


@dataclass(frozen=True, slots=True)
class UnitreeAudioConfig:
    network_interface: str
    timeout_seconds: float


def _load_audio_module() -> object:
    try:
        audio_module = importlib.import_module(
            "unitree_sdk2py.g1.audio.g1_audio_client"
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Unitree audio support requires the official Unitree SDK2 Python "
            "package, which exposes the `unitree_sdk2py` import path."
        ) from exc

    return audio_module


def _load_config() -> UnitreeAudioConfig:
    network_interface = os.environ.get("UNITREE_NETWORK_INTERFACE")
    if not network_interface:
        raise RuntimeError(
            "Set UNITREE_NETWORK_INTERFACE to the robot-facing network interface "
            "before using the audio tool, for example `eth0`."
        )

    timeout_raw = os.environ.get("UNITREE_AUDIO_TIMEOUT", "10.0")
    try:
        timeout_seconds = float(timeout_raw)
    except ValueError as exc:
        raise RuntimeError("UNITREE_AUDIO_TIMEOUT must be a valid float.") from exc

    return UnitreeAudioConfig(
        network_interface=network_interface,
        timeout_seconds=timeout_seconds,
    )


class UnitreeAudioService:
    def __init__(self, config: UnitreeAudioConfig) -> None:
        ensure_channel_initialized(config.network_interface)
        audio_module = _load_audio_module()

        client = audio_module.AudioClient()  # type: ignore[attr-defined]
        client.Init()
        client.SetTimeout(config.timeout_seconds)
        self._client = client

    def say_text(self, text: str, speaker_id: int = 1) -> dict[str, object]:
        message = text.strip()
        if not message:
            raise ValueError("Text to speak must not be empty.")

        status_code = int(self._client.TtsMaker(message, speaker_id))
        if status_code != 0:
            raise RuntimeError(f"Unitree TTS failed with status code {status_code}.")

        return {
            "text": message,
            "speaker_id": speaker_id,
            "status": "queued",
            "status_code": status_code,
        }


@lru_cache(maxsize=1)
def get_unitree_audio_service() -> UnitreeAudioService:
    return UnitreeAudioService(_load_config())
