from __future__ import annotations

import audioop
import importlib
import io
import logging
import os
import socket
import time
import wave
from dataclasses import dataclass
from functools import lru_cache

from dotenv import dotenv_values, load_dotenv

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class UnitreeAudioConfig:
    network_interface: str | None
    timeout_seconds: float
    interface_source: str
    available_interfaces: tuple[str, ...]
    resolution_notes: tuple[str, ...] = ()


def _load_unitree_sdk() -> tuple:  # pyright: ignore[reportMissingTypeArgument]
    try:
        channel_module = importlib.import_module("unitree_sdk2py.core.channel")
        audio_module = importlib.import_module(
            "unitree_sdk2py.g1.audio.g1_audio_client"
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Unitree audio support requires the official Unitree SDK2 Python "
            "package, which exposes the `unitree_sdk2py` import path."
        ) from exc

    return channel_module, audio_module


def _normalize_interface_name(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _list_available_interfaces() -> tuple[str, ...]:
    try:
        return tuple(name for _, name in socket.if_nameindex())
    except OSError:
        return ()


def _resolve_network_interface() -> tuple[
    str | None,
    str,
    tuple[str, ...],
    tuple[str, ...],
]:
    available_interfaces = _list_available_interfaces()
    configured_env = _normalize_interface_name(os.environ.get("UNITREE_NETWORK_INTERFACE"))
    configured_dotenv = _normalize_interface_name(
        dotenv_values().get("UNITREE_NETWORK_INTERFACE")
    )
    notes: list[str] = []

    def is_available(interface_name: str | None) -> bool:
        return bool(interface_name) and (
            not available_interfaces or interface_name in available_interfaces
        )

    if configured_env and is_available(configured_env):
        return configured_env, "environment", available_interfaces, tuple(notes)

    if configured_env and available_interfaces:
        notes.append(
            "Ignoring stale UNITREE_NETWORK_INTERFACE=%r from the process "
            "environment because it is not available on this host. Available "
            "interfaces: %s."
            % (configured_env, ", ".join(available_interfaces))
        )

    if configured_dotenv and is_available(configured_dotenv):
        return configured_dotenv, ".env", available_interfaces, tuple(notes)

    if configured_dotenv and available_interfaces:
        notes.append(
            "Configured UNITREE_NETWORK_INTERFACE=%r from .env is not available "
            "on this host. Available interfaces: %s."
            % (configured_dotenv, ", ".join(available_interfaces))
        )

    return None, "autodetect", available_interfaces, tuple(notes)


def _load_config() -> UnitreeAudioConfig:
    # Keep robot TTS usable from any entrypoint, not only scripts that already
    # loaded the repository's .env file.
    load_dotenv()

    network_interface, interface_source, available_interfaces, resolution_notes = (
        _resolve_network_interface()
    )

    timeout_raw = os.environ.get("UNITREE_AUDIO_TIMEOUT", "10.0")
    try:
        timeout_seconds = float(timeout_raw)
    except ValueError as exc:
        raise RuntimeError("UNITREE_AUDIO_TIMEOUT must be a valid float.") from exc

    return UnitreeAudioConfig(
        network_interface=network_interface,
        timeout_seconds=timeout_seconds,
        interface_source=interface_source,
        available_interfaces=available_interfaces,
        resolution_notes=resolution_notes,
    )


def _init_channel_factory(
    channel_module: object,
    config: UnitreeAudioConfig,
) -> None:
    for note in config.resolution_notes:
        log.warning("%s", note)

    available = ", ".join(config.available_interfaces) or "unknown"

    if config.network_interface:
        try:
            channel_module.ChannelFactoryInitialize(0, config.network_interface)  # pyright: ignore[reportAttributeAccessIssue]
            return
        except Exception:
            log.warning(
                "Unitree DDS init failed on interface %r from %s; retrying with "
                "autodetect. Available interfaces: %s.",
                config.network_interface,
                config.interface_source,
                available,
                exc_info=True,
            )
            try:
                channel_module.ChannelFactoryInitialize(0, None)  # pyright: ignore[reportAttributeAccessIssue]
                log.info(
                    "Unitree DDS initialized with autodetected interface after "
                    "explicit init failed."
                )
                return
            except Exception as fallback_exc:
                raise RuntimeError(
                    "Failed to initialize Unitree DDS using interface "
                    f"{config.network_interface!r} from {config.interface_source}, "
                    "and autodetect also failed. Available interfaces: "
                    f"{available}."
                ) from fallback_exc

    try:
        channel_module.ChannelFactoryInitialize(0, None)  # pyright: ignore[reportAttributeAccessIssue]
        log.info("Unitree DDS initialized with autodetected network interface.")
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize Unitree DDS with autodetect. Set "
            "UNITREE_NETWORK_INTERFACE to a valid robot-facing interface. "
            f"Available interfaces: {available}."
        ) from exc


class UnitreeAudioService:
    def __init__(self, config: UnitreeAudioConfig) -> None:
        channel_module, audio_module = _load_unitree_sdk()

        _init_channel_factory(channel_module, config)

        client = audio_module.AudioClient()  # pyright: ignore[reportAttributeAccessIssue]
        client.SetTimeout(config.timeout_seconds)
        client.Init()
        self._client = client

        volume = int(os.environ.get("UNITREE_AUDIO_VOLUME", "85"))
        self.set_volume(volume)

    def set_volume(self, volume: int) -> None:
        self._client.SetVolume(volume)

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

    def play_pcm_bytes(
        self,
        pcm_bytes: bytes,
        *,
        sample_rate_hz: int = 16000,
        num_channels: int = 1,
        sample_width_bytes: int = 2,
        stream_name: str = "codex-openai-tts",
    ) -> dict[str, object]:
        if sample_rate_hz != 16000:
            raise ValueError("Robot speaker streaming requires 16 kHz audio.")
        if num_channels != 1:
            raise ValueError("Robot speaker streaming requires mono audio.")
        if sample_width_bytes != 2:
            raise ValueError("Robot speaker streaming requires 16-bit PCM audio.")
        if not pcm_bytes:
            raise ValueError("PCM payload must not be empty.")

        bytes_per_second = sample_rate_hz * num_channels * sample_width_bytes
        chunk_size = bytes_per_second
        stream_id = str(int(time.time() * 1000))
        total_bytes = len(pcm_bytes)
        chunks_sent = 0

        for offset in range(0, total_bytes, chunk_size):
            chunk = pcm_bytes[offset : offset + chunk_size]
            ret_code, _ = self._client.PlayStream(stream_name, stream_id, chunk)
            if ret_code != 0:
                raise RuntimeError(
                    f"Robot PCM playback failed with status code {ret_code}."
                )
            chunks_sent += 1
            time.sleep(len(chunk) / bytes_per_second)

        stop_code = int(self._client.PlayStop(stream_name))
        if stop_code != 0:
            raise RuntimeError(
                f"Robot PCM playback stop failed with status code {stop_code}."
            )

        return {
            "status": "played",
            "stream_name": stream_name,
            "sample_rate_hz": sample_rate_hz,
            "num_channels": num_channels,
            "sample_width_bytes": sample_width_bytes,
            "bytes": total_bytes,
            "chunks_sent": chunks_sent,
            "stop_code": stop_code,
        }

    def play_wav_bytes(
        self,
        wav_bytes: bytes,
        *,
        stream_name: str = "codex-openai-tts",
    ) -> dict[str, object]:
        if not wav_bytes:
            raise ValueError("WAV payload must not be empty.")

        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            num_channels = wav_file.getnchannels()
            sample_width_bytes = wav_file.getsampwidth()
            sample_rate_hz = wav_file.getframerate()
            pcm_bytes = wav_file.readframes(wav_file.getnframes())

        if sample_width_bytes != 2:
            raise ValueError("Only 16-bit WAV audio is supported for robot playback.")

        if num_channels == 2:
            pcm_bytes = audioop.tomono(pcm_bytes, sample_width_bytes, 0.5, 0.5)
            num_channels = 1
        elif num_channels != 1:
            raise ValueError("Only mono or stereo WAV audio is supported.")

        if sample_rate_hz != 16000:
            pcm_bytes, _ = audioop.ratecv(
                pcm_bytes,
                sample_width_bytes,
                num_channels,
                sample_rate_hz,
                16000,
                None,
            )
            sample_rate_hz = 16000

        result = self.play_pcm_bytes(
            pcm_bytes,
            sample_rate_hz=sample_rate_hz,
            num_channels=num_channels,
            sample_width_bytes=sample_width_bytes,
            stream_name=stream_name,
        )
        return {
            **result,
            "source_format": "wav",
        }


@lru_cache(maxsize=1)
def get_unitree_audio_service() -> UnitreeAudioService:
    return UnitreeAudioService(_load_config())
