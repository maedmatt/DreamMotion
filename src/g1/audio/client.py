from __future__ import annotations

import audioop
import importlib
import io
import os
import time
import wave
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class UnitreeAudioConfig:
    network_interface: str
    timeout_seconds: float


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


def _load_config() -> UnitreeAudioConfig:
    # Keep robot TTS usable from any entrypoint, not only scripts that already
    # loaded the repository's .env file.
    load_dotenv()

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
        channel_module, audio_module = _load_unitree_sdk()

        channel_module.ChannelFactoryInitialize(0, config.network_interface)  # pyright: ignore[reportAttributeAccessIssue]

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
