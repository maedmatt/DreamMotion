# pyright: reportAttributeAccessIssue=false
from __future__ import annotations

import math
import time
from array import array
from dataclasses import dataclass

import tyro

from g1.audio.client import _load_config, get_unitree_audio_service


@dataclass
class Config:
    """Robot speaker diagnostics for the Unitree G1."""

    volume: int = 85
    pause_seconds: float = 3.0
    tone_frequency_hz: float = 880.0
    tone_duration_seconds: float = 1.5
    sample_rate_hz: int = 16000
    skip_tts: bool = False
    skip_tone: bool = False


def _make_tone_pcm_bytes(
    *,
    sample_rate_hz: int,
    frequency_hz: float,
    duration_seconds: float,
    amplitude: float = 0.35,
) -> bytes:
    sample_count = max(1, int(sample_rate_hz * duration_seconds))
    samples = array(
        "h",
        (
            int(
                32767
                * amplitude
                * math.sin(2.0 * math.pi * frequency_hz * i / sample_rate_hz)
            )
            for i in range(sample_count)
        ),
    )
    return samples.tobytes()


def _queue_tts_test(
    *,
    service: object,
    text: str,
    speaker_id: int,
    pause_seconds: float,
) -> None:
    print(f"\nTTS test: speaker_id={speaker_id} text={text!r}")
    result = service.say_text(text=text, speaker_id=speaker_id)
    print(f"  result: {result}")
    time.sleep(pause_seconds)


def _play_tone(
    *,
    client: object,
    sample_rate_hz: int,
    frequency_hz: float,
    duration_seconds: float,
) -> None:
    print(
        f"\nRaw tone test: {frequency_hz:.0f} Hz for "
        f"{duration_seconds:.1f}s at {sample_rate_hz} Hz mono"
    )
    pcm_bytes = _make_tone_pcm_bytes(
        sample_rate_hz=sample_rate_hz,
        frequency_hz=frequency_hz,
        duration_seconds=duration_seconds,
    )
    stream_name = "codex-audio-test"
    stream_id = str(int(time.time() * 1000))
    ret_code, _ = client.PlayStream(stream_name, stream_id, pcm_bytes)
    print(f"  PlayStream returned: {ret_code}")
    time.sleep(duration_seconds + 0.25)
    stop_code = client.PlayStop(stream_name)
    print(f"  PlayStop returned: {stop_code}")


def main() -> None:
    cfg = tyro.cli(Config)
    resolved = _load_config()
    service = get_unitree_audio_service()
    client = service._client

    print("Robot audio diagnostics")
    print(f"  interface: {resolved.network_interface}")
    print(f"  timeout: {resolved.timeout_seconds}s")

    volume_code, volume_payload = client.GetVolume()
    print(f"  volume before: code={volume_code}, payload={volume_payload}")

    set_code = client.SetVolume(cfg.volume)
    print(f"  set volume -> {cfg.volume}: code={set_code}")

    volume_code, volume_payload = client.GetVolume()
    print(f"  volume after: code={volume_code}, payload={volume_payload}")

    if not cfg.skip_tts:
        _queue_tts_test(
            service=service,
            text="Hello everyone, I am a robot.",
            speaker_id=0,
            pause_seconds=cfg.pause_seconds,
        )
        _queue_tts_test(
            service=service,
            text="Hello from speaker one.",
            speaker_id=1,
            pause_seconds=cfg.pause_seconds,
        )

    if not cfg.skip_tone:
        _play_tone(
            client=client,
            sample_rate_hz=cfg.sample_rate_hz,
            frequency_hz=cfg.tone_frequency_hz,
            duration_seconds=cfg.tone_duration_seconds,
        )

    print("\nInterpretation:")
    print(
        "  - If you hear the raw tone but not the TTS lines,"
        " the speaker works and the issue is TTS related."
    )
    print(
        "  - If you hear Chinese TTS but not English,"
        " English synthesis is the likely issue."
    )
    print(
        "  - If you hear nothing at all, the issue is"
        " downstream: audio output, service state, or hardware."
    )


if __name__ == "__main__":
    main()
