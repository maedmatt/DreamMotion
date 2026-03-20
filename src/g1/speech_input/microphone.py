from __future__ import annotations

import importlib
import tempfile
import wave
from pathlib import Path

from g1.speech_input.config import SpeechInputConfig


def _load_sounddevice() -> object:
    try:
        return importlib.import_module("sounddevice")
    except Exception as exc:
        raise RuntimeError(
            "Microphone mode requires the `sounddevice` package and a working "
            "PortAudio installation."
        ) from exc


class LaptopMicrophoneRecorder:
    def __init__(self, config: SpeechInputConfig) -> None:
        self._config = config
        self._sounddevice = _load_sounddevice()

    def record_until_enter(self) -> Path:
        frames: list[object] = []

        def callback(indata, _frames, _time_info, status) -> None:
            if status:
                print(f"Microphone status: {status}")
            frames.append(indata.copy())

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            audio_path = Path(temp_file.name)

        try:
            with self._sounddevice.InputStream(
                samplerate=self._config.sample_rate,
                channels=self._config.channels,
                dtype="int16",
                device=self._config.microphone_device,
                callback=callback,
            ):
                input("Recording... press Enter again to stop. ")

            if not frames:
                raise RuntimeError("No microphone audio was captured.")

            audio_bytes = b"".join(frame.tobytes() for frame in frames)
            with wave.open(str(audio_path), "wb") as wav_file:
                wav_file.setnchannels(self._config.channels)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self._config.sample_rate)
                wav_file.writeframes(audio_bytes)

            return audio_path
        except Exception:
            audio_path.unlink(missing_ok=True)
            raise
