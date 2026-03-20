from __future__ import annotations

import importlib
from pathlib import Path

from g1.speech_input.config import SpeechInputConfig


def _create_openai_client() -> object:
    try:
        openai_module = importlib.import_module("openai")
    except Exception as exc:
        raise RuntimeError(
            "Speech-to-text mode requires the `openai` package in the current environment."
        ) from exc

    return openai_module.OpenAI()


class OpenAISpeechTranscriber:
    def __init__(self, config: SpeechInputConfig) -> None:
        self._config = config
        self._client = _create_openai_client()

    def transcribe(self, audio_path: Path) -> str:
        with audio_path.open("rb") as audio_file:
            transcription = self._client.audio.transcriptions.create(
                model=self._config.transcribe_model,
                file=audio_file,
                response_format="text",
            )

        text = getattr(transcription, "text", transcription)
        return str(text).strip()
