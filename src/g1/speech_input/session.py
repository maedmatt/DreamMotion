from __future__ import annotations

from g1.speech_input.config import SpeechInputConfig, load_speech_input_config
from g1.speech_input.microphone import LaptopMicrophoneRecorder
from g1.speech_input.transcriber import OpenAISpeechTranscriber


class SpeechInputSession:
    def __init__(self, config: SpeechInputConfig | None = None) -> None:
        resolved_config = config or load_speech_input_config()
        self._recorder = LaptopMicrophoneRecorder(resolved_config)
        self._transcriber = OpenAISpeechTranscriber(resolved_config)

    def listen_once(self) -> str | None:
        command = input("\nPress Enter to start recording or type 'exit' to quit: ")
        if command.strip().lower() in ("exit", "quit"):
            return None

        audio_path = self._recorder.record_until_enter()
        try:
            transcript = self._transcriber.transcribe(audio_path)
        finally:
            audio_path.unlink(missing_ok=True)

        if transcript:
            print(f"Transcript: {transcript}")
        else:
            print("Transcript: <empty>")

        return transcript
