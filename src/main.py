from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from enum import StrEnum

import httpx
import tyro
from dotenv import load_dotenv

from agent.agent import create_agent


class InputMode(StrEnum):
    text = "text"
    mic = "mic"


@dataclass
class Config:
    """G1 motion agent configuration."""

    mode: InputMode = InputMode.text
    """Input mode: text (keyboard) or mic (microphone)."""

    tts: bool = True
    """Enable text-to-speech on the robot."""

    zmq: bool = True
    """Enable ZMQ motion publishing."""

    kimodo_url: str = ""
    """Kimodo server URL. Overrides KIMODO_URL env var if set."""


def preflight(cfg: Config) -> list[str]:
    """Run startup checks. Returns list of errors."""
    errors: list[str] = []

    if not os.environ.get("OPENAI_API_KEY"):
        errors.append("OPENAI_API_KEY is not set")

    kimodo_url = cfg.kimodo_url or os.environ.get("KIMODO_URL", "http://localhost:8420")
    try:
        r = httpx.get(
            f"{kimodo_url}/health",
            headers={"ngrok-skip-browser-warning": "true"},
            timeout=5.0,
        )
        r.raise_for_status()
        print(f"  kimodo: ok ({kimodo_url})")
    except httpx.HTTPError:
        errors.append(f"Kimodo unreachable at {kimodo_url}")

    if cfg.tts:
        try:
            import unitree_sdk2py  # noqa: F401  # pyright: ignore[reportMissingImports]
        except ImportError:
            errors.append(
                "TTS enabled but unitree_sdk2py not installed (use --no-tts to skip)"
            )
        iface = os.environ.get("UNITREE_NETWORK_INTERFACE")
        if not iface:
            errors.append(
                "TTS enabled but UNITREE_NETWORK_INTERFACE not set "
                "(use --no-tts to skip)"
            )
        else:
            import socket

            available = [name for _, name in socket.if_nameindex()]
            if iface not in available:
                errors.append(
                    f"Interface '{iface}' not found. "
                    f"Available: {', '.join(available)} "
                    "(use --no-tts to skip)"
                )
            else:
                print(f"  tts: ok (interface={iface})")

    if cfg.zmq:
        print(f"  zmq: ok ({os.environ.get('ZMQ_PUB_ADDRESS', 'tcp://*:5555')})")

    if cfg.mode == InputMode.mic:
        try:
            import sounddevice as sd  # noqa: F401  # pyright: ignore[reportMissingImports]

            print("  mic: ok")
        except ImportError:
            errors.append(
                "Mic mode requires sounddevice + PortAudio (use --mode text to skip)"
            )

    return errors


def run_text_loop(agent) -> None:  # pyright: ignore[reportMissingParameterType, reportUnknownParameterType]
    print("Type 'exit' to quit.\n")
    while True:
        try:
            user_input = input("> ")
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.strip().lower() in ("exit", "quit"):
            break
        agent(user_input)
        print()


def run_mic_loop(agent) -> None:  # pyright: ignore[reportMissingParameterType, reportUnknownParameterType]
    from g1.speech_input import SpeechInputSession

    session = SpeechInputSession()
    print("Press Enter to record, Enter again to stop. Type 'exit' to quit.\n")
    while True:
        try:
            transcript = session.listen_once()
        except (EOFError, KeyboardInterrupt):
            break
        if transcript is None:
            break
        if not transcript.strip():
            print("No speech detected. Try again.")
            continue
        agent(transcript)
        print()


def main() -> None:
    load_dotenv()
    cfg = tyro.cli(Config)

    if cfg.kimodo_url:
        os.environ["KIMODO_URL"] = cfg.kimodo_url

    print("preflight checks:")
    errors = preflight(cfg)
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        sys.exit(1)

    if cfg.zmq:
        from g1.publisher import init_publisher

        init_publisher()

    agent = create_agent(tts=cfg.tts)
    print(f"\nG1 agent ready. mode={cfg.mode} tts={cfg.tts} zmq={cfg.zmq}")

    try:
        if cfg.mode == InputMode.mic:
            run_mic_loop(agent)
        else:
            run_text_loop(agent)
    finally:
        if cfg.zmq:
            from g1.publisher import close_publisher

            close_publisher()


if __name__ == "__main__":
    main()
