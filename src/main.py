from __future__ import annotations

import argparse

from dotenv import load_dotenv

from agent.agent import create_agent
from g1.speech_input import SpeechInputSession
from publisher import close_publisher, init_publisher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G1 motion agent")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--text", action="store_true", help="Use typed input mode.")
    group.add_argument(
        "--mic", action="store_true", help="Use laptop microphone input mode."
    )
    return parser.parse_args()


def run_text_loop(agent) -> None:  # pyright: ignore[reportMissingParameterType, reportUnknownParameterType]
    print("G1 motion agent ready. Text mode. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.strip().lower() in ("exit", "quit"):
            break
        agent(user_input)
        print()


def run_mic_loop(agent) -> None:  # pyright: ignore[reportMissingParameterType, reportUnknownParameterType]
    session = SpeechInputSession()
    print("G1 motion agent ready. Microphone mode.")
    print("Press Enter to start recording, then press Enter again to stop.")
    print("Type 'exit' instead of pressing Enter to quit.")
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
    args = parse_args()
    load_dotenv()
    init_publisher()
    agent = create_agent()

    try:
        if args.mic:
            run_mic_loop(agent)
        else:
            run_text_loop(agent)
    finally:
        close_publisher()


if __name__ == "__main__":
    main()
