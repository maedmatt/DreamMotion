from __future__ import annotations

from dotenv import load_dotenv

from agent.agent import create_agent


def main() -> None:
    load_dotenv()
    agent = create_agent()

    print("G1 motion agent ready. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.strip().lower() in ("exit", "quit"):
            break
        agent(user_input)
        print()


if __name__ == "__main__":
    main()
