from __future__ import annotations

from g1.agent import create_agent


def main() -> None:
    agent = create_agent()

    print("G1 motion agent ready. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.strip().lower() in ("exit", "quit"):
            break
        result = agent(user_input)
        for block in result.message.get("content", []):
            if "text" in block:
                print(f"\n{block['text']}")
                break


if __name__ == "__main__":
    main()
