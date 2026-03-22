from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import httpx
import tyro
from dotenv import load_dotenv

from agent.agent import create_agent


@dataclass
class Config:
    """G1 motion agent (text CLI)."""

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

    return errors


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

    agent = create_agent()
    print("\nG1 agent ready. Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("> ")
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.strip().lower() in ("exit", "quit"):
            break
        agent(user_input)
        print()


if __name__ == "__main__":
    main()
