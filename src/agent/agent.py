from __future__ import annotations

from textwrap import dedent

from strands import Agent
from strands.models.openai import OpenAIModel

from agent.tools.generate_motion import generate_motion

SYSTEM_PROMPT = dedent("""
    You are a motion planner for the Unitree G1 humanoid robot.
    You help users generate robot motions by calling the generate_motion tool.

    When the user describes a motion, pose, or action, call generate_motion with
    a clear prompt and appropriate duration. Report the resulting file paths back
    to the user.
""").strip()


def create_agent() -> Agent:
    model = OpenAIModel(model_id="gpt-4o")
    return Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[generate_motion],
    )
