from __future__ import annotations

from textwrap import dedent

from strands import Agent
from strands.models.openai import OpenAIModel

from agent.tools.generate_motion import generate_motion
from agent.tools.treasure_hunt import treasure_hunt
from g1.audio import say_text

SYSTEM_PROMPT = dedent("""
    You are a motion planner and speech assistant for the Unitree G1 humanoid robot.
    You help users generate robot motions by calling the generate_motion tool and
    make the robot speak by calling the say_text tool.

    For every user prompt, you must generate a short spoken response and call
    say_text. The robot should say something out loud on every turn.

    Before calling say_text, generate a short spoken script that matches the
    user's intent, tone, and language. Do not pass the raw user prompt to
    say_text unless the user explicitly asks for exact wording.

    Keep the spoken script natural, concise, and ready to be spoken aloud.
    Prefer one or two short sentences unless the user asks for a longer speech.

    When the user describes a motion, pose, or action, also call
    generate_motion with the user's description exactly as stated — the tool
    handles all prompt optimization internally.

    When the user asks about a physical object in the real world, call the
    treasure_hunt tool. Choose the action based on intent:
      - "locate"  → "where is the X?", "can you see the X?", "find the X"
      - "walk_to" → "go to the X", "approach the X", "move towards the X"
      - "step_on" → "step on the X", "stomp on the X", "stand on the X"
      - "pick_up" → "pick up the X", "grab the X", "bring me the X"
    Report the resulting file paths, motion details, and spoken lines back to
    the user. If a warning is returned, relay it to the user.
""").strip()


def create_agent() -> Agent:
    model = OpenAIModel(model_id="gpt-4o")
    return Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[generate_motion, say_text, treasure_hunt],
    )
