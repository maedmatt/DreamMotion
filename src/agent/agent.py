from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

from strands import Agent
from strands.models.openai import OpenAIModel

from agent.tools.generate_motion import generate_motion
from agent.tools.treasure_hunt import treasure_hunt


_BASE_PROMPT = dedent("""
    You are a motion planner for the Unitree G1 humanoid robot.

    When the user describes a motion, pose, or action, call generate_motion
    with the user's description exactly as stated - the tool handles prompt
    optimization internally. The tool may return multiple motion clips if the
    description involves a sequence.

    Summarize results naturally: how many motions, what they are, and the most
    important warnings.

    If a tool fails, explain the error in plain language and suggest a practical
    alternative. Do not expose raw file paths or status codes unless the user
    explicitly asks for them.
""").strip()

_TTS_PROMPT = dedent("""
    You also have a say_text tool.

    For every user prompt, you must draft a short spoken response and call
    say_text exactly once. Keep the spoken line natural and concise.
""").strip()

_TREASURE_HUNT_PROMPT = dedent("""
    You also have a treasure_hunt tool for real-world object tasks.

    Use treasure_hunt when the user asks about a physical object in the room,
    for example locating it, going to it, pointing at it, stepping on it, or
    picking it up. Prefer treasure_hunt over generate_motion for these cases,
    because treasure_hunt already handles vision, the FSM, and constrained
    motion generation.

    Choose the treasure_hunt action based on intent:
      - locate   -> "where is the bottle?", "can you see the bottle?", "find the bottle"
      - walk_to  -> "go to the bottle", "approach the bottle", "move towards the bottle"
      - point_at -> "point at the bottle", "show me the bottle"
      - step_on  -> "step on the bottle", "stomp on the marker"
      - pick_up  -> "pick up the bottle", "grab the bottle"
""").strip()


def _tool_names(tools: Sequence[Any]) -> set[str]:
    names: set[str] = set()
    for tool_obj in tools:
        name = getattr(tool_obj, "__name__", None)
        if isinstance(name, str) and name:
            names.add(name)
    return names


def _build_system_prompt(tool_names: set[str]) -> str:
    sections = [_BASE_PROMPT]
    if "say_text" in tool_names:
        sections.append(_TTS_PROMPT)
    if "treasure_hunt" in tool_names:
        sections.append(_TREASURE_HUNT_PROMPT)
    return "\n\n".join(section for section in sections if section).strip()


def create_agent(
    *,
    tools: Sequence[Any] | None = None,
    model_id: str = "gpt-4.1",
) -> Agent:
    model = OpenAIModel(model_id=model_id)
    resolved_tools = list(tools) if tools is not None else [generate_motion, treasure_hunt]
    prompt = _build_system_prompt(_tool_names(resolved_tools))
    return Agent(model=model, system_prompt=prompt, tools=resolved_tools)
