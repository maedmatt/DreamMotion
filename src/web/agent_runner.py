from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from strands import tool

from agent.agent import create_agent
from agent.tools.generate_motion import generate_motion_impl
from g1.audio.tool import say_text_impl

SpeechTarget = Literal["web", "robot"]


@dataclass(slots=True)
class WebAgentRunResult:
    reply_text: str
    spoken_text: str | None
    speech: dict[str, object] | None
    motions: list[dict[str, object]]
    warning: str | None = None


@dataclass(slots=True)
class _ToolState:
    tts_target: SpeechTarget
    speaker_id: int
    spoken_text: str | None = None
    speech: dict[str, object] | None = None
    motions: list[dict[str, object]] = field(default_factory=list)
    warning: str | None = None


def _extract_text_blocks(message: Any) -> str:
    if not isinstance(message, dict):
        return ""

    content = message.get("content")
    if not isinstance(content, list):
        return ""

    texts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        text = block.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())

    return "\n".join(texts).strip()


def run_agent_for_web(
    prompt: str,
    *,
    tts_target: SpeechTarget,
    speaker_id: int,
    diffusion_steps: int,
) -> WebAgentRunResult:
    state = _ToolState(tts_target=tts_target, speaker_id=speaker_id)
    default_diffusion_steps = diffusion_steps
    default_speaker_id = speaker_id

    @tool
    def say_text(text: str, speaker_id: int = default_speaker_id) -> dict[str, object]:
        """Speak a short line out loud for the current web request.

        Use this on every turn after drafting a concise spoken script.
        """
        message = text.strip()
        if not message:
            response: dict[str, object] = {
                "status": "failed",
                "target": state.tts_target,
                "handled": False,
                "error": "Text to speak must not be empty.",
            }
            state.speech = response
            return response

        state.spoken_text = message

        if state.tts_target == "robot":
            response = {
                **say_text_impl(message, speaker_id=speaker_id),
                "target": "robot",
                "handled": True,
            }
        else:
            response = {
                "text": message,
                "speaker_id": speaker_id,
                "status": "ready",
                "target": "web",
                "handled": False,
            }

        state.speech = response
        return response

    @tool
    def generate_motion(
        description: str,
        diffusion_steps: int = default_diffusion_steps,
    ) -> dict[str, Any]:
        """Generate G1 motion files for the current web request."""
        result = generate_motion_impl(description, diffusion_steps=diffusion_steps)
        motions = result.get("motions") or []
        for motion in motions:
            if isinstance(motion, dict):
                state.motions.append(dict(motion))

        warning = result.get("warning")
        if isinstance(warning, str) and warning.strip():
            state.warning = warning.strip()

        return result

    agent = create_agent(tools=[generate_motion, say_text])
    result = agent(prompt)

    reply_text = _extract_text_blocks(getattr(result, "message", None))
    if not reply_text:
        reply_text = state.spoken_text or ""

    if not state.spoken_text and reply_text:
        state.spoken_text = reply_text

    if state.speech is None and state.spoken_text:
        state.speech = {
            "text": state.spoken_text,
            "speaker_id": state.speaker_id,
            "status": "ready" if state.tts_target == "web" else "not_called",
            "target": state.tts_target,
            "handled": False,
        }

    return WebAgentRunResult(
        reply_text=reply_text,
        spoken_text=state.spoken_text,
        speech=state.speech,
        motions=state.motions,
        warning=state.warning,
    )
