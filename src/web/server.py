"""Web UI server wrapping the g1-treasure-hunt agent.

Provides a FastAPI web interface with:
- Text and voice input for motion descriptions
- LLM text reply generation (always)
- Kimodo motion generation + 3D preview (always)
- ZMQ deployment to the robot

Usage::

    uv run web
    uv run web --kimodo-url https://xxx.ngrok-free.app --port 8000
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import tempfile
import threading
import time
import webbrowser
from pathlib import Path

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

from agent.prompt_refiner import refine_prompt
from agent.tools.generate_motion import _call_kimodo

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
TEMPLATES_DIR = WEB_DIR / "templates"

# ---------------------------------------------------------------------------
# Reply generation prompt
# ---------------------------------------------------------------------------

REPLY_SYSTEM_PROMPT = """\
You are a casual, witty companion living inside a humanoid robot.

You receive the user's message. A motion has already been generated in the
background — you do NOT need to describe it, summarize it, or mention its
duration. Pretend the motion side of things is handled by someone else.

Your ONLY job is to reply to the user the way a friend would in a chat:
- Be natural, warm, maybe a little playful or humorous.
- React to what the user said, not to the motion that was generated.
- If the user says "throw a basketball", you might say "Let's go! 🏀"
  or "三分球!稳了" — NOT "I've created a 6-second throwing motion...".
- Keep it short: 1-2 sentences max.
- NEVER mention prompts, durations, motion generation, diffusion,
  Kimodo, CSV, PT files, or any technical details.
- ALWAYS reply in English, regardless of what language the user writes in.
"""

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    prompt: str


class DeployRequest(BaseModel):
    task_id: str


class TranscribeRequest(BaseModel):
    audio_base64: str
    mime_type: str | None = None


class SpeakRequest(BaseModel):
    text: str
    target: str = "web"
    voice: str = "alloy"
    speaker_id: int = 1


# ---------------------------------------------------------------------------
# In-memory stores
# ---------------------------------------------------------------------------

GENERATED_MOTION_DATA: dict[str, dict[str, str]] = {}
BOOTSTRAP_TASK_ID = "__bootstrap__"
GENERATED_MOTION_DATA[BOOTSTRAP_TASK_ID] = {"prompt": "Ready", "csv": ""}

TRANSCRIBE_MAX_BYTES = 20 * 1024 * 1024
MIME_TO_SUFFIX = {
    "audio/webm": ".webm",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
    "audio/ogg": ".ogg",
}

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="G1 Motion Agent")


@app.get("/")
def index():
    return FileResponse(TEMPLATES_DIR / "index.html")


@app.post("/api/generate")
def api_generate(req: GenerateRequest):
    """Generate motion + text reply for a user prompt.

    Always produces both outputs regardless of what the user says.
    """
    diffusion_steps: int = app.state.diffusion_steps  # pyright: ignore[reportAttributeAccessIssue]

    # 1. Refine prompt → Kimodo prompts
    refined = refine_prompt(req.prompt)
    prompts: list[str] = refined.get("prompts") or [req.prompt]
    durations: list[float] = refined.get("durations") or [5.0] * len(prompts)
    warning: str | None = refined.get("warning")

    if len(durations) < len(prompts):
        durations.extend([5.0] * (len(prompts) - len(durations)))

    # 2. Generate motion (first prompt for the viewer)
    prompt_text = prompts[0]
    duration = durations[0]

    try:
        qpos_path, pt_path = _call_kimodo(prompt_text, duration, diffusion_steps)
    except httpx.HTTPError as exc:
        logger.exception("Kimodo generation failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    # 3. Store results for viewer / deploy
    task_id = str(int(time.time() * 1000))
    csv_content = qpos_path.read_text() if qpos_path.exists() else ""

    motion_data: dict[str, str] = {"prompt": prompt_text, "csv": csv_content}
    if pt_path:
        motion_data["pt_path"] = str(pt_path)
    GENERATED_MOTION_DATA[task_id] = motion_data

    # 4. Generate text reply via LLM
    text_reply = _generate_reply(req.prompt, prompts, durations, warning)

    result: dict = {
        "text_reply": text_reply,
        "prompts": prompts,
        "durations": durations,
        "task_id": task_id,
        "viewer_url": f"/viewer/{task_id}",
    }
    if warning:
        result["warning"] = warning
    return result


@app.get("/viewer/{task_id}")
def viewer_page(task_id: str):
    if task_id not in GENERATED_MOTION_DATA:
        raise HTTPException(status_code=404, detail="Viewer not found")
    raw = (TEMPLATES_DIR / "viewer.html").read_text()
    html = raw.replace("__TASK_ID_JSON__", json.dumps(task_id))
    return HTMLResponse(content=html)


@app.get("/api/motion/{task_id}")
def get_motion_data(task_id: str):
    payload = GENERATED_MOTION_DATA.get(task_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Motion data not found")
    return payload


@app.post("/api/transcribe")
def transcribe_audio(req: TranscribeRequest):
    transcriber = getattr(app.state, "transcriber", None)
    if transcriber is None:
        raise HTTPException(
            status_code=503, detail="Speech transcriber not initialized"
        )

    try:
        audio_bytes = base64.b64decode(req.audio_base64, validate=True)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail="Invalid base64 audio payload"
        ) from exc

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio payload")
    if len(audio_bytes) > TRANSCRIBE_MAX_BYTES:
        raise HTTPException(status_code=413, detail="Audio payload too large")

    suffix = MIME_TO_SUFFIX.get(
        (req.mime_type or "").split(";")[0].strip().lower(), ".webm"
    )
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="stt_", suffix=suffix, delete=False
        ) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = Path(tmp_file.name)

        text = transcriber.transcribe(tmp_path)
        return {"text": text}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Speech transcription failed")
        raise HTTPException(
            status_code=500, detail=f"Transcription failed: {exc}"
        ) from exc
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@app.post("/api/deploy")
def deploy_motion(req: DeployRequest):
    data = GENERATED_MOTION_DATA.get(req.task_id)
    if not data:
        raise HTTPException(status_code=404, detail="Motion not found")

    pt_path = data.get("pt_path")
    if not pt_path:
        raise HTTPException(status_code=404, detail="No .pt file for this motion")

    return {"status": "deployed", "pt_path": pt_path}


@app.post("/api/speak")
def speak(req: SpeakRequest):
    """Speak the agent reply via web (OpenAI TTS) or robot (Unitree)."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    if req.target == "web":
        try:
            client = OpenAI()
            tts_resp = client.audio.speech.create(
                model="tts-1",
                voice=req.voice,
                input=req.text,
                response_format="mp3",
            )
            return Response(
                content=tts_resp.content,
                media_type="audio/mpeg",
            )
        except Exception as exc:
            logger.exception("OpenAI TTS failed")
            raise HTTPException(status_code=500, detail=f"TTS failed: {exc}") from exc

    if req.target == "robot":
        try:
            from g1.audio.client import get_unitree_audio_service

            service = get_unitree_audio_service()
            result = service.say_text(text=req.text, speaker_id=req.speaker_id)
            return {"status": "ok", "detail": result}
        except Exception as exc:
            logger.exception("Robot TTS failed")
            raise HTTPException(
                status_code=503,
                detail=f"Robot TTS unavailable: {exc}",
            ) from exc

    raise HTTPException(status_code=400, detail=f"Unknown target: {req.target}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_reply(
    user_prompt: str,
    prompts: list[str],
    durations: list[float],
    warning: str | None,
) -> str:
    """Call the LLM to produce a conversational reply about the generated motion."""
    try:
        client = OpenAI()
        motion_summary = json.dumps(
            [
                {"prompt": p, "duration": d}
                for p, d in zip(prompts, durations, strict=True)
            ]
        )
        user_msg = f"User message: {user_prompt}\n\nGenerated motions: {motion_summary}"
        if warning:
            user_msg += f"\nWarning: {warning}"

        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": REPLY_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        logger.exception("Reply generation failed, using fallback")
        total = sum(durations)
        return f"Generated {len(prompts)} motion(s): {prompts[0]} ({total:.1f}s total)"


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G1 Motion Agent — Web UI")
    parser.add_argument(
        "--kimodo-url",
        default=None,
        help="Kimodo server URL (default: KIMODO_URL env or http://localhost:8420)",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Web server port (default: 8000)"
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=50,
        help="Kimodo diffusion steps (default: 50)",
    )
    parser.add_argument(
        "--transcribe-model",
        default=None,
        help="OpenAI STT model (default: gpt-4o-transcribe)",
    )
    return parser.parse_args()


def _preflight(kimodo_url: str) -> list[str]:
    errors: list[str] = []
    if not os.environ.get("OPENAI_API_KEY"):
        errors.append("OPENAI_API_KEY is not set")
    try:
        r = httpx.get(f"{kimodo_url}/health", timeout=5.0)
        r.raise_for_status()
        print(f"  kimodo: ok ({kimodo_url})")
    except httpx.HTTPError:
        errors.append(f"Kimodo unreachable at {kimodo_url}")
    return errors


def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    args = _parse_args()
    kimodo_url = args.kimodo_url or os.environ.get(
        "KIMODO_URL", "http://localhost:8420"
    )
    os.environ["KIMODO_URL"] = kimodo_url

    print("preflight checks:")
    errors = _preflight(kimodo_url)

    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        if any("OPENAI_API_KEY" in e for e in errors):
            import sys

            sys.exit(1)

    # Speech transcriber
    from g1.speech_input.config import SpeechInputConfig
    from g1.speech_input.transcriber import OpenAISpeechTranscriber

    transcribe_model = args.transcribe_model or os.environ.get(
        "OPENAI_TRANSCRIBE_MODEL", "gpt-4o-transcribe"
    )
    speech_cfg = SpeechInputConfig(
        sample_rate=48000,
        channels=1,
        microphone_device=None,
        transcribe_model=transcribe_model,
    )
    app.state.transcriber = OpenAISpeechTranscriber(speech_cfg)
    app.state.kimodo_url = kimodo_url
    app.state.diffusion_steps = args.diffusion_steps

    # Mount static files (g1_description first for higher priority)
    g1_desc_dir = (
        WEB_DIR.parent.parent.parent
        / "hack26-ethrc-deploy"
        / "scripts"
        / "static"
        / "g1_description"
    )
    if g1_desc_dir.is_dir():
        app.mount(
            "/static/g1_description",
            StaticFiles(directory=g1_desc_dir),
            name="g1_description",
        )
    elif (STATIC_DIR / "g1_description").is_dir():
        pass  # already covered by the general static mount
    else:
        print("  WARNING: g1_description not found, 3D viewer may not work")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    print(f"\nWeb UI starting on http://localhost:{args.port}")

    def _open_browser() -> None:
        time.sleep(1.5)
        webbrowser.open(f"http://localhost:{args.port}")

    threading.Thread(target=_open_browser, daemon=True).start()

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
