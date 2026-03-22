"""Web UI server wrapping the g1-treasure-hunt agent.

Provides a FastAPI web interface with:
- Text and voice input for agent requests
- OpenAI speech-to-text from the laptop microphone
- Agent-driven speech, optional motion generation, and 3D preview

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
from web.agent_runner import run_agent_for_web

logger = logging.getLogger(__name__)

CANDIDATES_DIR = Path("output/candidates")
CANDIDATE_SESSIONS: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
TEMPLATES_DIR = WEB_DIR / "templates"

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    prompt: str
    tts_target: str = "web"
    speaker_id: int = 1
    voice: str = "alloy"


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


_CSV_HEADER = ",".join(
    ["root_x", "root_y", "root_z", "quat_w", "quat_x", "quat_y", "quat_z"]
    + [f"joint_{i}" for i in range(29)]
)


def _ensure_csv_header(csv_text: str) -> str:
    """Prepend a header row if the CSV has none (raw Kimodo output)."""
    if not csv_text or not csv_text.strip():
        return csv_text
    first_line = csv_text.lstrip().split("\n", 1)[0]
    try:
        float(first_line.split(",", 1)[0])
    except ValueError:
        return csv_text
    return _CSV_HEADER + "\n" + csv_text


def _store_motion_for_viewer(
    motions: list[dict[str, object]],
) -> tuple[str | None, str | None]:
    for motion in motions:
        qpos_path_raw = motion.get("qpos_path")
        if not isinstance(qpos_path_raw, str) or not qpos_path_raw:
            continue

        qpos_path = Path(qpos_path_raw)
        if not qpos_path.exists():
            continue

        task_id = str(time.time_ns())
        csv_content = _ensure_csv_header(qpos_path.read_text())
        payload: dict[str, str] = {
            "prompt": str(motion.get("prompt") or qpos_path.name),
            "csv": csv_content,
        }

        pt_path_raw = motion.get("pt_path")
        if isinstance(pt_path_raw, str) and pt_path_raw:
            payload["pt_path"] = pt_path_raw

        GENERATED_MOTION_DATA[task_id] = payload
        return task_id, f"/viewer/{task_id}"

    return None, None


@app.post("/api/generate")
def api_generate(req: GenerateRequest):
    """Run the agent on a text prompt and expose any generated motion to the UI."""
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Empty prompt")
    if req.tts_target not in {"web", "robot"}:
        raise HTTPException(status_code=400, detail="Unsupported speech target")

    diffusion_steps: int = app.state.diffusion_steps  # pyright: ignore[reportAttributeAccessIssue]

    try:
        agent_result = run_agent_for_web(
            prompt,
            tts_target=req.tts_target,  # pyright: ignore[reportArgumentType]
            speaker_id=req.speaker_id,
            diffusion_steps=diffusion_steps,
            voice=req.voice,
        )
    except Exception as exc:
        logger.exception("Agent invocation failed")
        raise HTTPException(
            status_code=500,
            detail=f"Agent invocation failed: {exc}",
        ) from exc

    prompts = [
        motion["prompt"]
        for motion in agent_result.motions
        if isinstance(motion.get("prompt"), str)
    ]
    durations = [
        float(motion["duration"])  # pyright: ignore[reportArgumentType]
        for motion in agent_result.motions
        if isinstance(motion.get("duration"), int | float)
    ]

    task_id, viewer_url = _store_motion_for_viewer(agent_result.motions)

    result: dict[str, object] = {
        "text_reply": agent_result.reply_text,
        "spoken_text": agent_result.spoken_text,
        "speech": agent_result.speech,
        "motions": agent_result.motions,
        "prompts": prompts,
        "durations": durations,
    }
    if task_id and viewer_url:
        result["task_id"] = task_id
        result["viewer_url"] = viewer_url
    if agent_result.warning:
        result["warning"] = agent_result.warning
    return result


# ---------------------------------------------------------------------------
# Multi-sample candidate selection
# ---------------------------------------------------------------------------


class CandidateRequest(BaseModel):
    prompt: str
    num_samples: int = 3
    diffusion_steps: int = 50


class SelectRequest(BaseModel):
    session_id: str
    sample_index: int


@app.post("/api/generate-candidates")
def api_generate_candidates(req: CandidateRequest):
    """Generate multiple motion candidates for user selection."""
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Empty prompt")

    refined = refine_prompt(prompt)
    kimodo_prompt = refined["prompts"][0]
    duration = refined["durations"][0]

    kimodo_url = os.environ.get("KIMODO_URL", "http://localhost:8420")
    headers = {"ngrok-skip-browser-warning": "true"}
    body = {
        "prompt": kimodo_prompt,
        "duration": duration,
        "diffusion_steps": req.diffusion_steps,
        "num_samples": req.num_samples,
    }

    json_resp = httpx.post(
        f"{kimodo_url}/generate", headers=headers, json=body, timeout=120.0
    )
    json_resp.raise_for_status()
    data = json_resp.json()
    samples = data["qpos"]

    pt_resp = httpx.post(
        f"{kimodo_url}/generate/pt", headers=headers, json=body, timeout=120.0
    )
    pt_resp.raise_for_status()

    session_id = str(time.time_ns())
    session_dir = CANDIDATES_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    pt_path = session_dir / "motions.pt"
    pt_path.write_bytes(pt_resp.content)

    csv_header = "root_x,root_y,root_z,quat_w,quat_x,quat_y,quat_z," + ",".join(
        f"joint_{i}" for i in range(29)
    )

    candidate_tasks = []
    for i, sample_qpos in enumerate(samples):
        csv_lines = [csv_header]
        for frame in sample_qpos:
            csv_lines.append(",".join(str(v) for v in frame))
        csv_text = "\n".join(csv_lines)

        task_id = f"{session_id}_sample_{i}"
        GENERATED_MOTION_DATA[task_id] = {
            "prompt": kimodo_prompt,
            "csv": csv_text,
        }
        candidate_tasks.append(
            {
                "task_id": task_id,
                "viewer_url": f"/viewer/{task_id}",
                "num_frames": len(sample_qpos),
            }
        )

    CANDIDATE_SESSIONS[session_id] = {
        "pt_path": str(pt_path),
        "num_samples": req.num_samples,
        "prompt": kimodo_prompt,
        "duration": duration,
    }

    return {
        "session_id": session_id,
        "prompt": kimodo_prompt,
        "duration": duration,
        "candidates": candidate_tasks,
        "warning": refined.get("warning"),
    }


@app.post("/api/select-candidate")
def api_select_candidate(req: SelectRequest):
    """Extract selected motion sample and save to output/ for deploy."""
    import torch

    session = CANDIDATE_SESSIONS.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if req.sample_index < 0 or req.sample_index >= session["num_samples"]:
        raise HTTPException(status_code=400, detail="Invalid sample index")

    pt_path = Path(session["pt_path"])
    if not pt_path.exists():
        raise HTTPException(status_code=404, detail="PT file not found")

    lib = torch.load(str(pt_path), weights_only=False)
    start = int(lib["length_starts"][req.sample_index])
    nf = int(lib["motion_num_frames"][req.sample_index])

    single = {}
    for key in ("gts", "grs", "gavs", "gvs", "dvs", "dps", "contacts"):
        single[key] = lib[key][start : start + nf]
    single["motion_num_frames"] = torch.tensor([nf])
    single["length_starts"] = torch.tensor([0])
    single["motion_dt"] = lib["motion_dt"][:1]
    single["motion_lengths"] = lib["motion_lengths"][
        req.sample_index : req.sample_index + 1
    ]
    single["motion_weights"] = torch.ones(1)
    single["motion_files"] = (f"selected_{req.session_id}",)

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    from datetime import UTC, datetime

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    out_path = output_dir / f"qpos_{timestamp}.pt"
    torch.save(single, str(out_path))

    return {
        "status": "selected",
        "sample_index": req.sample_index,
        "output_path": str(out_path),
    }


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
            from g1.audio.tool import say_text_impl

            result = say_text_impl(
                text=req.text,
                speaker_id=req.speaker_id,
                voice=req.voice,
            )
            if result.get('status') == 'failed':
                raise RuntimeError(result.get('error') or 'Robot speaker failed')
            return {"status": "ok", "detail": result}
        except Exception as exc:
            logger.exception("Robot TTS failed")
            raise HTTPException(
                status_code=503,
                detail=f"Robot TTS unavailable: {exc}",
            ) from exc

    raise HTTPException(status_code=400, detail=f"Unknown target: {req.target}")


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
        transcribe_language="en",
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
