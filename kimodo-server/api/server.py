"""
Kimodo Motion Generation API
FastAPI wrapper around NVIDIA Kimodo for G1 humanoid motion generation.
"""

import io
import time
import logging
import tempfile
import os
from contextlib import asynccontextmanager
from typing import Optional, List

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from kimodo import load_model
from kimodo.model.registry import get_model_info
from dof_constraints import load_all_constraints
from csv_to_motionlib import convert_csv_to_motion, _pack_motionlib

logger = logging.getLogger("kimodo_api")
logging.basicConfig(level=logging.INFO)

MODEL = None
RESOLVED_MODEL = None
SKELETON = None
QPOS_CONVERTER = None
DEVICE = "cuda:0"
MODEL_NAME = "Kimodo-G1-RP-v1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, RESOLVED_MODEL, SKELETON, QPOS_CONVERTER

    logger.info(f"Loading model: {MODEL_NAME}")
    t0 = time.time()

    MODEL, RESOLVED_MODEL = load_model(
        MODEL_NAME,
        device=DEVICE,
        default_family="Kimodo",
        return_resolved_name=True,
    )
    SKELETON = MODEL.skeleton

    info = get_model_info(RESOLVED_MODEL)
    display = info.display_name if info else RESOLVED_MODEL
    logger.info(f"Loaded: {display} ({RESOLVED_MODEL}) in {time.time() - t0:.1f}s")
    logger.info(f"FPS: {MODEL.fps}")

    if "g1" in RESOLVED_MODEL:
        from kimodo.exports.mujoco import MujocoQposConverter
        QPOS_CONVERTER = MujocoQposConverter(SKELETON)
        logger.info("MuJoCo qpos converter ready")

    yield

    logger.info("Shutting down")
    del MODEL
    torch.cuda.empty_cache()


app = FastAPI(title="Kimodo API", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Motion description")
    duration: float = Field(default=3.0, ge=0.5, le=30.0)
    num_samples: int = Field(default=1, ge=1, le=4)
    diffusion_steps: int = Field(default=100, ge=10, le=200)
    num_transition_frames: int = Field(default=5)
    cfg_type: Optional[str] = Field(
        default=None,
        description="CFG mode: nocfg, regular, or separated"
    )
    cfg_weight: Optional[List[float]] = Field(
        default=None,
        description="CFG weight(s): 1 float for regular, 2 floats [text_weight, constraint_weight] for separated"
    )
    initial_dof_pos: Optional[List[float]] = Field(
        default=None,
        description="29 DOF angles (rad) for initial pose constraint (soft guidance)"
    )
    final_dof_pos: Optional[List[float]] = Field(
        default=None,
        description="29 DOF angles (rad) for final pose constraint (soft guidance)"
    )
    initial_root_pos: Optional[List[float]] = Field(
        default=None,
        description="MuJoCo root position [x,y,z] (Z-up) for initial frame"
    )
    initial_root_quat: Optional[List[float]] = Field(
        default=None,
        description="MuJoCo root quaternion [w,x,y,z] for initial frame"
    )
    final_root_pos: Optional[List[float]] = Field(
        default=None,
        description="MuJoCo root position [x,y,z] (Z-up) for final frame"
    )
    final_root_quat: Optional[List[float]] = Field(
        default=None,
        description="MuJoCo root quaternion [w,x,y,z] for final frame"
    )
    constraints: Optional[List[dict]] = Field(
        default=None,
        description="Raw Kimodo constraint dicts. Types: root2d, fullbody, left-hand, right-hand, left-foot, right-foot, end-effector"
    )


class GenerateResponse(BaseModel):
    prompt: str
    duration: float
    num_frames: int
    fps: float
    model: str
    generation_time_s: float
    qpos: list[list[float]]


@app.get("/health")
async def health():
    return {
        "status": "ok" if MODEL is not None else "loading",
        "model": RESOLVED_MODEL,
        "device": DEVICE,
        "gpu_mem_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate_motion(req: GenerateRequest):
    if MODEL is None:
        raise HTTPException(503, "Model not loaded")

    fps = MODEL.fps
    texts = [t.strip() + "." for t in req.prompt.split(".") if t.strip()]
    num_frames = [int(req.duration * fps)] * len(texts)
    total_frames = sum(num_frames)

    logger.info(f"Generating: {texts} | {req.duration}s | {total_frames} frames")

    # Build constraints
    constraint_lst = load_all_constraints(
        SKELETON, total_frames,
        initial_dof_pos=req.initial_dof_pos,
        final_dof_pos=req.final_dof_pos,
        constraints=req.constraints,
        initial_root_pos=req.initial_root_pos,
        initial_root_quat=req.initial_root_quat,
        final_root_pos=req.final_root_pos,
        final_root_quat=req.final_root_quat,
    )
    if constraint_lst:
        logger.info(f"Using {len(constraint_lst)} constraint(s)")

    # Build CFG kwargs
    cfg_kwargs = {}
    if req.cfg_type is not None:
        cfg_kwargs["cfg_type"] = req.cfg_type
    if req.cfg_weight is not None:
        if len(req.cfg_weight) == 1:
            cfg_kwargs["cfg_weight"] = req.cfg_weight[0]
        else:
            cfg_kwargs["cfg_weight"] = req.cfg_weight
    if cfg_kwargs:
        logger.info(f"CFG: {cfg_kwargs}")

    t0 = time.time()
    try:
        use_postprocess = "g1" not in RESOLVED_MODEL

        output = MODEL(
            texts,
            num_frames,
            constraint_lst=constraint_lst,
            num_denoising_steps=req.diffusion_steps,
            num_samples=req.num_samples,
            multi_prompt=True,
            num_transition_frames=req.num_transition_frames,
            post_processing=use_postprocess,
            return_numpy=True,
            **cfg_kwargs,
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(500, str(e))

    gen_time = time.time() - t0
    logger.info(f"Generated in {gen_time:.1f}s")

    if QPOS_CONVERTER is not None:
        qpos = QPOS_CONVERTER.dict_to_qpos(output, DEVICE)
        if isinstance(qpos, torch.Tensor):
            qpos = qpos.cpu().numpy()
        # Unwrap single sample: (1, T, 36) -> (T, 36)
        if qpos.ndim == 3:
            qpos = qpos[0]
        qpos_list = qpos.tolist()
    else:
        qpos_list = output["posed_joints"][0].tolist()

    return GenerateResponse(
        prompt=req.prompt,
        duration=req.duration,
        num_frames=len(qpos_list),
        fps=fps,
        model=RESOLVED_MODEL,
        generation_time_s=round(gen_time, 2),
        qpos=qpos_list,
    )


@app.post("/generate/csv")
async def generate_csv(req: GenerateRequest):
    response = await generate_motion(req)
    csv_lines = [",".join(f"{v:.10e}" for v in row) for row in response.qpos]
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        io.StringIO("\n".join(csv_lines)),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=motion_qpos.csv"},
    )


@app.post("/generate/pt")
async def generate_pt(req: GenerateRequest):
    """Generate motion and return packaged ProtoMotions .pt MotionLib file."""
    response = await generate_motion(req)

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "motion.csv")
        with open(csv_path, "w") as f:
            for row in response.qpos:
                f.write(",".join(f"{v:.10e}" for v in row) + "\n")

        motion = convert_csv_to_motion(csv_path, fps=int(response.fps))
        lib = _pack_motionlib([motion], [f"kimodo_{req.prompt[:40]}"])
        pt_path = os.path.join(tmpdir, "motion.pt")
        torch.save(lib, pt_path)

        with open(pt_path, "rb") as f:
            pt_bytes = f.read()

    return Response(
        content=pt_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=kimodo_motion.pt"},
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8420)
