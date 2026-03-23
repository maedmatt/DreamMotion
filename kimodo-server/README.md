# Kimodo Motion Generation API

FastAPI wrapper around NVIDIA's [Kimodo](https://research.nvidia.com/labs/sil/projects/kimodo/) text-to-motion diffusion model for the Unitree G1 humanoid robot. One HTTP call: text prompt in, robot-ready motion trajectory out.

## Requirements

- NVIDIA GPU with 17GB+ VRAM (tested: RTX 3090, RTX 4090, A10G, A100)
- Docker with NVIDIA Container Toolkit
- Hugging Face account with access to `meta-llama/Meta-Llama-3-8B-Instruct`

## Quick Start

```bash
# 1. Save your HF token
mkdir -p ~/.cache/huggingface
echo "hf_YOUR_TOKEN" > ~/.cache/huggingface/token

# 2. Run setup (clones Kimodo, builds images, starts services)
chmod +x setup.sh
./setup.sh

# 3. Test
curl -s -X POST http://localhost:8420/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"A person walks forward","duration":2.0,"diffusion_steps":50}'
```

First run takes 10-15 minutes (Docker build + model weight download). Subsequent starts take ~30 seconds.

## Manual Setup

If you prefer step-by-step:

```bash
# Clone Kimodo
git clone https://github.com/nv-tlabs/kimodo.git
cd kimodo && git clone https://github.com/nv-tlabs/kimodo-viser.git && cd ..

# Build base image
cd kimodo && docker compose build && cd ..

# Build API image
docker build -f Dockerfile.api -t kimodo-api:latest .

# Start text encoder (first run downloads Llama 3 8B, ~16GB)
docker compose up -d text-encoder

# Wait for it (check with: docker compose logs -f text-encoder)
# Once healthy, start API
docker compose up -d api

# Optional: start visual demo on port 7860
docker compose --profile demo up -d demo
```

## Endpoints

All endpoints accept the same request body and differ only in response format.

### POST /generate

Returns JSON with the G1 qpos trajectory.

### POST /generate/csv

Returns a downloadable CSV file (no header, 36 columns per row).

### POST /generate/pt

Returns a binary ProtoMotions MotionLib `.pt` file. Load with `torch.load()`.

### GET /health

Returns model status, device info, and GPU memory usage.

## Request Body

```json
{
  "prompt": "A person walks forward and waves",
  "duration": 3.0,
  "diffusion_steps": 50,
  "num_samples": 1,
  "num_transition_frames": 5,
  "cfg_type": "regular",
  "cfg_weight": [3.0],
  "initial_dof_pos": [29 floats],
  "final_dof_pos": [29 floats],
  "initial_root_pos": [x, y, z],
  "initial_root_quat": [w, x, y, z],
  "final_root_pos": [x, y, z],
  "final_root_quat": [w, x, y, z],
  "constraints": [array of constraint objects]
}
```

Only `prompt` is required. All other fields are optional.

### Parameters

`prompt` (string, required): text description of the desired motion. Periods split the prompt into multiple segments — `"Walk forward. Wave hand."` generates two consecutive motions of `duration` seconds each.

`duration` (float, default 3.0): motion length in seconds per prompt segment (0.5 to 30.0). Output is at 30fps.

`diffusion_steps` (int, default 100): denoising steps. 50 for speed (~2s), 100 for quality (~4s).

`num_samples` (int, default 1): number of motion variations (1 to 4).

`num_transition_frames` (int, default 5): blending frames between consecutive prompt segments. Only relevant for multi-segment prompts.

`cfg_type` (string, optional): classifier-free guidance mode. "nocfg", "regular", or "separated".

`cfg_weight` (list of floats, optional): 1 float for regular, 2 floats [text_weight, constraint_weight] for separated.

`initial_dof_pos` / `final_dof_pos` (list of 29 floats, optional): G1 joint angles in radians for soft pose constraints on first/last frame.

`initial_root_pos` / `final_root_pos` (list of 3 floats, optional): MuJoCo root position [x,y,z] in Z-up coordinates. The server converts to Kimodo's Y-up space internally.

`initial_root_quat` / `final_root_quat` (list of 4 floats, optional): MuJoCo root quaternion [w,x,y,z]. Converted to Kimodo space internally.

`constraints` (list of dicts, optional): raw Kimodo constraint objects for fine-grained control.

### Response (JSON)

```json
{
  "prompt": "A person walks forward.",
  "duration": 3.0,
  "num_frames": 90,
  "fps": 30.0,
  "model": "kimodo-g1-rp",
  "generation_time_s": 2.34,
  "qpos": [[36 floats], ...]
}
```

Note: G1 models skip Kimodo's post-processing step — the qpos output is the raw diffusion model output converted to MuJoCo joint space.

## Qpos Output Format

Each frame contains 36 values in MuJoCo coordinate system (Z-up, X-forward):

- Indices 0-2: root translation (x, y, z) in meters
- Indices 3-6: root rotation quaternion (w, x, y, z)
- Indices 7-35: 29 joint angles in radians

DOF order: left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, left_ankle_pitch, left_ankle_roll, right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee, right_ankle_pitch, right_ankle_roll, waist_yaw, waist_roll, waist_pitch, left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow, left_wrist_roll, left_wrist_pitch, left_wrist_yaw, right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow, right_wrist_roll, right_wrist_pitch, right_wrist_yaw.

## Constraint Types

All constraints are soft diffusion guidance (~5-10cm position error).

### root2d

Controls ground-plane trajectory:

```json
{"type": "root2d", "frame_indices": [0, 30, 59], "smooth_root_2d": [[0,0], [0.5,0], [1.0,0]]}
```

### fullbody

Pins entire body pose at specific frames:

```json
{"type": "fullbody", "frame_indices": [59], "local_joints_rot": [[[ax,ay,az] x 34]], "root_positions": [[x,y,z]]}
```

### End-effector

Controls hands/feet only. Types: `left-hand`, `right-hand`, `left-foot`, `right-foot`. Same fields as fullbody.

## Python Client Example

```python
import httpx

API = "http://localhost:8420"

resp = httpx.post(f"{API}/generate", json={
    "prompt": "A person walks forward",
    "duration": 3.0,
    "diffusion_steps": 50,
}, timeout=120.0)

data = resp.json()
trajectory = data["qpos"]  # [frames][36 values]

# Get .pt file
resp = httpx.post(f"{API}/generate/pt", json={
    "prompt": "A person walks forward",
    "duration": 3.0,
    "diffusion_steps": 50,
}, timeout=120.0)
with open("motion.pt", "wb") as f:
    f.write(resp.content)
```

## Remote Access

### SSH tunnel

If the server runs on a remote machine (e.g. a GPU workstation):

```bash
ssh -L 8420:localhost:8420 user@gpu-host
```

Then set `KIMODO_URL=http://localhost:8420` in your local `.env`.

### ngrok

To expose the API publicly (e.g. for a hackathon):

```bash
sudo snap install ngrok  # or: brew install ngrok
ngrok config add-authtoken <your-token>
ngrok http 8420
```

Add `ngrok-skip-browser-warning: true` header to API calls from code.

## Architecture

```
text-encoder (port 9550)       api (port 8420)
Llama 3 8B + LLM2Vec    <---  FastAPI + Kimodo G1 diffusion
~14GB VRAM                     ~3GB VRAM
```

## Troubleshooting

Text encoder stays unhealthy: first run downloads ~16GB of model weights. Wait 5-10 minutes. Check logs with `docker compose logs -f text-encoder`.

API returns 503: text encoder isn't ready yet. Wait for the healthcheck to pass.

Generation is slow: 50 diffusion steps = ~2-3s for a 2s motion. 100 steps = ~4-5s. First request after startup is slower (model warmup).

Out of disk space: remove unused CUDA toolkit versions (`sudo rm -rf /usr/local/cuda-12.{4,5,6}`). The HF cache (~18GB) cannot be removed.

## Files

```
.
├── setup.sh                  # One-command setup
├── Dockerfile.api            # API image (extends kimodo:1.0)
├── docker-compose.yaml       # Orchestrates all services
├── api/
│   ├── server.py             # FastAPI server
│   ├── dof_constraints.py    # DOF → constraint conversion (MuJoCo ↔ Kimodo coords)
│   ├── csv_to_motionlib.py   # Standalone CSV → ProtoMotions .pt converter
│   └── CSV_TO_MOTIONLIB_GUIDE.md
└── kimodo/                   # Cloned by setup.sh (not committed)
```

## License

MIT. See [LICENSE](../LICENSE). Kimodo itself is licensed under Apache-2.0 by NVIDIA. Llama 3 requires accepting Meta's community license.
