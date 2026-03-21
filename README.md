# G1 Voice2Motion

LLM agent that generates Unitree G1 humanoid robot motions from natural language, and a deploy pipeline that plays them on the robot (sim or real).

The system has two entry points that run as separate processes:
- **`uv run agent`** — interactive agent loop. Takes text or voice input, calls [Kimodo](https://github.com/NVIDIA/Kimodo) to generate motions, saves `.pt` files to `output/`.
- **`uv run deploy`** — robot control loop. Watches `output/` for new `.pt` files. Starts in locomotion mode (AMO policy), auto-switches to motion tracking when a file arrives, returns to locomotion when done.

## Architecture

```
User ──► Agent (GPT-4.1) ──► Prompt Refiner (GPT-4.1-mini) ──► Kimodo (diffusion model)
                                                                      │
                                                                 .pt file
                                                                      │
                                                              output/ directory
                                                                      │
                                                    Deploy pipeline (file watcher)
                                                           │              │
                                                      Locomotion ◄──► Motion Tracking
                                                       (AMO)        (BeyondMimic tracker)
                                                           │
                                                    MuJoCo sim / Real G1
```

## Setup

Clone repos side by side:

```bash
git clone --recurse-submodules <this-repo> g1-treasure-hunt
git clone https://github.com/HansZ8/RoboJuDo
git clone https://github.com/unitreerobotics/unitree_sdk2_python  # optional, for TTS
```

If you already cloned without `--recurse-submodules`, pull the submodules manually:

```bash
git submodule update --init --recursive
```

Install:

```bash
cd g1-treasure-hunt
uv sync                    # core agent only
uv sync --extra deploy     # + robot deploy pipeline (requires ../RoboJuDo)
uv sync --extra tts        # + robot speaker (requires ../unitree_sdk2_python + CycloneDDS)
```

The deploy extra requires [RoboJuDo](https://github.com/HansZ8/RoboJuDo) cloned as `../RoboJuDo`. The TTS extra requires [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python) cloned as `../unitree_sdk2_python` and CycloneDDS built locally (`CYCLONEDDS_HOME` must be set before install).

### Environment

Create a `.env` file:

```
OPENAI_API_KEY=sk-...
KIMODO_URL=https://<ngrok-subdomain>.ngrok-free.app
```

Add these if using the robot speaker:

```
UNITREE_NETWORK_INTERFACE=en11
UNITREE_AUDIO_VOLUME=100
```

### Kimodo server

Kimodo runs on a remote GPU instance. Access via ngrok URL (set in `.env`) or SSH tunnel:

```bash
ssh -L 8420:localhost:8420 -L 7860:localhost:7860 <remote-host>
```

- `localhost:8420` — Kimodo API
- `localhost:7860` — Kimodo visualizer ([Viser](https://github.com/nerfstudio-project/viser)), SSH tunnel only

## Kimodo API

The Kimodo server exposes three generation endpoints, all accepting the same request body:

| Endpoint | Response |
|---|---|
| `POST /generate` | JSON with `qpos` trajectory |
| `POST /generate/csv` | Downloadable CSV file |
| `POST /generate/pt` | Binary ProtoMotions MotionLib `.pt` file |
| `GET /health` | Model status, device info, GPU memory |

All requests to ngrok URLs require the header `ngrok-skip-browser-warning: true`.

```bash
curl -X POST http://localhost:8420/generate \
  -H "Content-Type: application/json" \
  -H "ngrok-skip-browser-warning: true" \
  -d '{
    "prompt": "A person walks forward",
    "duration": 3.0,
    "diffusion_steps": 50
  }'
```

Request body fields:

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt` | string | *required* | Motion description. Multiple sentences are treated as sequential segments. |
| `duration` | float | 3.0 | Motion length in seconds (0.5–30.0). Output is 30 fps. |
| `diffusion_steps` | int | 100 | Denoising steps. 50 is fast (~2s), 100 is higher quality (~4s). |
| `num_samples` | int | 1 | Number of motion variations (1–4). |
| `num_transition_frames` | int | 5 | Blending frames between sequential prompt segments. |
| `initial_dof_pos` | float[29] | — | Joint angles in radians for soft pose guidance on frame 0. |
| `final_dof_pos` | float[29] | — | Joint angles in radians for soft pose guidance on the last frame. |
| `constraints` | array | — | Raw Kimodo constraint dicts (root2d, fullbody, end-effector). |

Each output frame has 36 values: root position xyz (3), root quaternion wxyz (4), and 29 joint angles in radians. The `/generate/pt` endpoint returns a ProtoMotions MotionLib file ready for the deploy pipeline. Set client timeout to at least 60 seconds.

## Usage

Run `--help` on either entry point for all available flags.

### Agent

```bash
uv run agent
```

Generates `.pt` motion files into `output/`. The agent runs preflight checks at startup and exits early if dependencies are missing.

### Deploy pipeline

```bash
uv run deploy --motion-path output/some_motion.pt
```

Starts in locomotion mode. Watches `output/` for new `.pt` files. When one appears, auto-switches to motion tracking, plays the motion, then returns to locomotion.

For the real robot:

```bash
uv run deploy --config g1_agent_locomimic_real --motion-path output/some_motion.pt
```

The ONNX tracker model (`unified_pipeline.onnx` + `.yaml`) must be placed in `../RoboJuDo/assets/models/g1/protomotions_bm_tracker/`.

### End-to-end demo

```bash
# Terminal 1: agent
uv run agent --no-tts

# Terminal 2: deploy (sim)
uv run deploy --motion-path output/some_motion.pt
```

Talk to the agent, it generates motions, the robot plays them automatically.

## Project structure

```
src/
  main.py                          # agent entry point
  agent/
    agent.py                       # Strands agent setup, system prompt
    prompt_refiner.py              # GPT-4.1-mini prompt optimization for Kimodo
    tools/
      generate_motion.py           # @tool — calls Kimodo API, saves .pt files
  g1/
    audio/                         # Unitree TTS (robot speaker)
    speech_input/                  # Microphone input + OpenAI transcription
  deploy/
    run.py                         # deploy entry point (tyro CLI)
    agent_tracker_policy.py        # file watcher + auto-switch policy
    configs.py                     # LocoMimic configs for sim and real robot
```

## Development

```bash
make          # uv sync + ruff + basedpyright
make lint     # ruff + basedpyright only
```
