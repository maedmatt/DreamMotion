# G1 Treasure Hunt

LLM agent that generates Unitree G1 humanoid robot motions from natural language. Uses [Kimodo](https://github.com/NVIDIA/Kimodo) for motion generation via a Strands agent with OpenAI.

## Setup

```bash
uv sync
```

### Unitree SDK install

The audio tool uses Unitree's official Python SDK, which depends on CycloneDDS.
Install it on the same machine where you will run this project.

1. Build and install CycloneDDS:

```bash
sudo apt update
sudo apt install -y cmake build-essential python3-pip

cd ~
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds
mkdir -p build install
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/cyclonedds/install
cmake --build . --target install -j"$(nproc)"
```

2. Clone Unitree's Python SDK:

```bash
cd ~
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
```

3. Install the SDK into this project's environment:

```bash
export CYCLONEDDS_HOME=$HOME/cyclonedds/install
cd <path_to_repo>
uv pip install -e ~/unitree_sdk2_python
```

4. Before using the speaker tool, set the robot-facing network interface:

```bash
export UNITREE_NETWORK_INTERFACE=<your_ethernet_interface>
```

### Microphone input setup

Microphone mode uses your laptop microphone through `sounddevice`, which requires a local PortAudio installation.

1. Install PortAudio on your machine:

```bash
sudo apt update
sudo apt install -y libportaudio2 portaudio19-dev
```

2. List available audio devices:

```bash
cd <path_to_repo>
uv run python - <<'PY'
import sounddevice as sd
print(sd.query_devices())
PY
```

Pick a device with input channels greater than `0`. Ignore HDMI devices because they are outputs only.

3. Select the microphone device by index:

```bash
export VOICE_INPUT_DEVICE=<device_index>
```

### Kimodo

The Kimodo server runs on a remote GPU instance. Either use the ngrok URL or an SSH tunnel:

```bash
# Option 1: ngrok (set in .env)
KIMODO_URL=https://<ngrok-subdomain>.ngrok-free.app

# Option 2: SSH tunnel
ssh -L 8420:localhost:8420 -L 7860:localhost:7860 <remote-host>
```

- `localhost:7860` — Kimodo visualizer ([Viser](https://github.com/nerfstudio-project/viser)), requires SSH tunnel

Verify the connection (ngrok requires the skip-browser-warning header):

```bash
curl -H "ngrok-skip-browser-warning: true" http://localhost:8420/health
```

### Environment

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
KIMODO_URL=http://localhost:8420
UNITREE_NETWORK_INTERFACE=en11
UNITREE_AUDIO_VOLUME=100
```

## Usage

### Agent (interactive)

```bash
uv run agent                    # text input, all modules on
uv run agent --mode mic         # microphone input
uv run agent --no-tts           # disable robot speaker
uv run agent --kimodo-url URL   # override Kimodo endpoint
```

In microphone mode, press Enter to start recording, press Enter again to stop.

The agent runs preflight checks at startup — it verifies the OpenAI key, Kimodo health, network interface (if TTS), and microphone (if mic mode). Missing dependencies are caught before the loop starts.

Results are saved as CSV and `.pt` in `output/`. The deploy pipeline ([hack26-ethrc-deploy](https://github.com/shafeef901/hack26-ethrc-deploy)) watches this directory and plays motions on the robot.

### Direct API

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

Each output frame has 36 values: root position xyz (3), root quaternion wxyz (4), and 29 joint angles in radians. Set client timeout to at least 60 seconds.

## Development

```bash
make          # uv sync + ruff + basedpyright
make lint     # ruff + basedpyright only
```
