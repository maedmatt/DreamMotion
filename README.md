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

4. Verify the install:

```bash
cd <path_to_repo>
uv run python - <<'PY'
import unitree_sdk2py
print("unitree_sdk2py import OK")
PY
```

5. Before using the speaker tool, set the robot-facing network interface:

```bash
export UNITREE_NETWORK_INTERFACE=<your_ethernet_interface>
```

Example:

```bash
export UNITREE_NETWORK_INTERFACE=enx806d97161839
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

Example:

```bash
export VOICE_INPUT_DEVICE=5
```

4. If microphone mode fails with `Invalid sample rate`, override the capture sample rate:

```bash
export VOICE_INPUT_SAMPLE_RATE=48000
```

If `48000` does not work, try `44100`. The code defaults to `16000`, but some ALSA devices only accept higher sample rates.

5. Run microphone mode:

```bash
OPENAI_API_KEY=sk-... \
UNITREE_NETWORK_INTERFACE=<your_ethernet_interface> \
VOICE_INPUT_DEVICE=<device_index> \
VOICE_INPUT_SAMPLE_RATE=48000 \
uv run agent --mic
```

### Kimodo tunnel

The Kimodo server runs on a remote GPU instance. Open an SSH tunnel to access it locally:

```bash
ssh -L 8420:localhost:8420 -L 7860:localhost:7860 <remote-host>
```

- `localhost:8420` — Kimodo API
- `localhost:7860` — Kimodo visualizer ([Viser](https://github.com/nerfstudio-project/viser))

Verify the connection:

```bash
curl http://localhost:8420/health
```

## Usage

### Agent (interactive)

Text mode:

```bash
OPENAI_API_KEY=sk-... uv run agent --text
```

Microphone mode:

```bash
OPENAI_API_KEY=sk-... UNITREE_NETWORK_INTERFACE=<your_ethernet_interface> uv run agent --mic
```

In microphone mode, press Enter to start recording, press Enter again to stop, and the transcribed speech is printed before it is sent to the agent.

Describe a motion and the agent calls Kimodo to generate it. Results are saved as CSV in `output/`.

### Direct API

```bash
curl -X POST http://localhost:8420/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A person walks forward", "duration": 2.0, "diffusion_steps": 50}'
```

Returns JSON with a `qpos` array — each frame is 36 values (root xyz, root quaternion wxyz, 29 joint angles).

## Development

```bash
make          # uv sync + ruff + basedpyright
make lint     # ruff + basedpyright only
```
