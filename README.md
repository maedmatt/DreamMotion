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

4. If microphone mode fails with `Invalid sample rate`, override the capture sample rate:

```bash
export VOICE_INPUT_SAMPLE_RATE=48000
```

If `48000` does not work, try `44100`. The code defaults to `16000`, but some ALSA devices only accept higher sample rates.

### Kimodo

The Kimodo server runs on a remote GPU instance. Either use the ngrok URL or an SSH tunnel:

```bash
# Option 1: ngrok (set in .env)
KIMODO_URL=https://<ngrok-subdomain>.ngrok-free.app

# Option 2: SSH tunnel
ssh -L 8420:localhost:8420 -L 7860:localhost:7860 <remote-host>
```

- `localhost:7860` — Kimodo visualizer ([Viser](https://github.com/nerfstudio-project/viser)), requires SSH tunnel

Verify the connection:

```bash
curl http://localhost:8420/health
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
uv run agent --no-zmq           # disable ZMQ publishing
uv run agent --no-tts --no-zmq  # motion generation only
uv run agent --kimodo-url URL   # override Kimodo endpoint
```

In microphone mode, press Enter to start recording, press Enter again to stop.

The agent runs preflight checks at startup — it verifies the OpenAI key, Kimodo health, network interface (if TTS), and microphone (if mic mode). Missing dependencies are caught before the loop starts.

Results are saved as CSV and `.pt` in `output/`. Generated motions are published over ZMQ (`tcp://*:5555`) for downstream consumers (tracking policy, visualizer, etc.).

### ZMQ subscriber example

```python
import zmq, json
ctx = zmq.Context()
s = ctx.socket(zmq.SUB)
s.connect("tcp://localhost:5555")
s.subscribe(b"motion")
topic, meta, pt = s.recv_multipart()
print(json.loads(meta))  # {"prompt": "...", "duration": 3.0}
# torch.load(io.BytesIO(pt)) to get the tensor
```

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
