# G1 Treasure Hunt

LLM agent that generates Unitree G1 humanoid robot motions from natural language. Uses [Kimodo](https://github.com/NVIDIA/Kimodo) for motion generation via a Strands agent with OpenAI.

## Setup

```bash
uv sync
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

```bash
OPENAI_API_KEY=sk-... uv run agent
```

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
