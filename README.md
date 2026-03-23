# DreamMotion

LLM agent that generates Unitree G1 humanoid robot motions from natural language. Speak or type what you want the robot to do, and DreamMotion plans the motion, generates it with a diffusion model, and deploys it to simulation or real hardware.

> [!NOTE]
> Built in 48 hours at HACK26, a hackathon organized by [ETH Robotics Club](https://www.ethrobotics.ch/).

```mermaid
graph LR
    subgraph Input
        MIC[Microphone] --> WEB[Web UI]
        TEXT[Text] --> CLI[Agent CLI]
    end

    subgraph Agent
        WEB --> AGENT[Strands Agent<br/>GPT-4.1]
        CLI --> AGENT
        AGENT --> REFINER[Prompt Refiner<br/>GPT-4.1]
        REFINER --> KIMODO[Kimodo<br/>Motion Diffusion]
    end

    KIMODO --> PT[.pt files<br/>output/]

    subgraph Deploy
        PT --> WATCHER[File Watcher]
        WATCHER --> MIMIC[Motion Tracking<br/>BeyondMimic]
        LOCO[Locomotion<br/>AMO] <--> MIMIC
    end

    LOCO --> ROBOT[MuJoCo Sim<br/>or Real G1]
    MIMIC --> ROBOT
```

## Quick start

```bash
git clone --recurse-submodules https://github.com/maedmatt/DreamMotion.git
cd DreamMotion
uv sync
cp .env.example .env   # fill in your keys
```

Set `OPENAI_API_KEY` and `KIMODO_URL` in `.env` (see [Kimodo server setup](kimodo-server/)).

## Dependencies by component

DreamMotion is modular — you only need the dependencies for the parts you use.

**Agent + Web UI** (motion generation from text/voice):
- Python 3.11+, [uv](https://docs.astral.sh/uv/)
- `OPENAI_API_KEY` — powers the agent (GPT-4.1), prompt refinement, speech-to-text, and TTS
- A running [Kimodo server](kimodo-server/) — NVIDIA GPU with 17GB+ VRAM, Docker

**Sim2Sim deployment** (play motions in MuJoCo):
- [RoboJuDo](https://github.com/HansZ8/RoboJuDo) cloned as a sibling directory: `git clone https://github.com/HansZ8/RoboJuDo ../RoboJuDo`
- Install with: `uv sync --extra deploy`

**Sim2Real deployment** (physical Unitree G1):
- Everything from Sim2Sim, plus:
- [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python): `git clone https://github.com/unitreerobotics/unitree_sdk2_python ../unitree_sdk2_python`
- [CycloneDDS](https://cyclonedds.io/) installed on the system
- Set `UNITREE_NETWORK_INTERFACE` in `.env` to your Ethernet adapter connected to the G1 (find with `ip link` or `ifconfig`)
- Install with: `uv sync --extra deploy --extra tts`

## Safety

> **Warning:** Always preview generated motions in MuJoCo simulation before deploying to a physical robot. Diffusion-generated motions can produce unexpected poses or joint configurations that may damage hardware. Use `uv run deploy` (sim2sim) to verify the motion looks reasonable before running with `--config g1_agent_locomimic_real`.

## Entry points

| Command | Description | Docs |
|---|---|---|
| `uv run web` | Web UI with voice input, 3D preview, and TTS | [docs/web-ui.md](docs/web-ui.md) |
| `uv run agent` | Text CLI for motion generation | - |
| `uv run deploy` | Robot control loop — watches `output/` and plays motions | [docs/sim2sim.md](docs/sim2sim.md), [docs/sim2real.md](docs/sim2real.md) |

## Project structure

```
kimodo-server/                       # Docker-based Kimodo motion generation API
  setup.sh                         # one-command setup
  docker-compose.yaml              # text-encoder + API services
  api/                             # FastAPI server, constraint helpers, csv→pt converter
src/
  main.py                          # agent CLI entry point
  agent/
    agent.py                       # Strands agent, system prompt
    prompt_refiner.py              # GPT-4.1 prompt optimization
    tools/
      generate_motion.py           # @tool — calls Kimodo, saves .pt files
  web/
    server.py                      # FastAPI web UI server
    agent_runner.py                # web-specific agent wrapper with TTS
  deploy/
    run.py                         # deploy entry point (tyro CLI)
    agent_tracker_policy.py        # BeyondMimic tracker with blend edges
    configs.py                     # LocoMimic configs (sim + real)
  g1/
    audio/                         # Unitree robot speaker (TTS)
    speech_input/                  # Microphone recording + transcription
```

## Docs

- [Kimodo server](kimodo-server/) — Docker-based motion generation backend
- [Web UI](docs/web-ui.md) — voice/text interface with 3D preview
- [Sim2Sim deployment](docs/sim2sim.md) — playing motions in MuJoCo
- [Sim2Real deployment](docs/sim2real.md) — deploying to a physical Unitree G1
- [Development](docs/development.md) — linting, type checking, contributing

## Credits

- [Kimodo](https://github.com/NVlabs/Kimodo) (NVIDIA) — motion diffusion model
- [RoboJuDo](https://github.com/HansZ8/RoboJuDo) — RL deploy pipeline with BeyondMimic tracking
- [Strands Agents](https://github.com/strands-agents/sdk-python) — LLM agent framework
- [g1_description](https://github.com/isri-aist/g1_description) — G1 URDF/meshes for 3D viewer
