# Web UI

The web UI provides a browser-based interface for generating robot motions through voice or text, with a 3D preview of the generated trajectories.

## Setup

```bash
uv sync
```

Required environment variables in `.env`:

```
OPENAI_API_KEY=sk-...        # for agent, prompt refinement, STT, and TTS
KIMODO_URL=http://localhost:8420
```

## Running

```bash
uv run web
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--port` | 8000 | Web server port |
| `--kimodo-url` | `KIMODO_URL` env | Override Kimodo server URL |
| `--diffusion-steps` | 50 | Kimodo diffusion steps (50 = fast, 100 = quality) |
| `--transcribe-model` | `gpt-4o-transcribe` | OpenAI speech-to-text model |

The server runs a preflight check on startup, verifying the OpenAI API key and Kimodo connectivity. It opens the browser automatically.

## Features

### Voice input

Click the microphone button or press and hold to record. Audio is transcribed via OpenAI Whisper (`gpt-4o-transcribe`) and sent to the agent.

### Text input

Type a motion description in the chat input and press enter.

### 3D preview

Generated motions are displayed in an embedded 3D viewer using the G1 URDF mesh. The viewer loads CSV motion data and animates the robot skeleton.

Requires the `g1_description` submodule (cloned with `--recurse-submodules`).

### Text-to-speech

The agent generates spoken responses on every turn. Two output targets:

- **Web** (default): OpenAI TTS (`tts-1` model) plays in the browser
- **Robot**: Unitree SDK2 built-in TTS plays on the G1's speaker (see [sim2real](sim2real.md))

### Selection mode

Enable in settings to generate 3 motion candidates per prompt. Each candidate gets its own 3D preview. Select the best one to save it to `output/` for deployment.

### Presentation mode

A minimal UI overlay for demos, showing a step-by-step pipeline visualization (Understand -> Generate) with the refined prompt.
