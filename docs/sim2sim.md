# Sim2Sim deployment (MuJoCo)

Play generated motions on a simulated Unitree G1 in MuJoCo. The deploy pipeline uses [RoboJuDo](https://github.com/HansZ8/RoboJuDo) for the RL control loop and BeyondMimic for motion tracking.

## Setup

Clone RoboJuDo next to the DreamMotion directory:

```bash
git clone https://github.com/HansZ8/RoboJuDo ../RoboJuDo
```

Install with the deploy extra:

```bash
uv sync --extra deploy
```

This installs `robojudo` as an editable dependency from `../RoboJuDo`, along with `mujoco-python-viewer`.

### ONNX tracker model

Place the BeyondMimic tracker model in:

```
../RoboJuDo/assets/models/g1/protomotions_bm_tracker/
  unified_pipeline.onnx
  unified_pipeline.yaml
```

This ONNX model is the trained neural network policy that maps reference motions to G1 motor commands.

## Running

Play a specific motion file:

```bash
uv run deploy --motion-path output/some_motion.pt
```

Watch `output/` for new motions (used with the web UI or agent):

```bash
uv run deploy
```

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--config` | `g1_agent_locomimic` | Config class name |
| `--motion-path` | - | Initial `.pt` motion file |
| `--motion-index` | - | Clip index within a multi-motion `.pt` library |
| `--watch-dir` | `output` | Directory to watch for new `.pt` files |
| `--watch-interval` | 50 | Check for new files every N steps (50 = 1s at 50Hz) |
| `--onnx-path` | - | Override ONNX policy path from config |
| `--simulate-deploy` | `False` | Run prepare phase even in simulation |
| `--prepare-seconds` | - | Duration of each prepare phase in seconds |
| `--hold-seconds` | - | Exit after motion ends + this many seconds |

## How it works

The deploy pipeline runs a 50Hz control loop with two policies:

1. **AMO** (locomotion): a learned walking policy that keeps the robot balanced
2. **BeyondMimic** (motion tracking): tracks reference motions from `.pt` files using the ONNX policy

A file watcher checks `output/` every second. When a new `.pt` file appears:

1. The motion is loaded into `AgentTrackerPolicy`
2. The motion is zero-aligned (xy position and yaw at frame 0)
3. The pipeline switches from locomotion to motion tracking
4. After the motion completes, it switches back to locomotion

### Motion file format

The `.pt` files are ProtoMotions MotionLib dictionaries containing:

- Rigid body positions, rotations, velocities, angular velocities (T x 33 bodies)
- Joint positions and velocities (T x 29 DOFs)
- Contact labels (T x 33 bodies)
- Metadata: frame count, duration, dt, file names
