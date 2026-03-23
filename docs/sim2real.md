# Sim2Real deployment (physical G1)

Deploy generated motions to a physical Unitree G1 humanoid robot. This extends the [sim2sim](sim2sim.md) setup with real hardware communication and safety controls.

## Additional setup

### Unitree SDK2 Python

Clone the Unitree SDK2 Python package next to the DreamMotion directory:

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python ../unitree_sdk2_python
```

### CycloneDDS

The Unitree SDK2 communicates with the robot via DDS. Install CycloneDDS:

```bash
# macOS
brew install eclipse-cyclonedds

# Ubuntu
sudo apt install ros-humble-cyclonedds  # or build from source
```

### Install extras

```bash
uv sync --extra deploy --extra tts
```

### Network interface

The robot communicates over a dedicated Ethernet connection. Set the network interface in `.env`:

```
UNITREE_NETWORK_INTERFACE=eth0
```

Find your interface name with `ip link` (Linux) or `ifconfig` (macOS). It's typically a USB-Ethernet adapter connected directly to the G1.

## Running

```bash
uv run deploy --config g1_agent_locomimic_real --motion-path output/some_motion.pt
```

Or watch `output/` for new motions:

```bash
uv run deploy --config g1_agent_locomimic_real
```

## Gamepad controls

The real robot config registers a Unitree gamepad controller:

| Button | Action |
|---|---|
| **A** | Emergency shutdown |
| **Select** | Switch to locomotion (AMO walking) |
| **Start** | Switch to motion tracking (BeyondMimic) |

The file watcher auto-switches to motion tracking when a new `.pt` file appears, just like in simulation.

## Robot TTS

The G1 has a built-in speaker. The web UI can route spoken replies to the robot instead of the browser:

1. Set `UNITREE_NETWORK_INTERFACE` in `.env`
2. In the web UI settings, change "Speech Output" to "Play on Robot"

The robot uses Unitree's built-in TTS engine (`speaker_id=1` for English). You can also test it directly:

```bash
uv run robot-audio-test
```

## Safety

- The real config enables `do_safety_check: bool = True`
- The **A button** is always mapped to emergency shutdown
- The control loop exits if frame drops exceed 200ms, preventing runaway behavior
- The deploy pipeline starts in locomotion mode (AMO) — you must explicitly switch to motion tracking or let the file watcher do it
- Generated motions include a return-to-standing constraint by default, so the robot ends in a stable pose
