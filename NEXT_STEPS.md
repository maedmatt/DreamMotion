# Next Steps — G1 Treasure Hunt

## Current state

All code is written and lint-clean. Unit tests pass locally (no hardware).
The full LOOK → MOVE → LOOK_AGAIN → ACT pipeline is implemented and wired
into the Strands agent as a `treasure_hunt` tool. What is missing is the
ability to actually run it end-to-end, because the OAK-D camera is USB-connected
to the robot's onboard computer, not to the laptop where the agent runs.

---

## Step 1 — Commit current work

Commit everything that exists now with a single message:

```
add Look-Move-Look-Act pipeline with OAK-D vision and dual locomotion
```

Files to stage: all new/modified files under `src/`, `tests/`, `pyproject.toml`.

---

## Step 2 — Solve the camera architecture problem (BLOCKER)

**Problem**: The agent runs on the laptop. The OAK-D is plugged into the robot
via USB. The laptop cannot access the camera directly.

**Two options — pick one:**

### Option A — Run the agent on the robot's onboard computer (recommended)

- SSH into the robot's onboard PC
- Clone the repo and `uv sync` there (needs Python 3.11–3.13 — avoid 3.14)
- Run `UNITREE_NETWORK_INTERFACE=eth0 OPENAI_API_KEY=sk-... uv run g1` on the robot
- Kimodo is still on AWS; robot PC reaches it through the same SSH tunnel
- Nothing in the code changes — the robot PC has both the camera (USB) and
  network access to the robot SDK (localhost DDS)

### Option B — Camera streaming server on the robot

- Write a small server (`scripts/camera_server.py`) on the robot PC using
  `depthai` that serves JPEG frames + depth over a simple HTTP or ZMQ socket
- Replace `OakCamera` with a `RemoteCamera` client on the laptop side that
  fetches frames from that server (SSH-tunnel the port)
- More moving parts, higher latency, but lets you keep running on your laptop

**Recommendation**: Start with Option A. It requires zero code changes and is
the fastest path to a working demo.

---

## Step 3 — Hardware checkerboard calibration

Once the camera is accessible (Step 2 resolved):

1. Print an OpenCV checkerboard (e.g. 9×6, 25mm squares)
2. Hold it at various positions/angles in front of the OAK-D
3. Run the calibration script (to be written in `scripts/calibrate_camera.py`)
   using `cv2.calibrateCamera` + `cv2.solvePnP` to get the 4×4 extrinsic
   matrix from camera frame to robot base_link
4. Save the result as a JSON file (e.g. `config/camera_to_base.json`)
5. Update `src/g1/transforms/static.py` to load from that JSON instead of
   relying on env-var offsets

Until calibration is done, the default env-var values in `static.py` are a
reasonable starting point (`G1_CAMERA_Z_OFFSET=0.45`, `G1_CAMERA_PITCH=-0.35`).

---

## Step 4 — Hardware smoke tests (in order)

Run these with the robot powered on. Each one is self-contained and will tell
you exactly what is broken before you try the full pipeline.

```bash
# 1. Camera only (OAK-D plugged in, no robot SDK needed)
uv run python tests/hardware/test_camera.py

# 2. Locomotion + odometry (robot in sport mode, Ethernet connected)
UNITREE_NETWORK_INTERFACE=eth0 uv run python tests/hardware/test_locomotion.py

# 3. Transform calibration verification (robot + camera + known reference point)
UNITREE_NETWORK_INTERFACE=eth0 uv run python tests/hardware/test_calibration.py

# 4. Full LOOK state only (no movement, just detect and localize the object)
UNITREE_NETWORK_INTERFACE=eth0 OPENAI_API_KEY=sk-... \
    uv run python tests/hardware/test_e2e.py --state look

# 5. Full LOOK → MOVE → LOOK_AGAIN → ACT pipeline
UNITREE_NETWORK_INTERFACE=eth0 OPENAI_API_KEY=sk-... \
    uv run python tests/hardware/test_e2e.py --state full
```

Fix failures in order — don't skip ahead.

---

## Step 5 — Tune locomotion gains

After the first real walk-to-point attempt:

- If the robot overshoots or oscillates: reduce `Kp_dist` in `sdk_controller.py`
  (default 0.5). Try 0.3.
- If the robot drifts sideways instead of turning first: increase `Kp_yaw`
  (default 1.0). Try 1.5.
- If it stops too far from the object: reduce `stop_short_m` in the
  `treasure_hunt` tool call (default 0.5m). Try 0.3m.
- Check actual vs expected distance in `test_locomotion.py` output.

---

## Step 6 — ACT integration (Kimodo foot target)

The ACT state currently POSTs to Kimodo with a `foot_target_xyz` field that
Kimodo may or may not support yet. Verify with the Kimodo team:

- Does `/generate` accept `foot_target_xyz`?
- If not, the fallback is to send `root_target_xy` pointing to the object and
  let the motion diffusion model handle foot placement naturally.
- Update `machine.py` `_handle_act` accordingly once the API is confirmed.

---

## Step 7 — Agent integration test (full voice-to-action loop)

```bash
# On the machine that has camera + robot network access:
UNITREE_NETWORK_INTERFACE=eth0 OPENAI_API_KEY=sk-... uv run g1
# Then type: "find the red box and step on it"
```

The agent should call `treasure_hunt(target_object="red box")` and run the
full pipeline. If it calls `generate_motion` instead, update the system prompt
in `agent.py` to reinforce when to use `treasure_hunt`.

---

## Optional improvements (post-demo)

- **Camera streaming server** (Option B from Step 2) if you want to run the
  agent on the laptop long-term
- **Calibration script** (`scripts/calibrate_camera.py`) for repeatable
  camera mount calibration
- **KIMODO walk mode** end-to-end test (currently only SDK walk tested on hardware)
- **Multi-object support**: let the user say "find the blue ball" vs "find the box"
  and have the detector handle multiple class prompts simultaneously
