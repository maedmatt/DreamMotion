# csv_to_motionlib.py — Integration Guide

## What it does

Single-file converter: **Kimodo G1 CSV → ProtoMotions `.pt` MotionLib**.

Internally performs: CSV parsing → Forward Kinematics → velocity estimation → contact detection → height correction → MotionLib packing. No dependency on the ProtoMotions codebase — all logic (rotation math, FK, MotionLib format) is self-contained.

## Dependencies

```
torch
numpy
```

No `mujoco`, `dm_control`, `scipy`, or `easydict` required. The G1 kinematic model (33 bodies, 29 DOFs) is embedded as JSON.

## CLI Usage

```bash
# Single CSV file
python csv_to_motionlib.py --input output.csv --output motions.pt

# Directory of CSV files (all *.csv will be packed into one .pt)
python csv_to_motionlib.py --input ./csv_dir/ --output motions.pt

# Custom FPS (default: 30)
python csv_to_motionlib.py --input output.csv --output motions.pt --fps 60

# With post-filter: smoothing + speed change
python csv_to_motionlib.py --input output.csv --output motions.pt --smooth 3 --speed 0.8
```

### Arguments

| Argument   | Required | Default      | Description                                                |
|------------|----------|--------------|------------------------------------------------------------|
| `--input`  | Yes      | —            | Path to a `.csv` file or a directory                       |
| `--output` | No       | `motions.pt` | Output `.pt` file path                                     |
| `--fps`    | No       | `30`         | Frame rate of the input CSV                                |
| `--smooth` | No       | `1`          | Gaussian smoothing sigma in frames (0=off, typical: 1~5)   |
| `--speed`  | No       | `1.0`        | Speed factor (0.5=half-speed/2x frames, 2.0=double-speed)  |

### Post-filter details

Filters are applied to the raw **qpos** (root position, root quaternion, joint angles) **before** forward kinematics, so the output is always kinematically consistent.

**Smoothing** (`--smooth`):
- Gaussian kernel with sigma = N frames, kernel size = 2*ceil(3N)+1
- Root quaternion is re-normalized after linear-space smoothing
- `--smooth 1` = subtle jitter removal, `--smooth 5` = heavy smoothing

**Speed** (`--speed`):
- Root position and joint angles use linear interpolation
- Root quaternion uses spherical linear interpolation (slerp)
- `--speed 0.5` = half-speed (doubles frame count), `--speed 2.0` = double-speed (halves frames)
- Output FPS remains unchanged; only frame count changes

Processing order: **retime → smooth → FK → velocities → contacts**

### Exit codes

- `0` — success
- `1` — no CSV files found / value error (column count mismatch)

## Python API

For programmatic integration (e.g. calling from another script or post-processing pipeline):

```python
from csv_to_motionlib import convert_csv_to_motion, _pack_motionlib
from csv_to_motionlib import smooth_qpos, retime_qpos  # standalone filter functions
import torch

# Convert a single CSV → motion dict (in-memory, no file I/O)
motion = convert_csv_to_motion("output.csv", fps=30, device="cpu",
                                smooth=2.0, speed=0.8)

# motion is a plain dict with these keys:
#   rigid_body_pos:      Tensor [N, 33, 3]     world positions
#   rigid_body_rot:      Tensor [N, 33, 4]     world quaternions (xyzw)
#   rigid_body_vel:      Tensor [N, 33, 3]     linear velocities
#   rigid_body_ang_vel:  Tensor [N, 33, 3]     angular velocities
#   dof_pos:             Tensor [N, 29]        joint angles (rad)
#   dof_vel:             Tensor [N, 29]        joint velocities
#   rigid_body_contacts: Tensor [N, 33]        bool contact labels
#   fps:                 float

# Pack one or more motion dicts into a MotionLib .pt
lib = _pack_motionlib([motion], ["output.csv"])
torch.save(lib, "motions.pt")

# ── Standalone filter functions (can be used independently) ──
# qpos shape: (T, 36) = [root_pos(3), root_quat_wxyz(4), joint_angles(29)]

# Retime: change speed (slerp for quaternions, lerp for positions/angles)
qpos_slow = retime_qpos(qpos, speed=0.5)   # half-speed, 2x frames
qpos_fast = retime_qpos(qpos, speed=2.0)   # double-speed, 0.5x frames

# Smooth: Gaussian filter (auto re-normalizes root quaternion)
qpos_smooth = smooth_qpos(qpos, sigma=3.0) # sigma=3 frames
```

## Input CSV Format (Kimodo G1)

- **No header row**, no frame index column
- Each row = one frame, comma-separated
- 36 columns per row:

```
col  0..2   root position (x, y, z) in meters
col  3..6   root quaternion (w, x, y, z)
col  7..35  29 joint angles in radians
```

Joint order (matches Unitree G1 MJCF):

```
left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee,
left_ankle_pitch, left_ankle_roll,
right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee,
right_ankle_pitch, right_ankle_roll,
waist_yaw, waist_roll, waist_pitch,
left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw,
left_elbow, left_wrist_roll, left_wrist_pitch, left_wrist_yaw,
right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw,
right_elbow, right_wrist_roll, right_wrist_pitch, right_wrist_yaw
```

## Output `.pt` Format (MotionLib)

The output is a `torch.save()`-d dict loadable via `torch.load()`. Key fields:

| Key                 | Shape                | Description                          |
|---------------------|----------------------|--------------------------------------|
| `gts`               | `[T_total, 33, 3]`  | Rigid body world positions           |
| `grs`               | `[T_total, 33, 4]`  | Rigid body world quaternions (xyzw)  |
| `gvs`               | `[T_total, 33, 3]`  | Rigid body linear velocities         |
| `gavs`              | `[T_total, 33, 3]`  | Rigid body angular velocities        |
| `dps`               | `[T_total, 29]`     | Joint positions (rad)                |
| `dvs`               | `[T_total, 29]`     | Joint velocities                     |
| `contacts`          | `[T_total, 33]`     | Contact labels (bool)                |
| `motion_num_frames` | `[M]`               | Frame count per motion clip          |
| `motion_lengths`    | `[M]`               | Duration (seconds) per clip          |
| `motion_dt`         | `[M]`               | 1/fps per clip                       |
| `motion_weights`    | `[M]`               | Sampling weights (all 1.0)           |
| `length_starts`     | `[M]`               | Cumulative frame offsets             |
| `motion_files`      | `tuple[str]`        | Source file paths                    |

`T_total` = sum of all frames across `M` motion clips.

This `.pt` file is directly consumable by `ProtoMotions.MotionLib`:

```python
from protomotions.components.motion_lib import MotionLib, MotionLibConfig
ml = MotionLib(MotionLibConfig(motion_file="motions.pt"), device="cuda")
```

## Typical Post-Processing Integration

```python
import subprocess, sys

csv_path = "/path/to/kimodo_output.csv"
pt_path  = "/path/to/output/motions.pt"

subprocess.run([
    sys.executable, "csv_to_motionlib.py",
    "--input", csv_path,
    "--output", pt_path,
    "--fps", "30",
], check=True)
```
