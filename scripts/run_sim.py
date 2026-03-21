# ruff: noqa: TC002
# pyright: reportMissingImports=false
"""Simulation loop that watches output/ for new ProtoMotions .pt files.

Runs the AgentTrackerPolicy in MuJoCo with AMO locomotion fallback.
When a new .pt file appears in the output directory, it auto-switches
to mimic mode and plays the motion. Between motions, the tracker holds
the default standing pose.

Keyboard controls (only when MuJoCo window is focused):
    L — switch to locomotion
    M — switch to mimic (tracker)

Usage (from the deploy repo's venv):
    python scripts/run_sim.py --watch-dir /path/to/g1-treasure-hunt/output
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import glfw
import tyro

logger = logging.getLogger("run_sim")

DEPLOY_REPO = Path(__file__).resolve().parent.parent.parent / "hack26-ethrc-deploy"
if DEPLOY_REPO.exists():
    sys.path.insert(0, str(DEPLOY_REPO))


@dataclass
class SimConfig:
    """G1 simulation with motion file watcher."""

    watch_dir: str = "output"
    """Directory to watch for new .pt files."""

    initial_motion: str | None = None
    """Optional .pt file to play on startup. If not set, uses latest in watch-dir."""

    check_interval: int = 50
    """Check for new files every N steps (default 50 = 1s at 50Hz)."""

    real: bool = False
    """Deploy on real robot instead of MuJoCo simulation."""


def find_latest_pt(watch_dir: Path, after_mtime: float) -> Path | None:
    """Find the newest .pt file modified after the given timestamp."""
    best: Path | None = None
    best_mtime = after_mtime
    for f in watch_dir.glob("*.pt"):
        mt = f.stat().st_mtime
        if mt > best_mtime:
            best = f
            best_mtime = mt
    return best


# Pending key commands from GLFW callback (only fires when window is focused)
_pending_commands: list[str] = []


def _setup_glfw_keys(viewer) -> None:
    """Hook into the MuJoCo viewer's GLFW window for key events."""
    window = viewer.window
    original_callback = (
        glfw.get_key_callback(window) if hasattr(glfw, "get_key_callback") else None
    )

    def key_callback(win, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_L:
                _pending_commands.append("loco")
                logger.info("Key: switch to locomotion")
            elif key == glfw.KEY_M:
                _pending_commands.append("mimic")
                logger.info("Key: switch to mimic")
        # Call the viewer's original key handler
        if original_callback is not None:
            original_callback(win, key, scancode, action, mods)

    glfw.set_key_callback(window, key_callback)


def main() -> None:
    cfg = tyro.cli(SimConfig)
    watch_dir = Path(cfg.watch_dir)
    watch_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("robojudo.pipeline.rl_pipeline").setLevel(logging.WARNING)

    initial_motion = cfg.initial_motion
    if initial_motion is None:
        latest = find_latest_pt(watch_dir, 0)
        if latest is None:
            logger.error(
                f"No .pt files in {watch_dir}/ and no --initial-motion given. "
                "Generate a motion first: uv run agent --no-tts --no-zmq"
            )
            sys.exit(1)
        initial_motion = str(latest)

    logger.info(f"Initial motion: {initial_motion}")

    import robojudo.pipeline
    from robojudo.config.config_manager import ConfigManager
    from robojudo.pipeline.rl_pipeline import RlPipeline

    config_name = "g1_agent_locomimic_real" if cfg.real else "g1_agent_locomimic"
    logger.info(f"Config: {config_name}")
    config_manager = ConfigManager(config_name=config_name)
    pipeline_cfg = config_manager.get_cfg()
    pipeline_cfg.mimic_policies[0].motion_path = initial_motion

    pipeline_class: type[RlPipeline] = getattr(
        robojudo.pipeline, pipeline_cfg.pipeline_type
    )
    pipeline = pipeline_class(cfg=pipeline_cfg)

    # Hook GLFW key callbacks for loco/mimic switching
    _setup_glfw_keys(pipeline.env.viewer)

    mimic_wrapper = pipeline.policy_manager.policy_by_id(
        pipeline.policy_manager.policy_mimic_ids[0]
    )
    inner_policy = mimic_wrapper.policy
    last_seen_mtime = time.time()

    pipeline._pending_blend_in = False

    logger.info(
        "Watching for new motions in %s/. "
        "Press L (loco) or M (mimic) in the MuJoCo window.",
        watch_dir,
    )

    try:
        while True:
            t0 = time.time()
            pipeline._pending_blend_in = False
            pipeline.step()

            # Process GLFW key commands
            while _pending_commands:
                cmd = _pending_commands.pop(0)
                if cmd == "loco" and pipeline.policy_locomotion_mimic_flag == 1:
                    pipeline.policy_manager.switch_to_loco()
                    pipeline.policy_locomotion_mimic_flag = 0
                elif cmd == "mimic" and pipeline.policy_locomotion_mimic_flag == 0:
                    pipeline.policy_manager.switch_to_mimic()
                    pipeline.policy_locomotion_mimic_flag = 1

            # Check for new motion files periodically
            if pipeline.timestep % cfg.check_interval == 0:
                new_file = find_latest_pt(watch_dir, last_seen_mtime)
                if new_file is not None:
                    last_seen_mtime = new_file.stat().st_mtime
                    try:
                        env_data = pipeline.env.get_data()
                        inner_policy.load_motion(str(new_file), env_data=env_data)
                        if pipeline.policy_locomotion_mimic_flag == 0:
                            pipeline.policy_manager.switch_to_mimic()
                            pipeline.policy_locomotion_mimic_flag = 1
                        logger.info("New motion: %s", new_file.name)
                    except Exception:
                        logger.warning(
                            "Failed to load %s", new_file.name, exc_info=True
                        )

            elapsed = time.time() - t0
            sleep_time = pipeline.dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        if hasattr(pipeline.env, "shutdown"):
            pipeline.env.shutdown()


if __name__ == "__main__":
    main()
