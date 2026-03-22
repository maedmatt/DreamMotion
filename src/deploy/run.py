# ruff: noqa: TC002
# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
"""Deploy pipeline entry point with file watcher.

Runs the LocoMimic pipeline with AgentTrackerPolicy. Watches a
directory for new .pt motion files and auto-switches between
locomotion and motion tracking.

Usage:
    uv run deploy
    uv run deploy --watch-dir output
    uv run deploy --config g1_agent_locomimic_real --watch-dir output
"""

from __future__ import annotations

# Fix OMP performance issue on ARM platform (Jetson)
import inspect
import os
import platform

if platform.machine().startswith("aarch64"):
    os.environ["OMP_NUM_THREADS"] = "1"

import logging
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

import numpy as np
import tyro

logger = logging.getLogger("robojudo")


def _enable_mujoco_viewer_compat() -> None:
    """Adapt stock mujoco-python-viewer to RoboJuDo's patched call signature."""
    try:
        import mujoco_viewer
    except Exception:
        return

    init = mujoco_viewer.MujocoViewer.__init__
    if getattr(init, "_g1_compat_patched", False):
        return
    if "diable_key_callbacks" in inspect.signature(init).parameters:
        return

    def compat_init(self, *args, diable_key_callbacks=None, **kwargs):
        del diable_key_callbacks
        return init(self, *args, **kwargs)

    compat_init._g1_compat_patched = True  # type: ignore[attr-defined]
    mujoco_viewer.MujocoViewer.__init__ = compat_init


@dataclass
class DeployConfig:
    """G1 deploy pipeline with motion file watcher."""

    config: str = "g1_agent_locomimic"
    """Config class name."""

    onnx_path: str | None = None
    """ONNX policy file (overrides config)."""

    motion_path: str | None = None
    """Initial motion .pt file (overrides config)."""

    motion_index: int | None = None
    """Motion clip index within a multi-motion .pt library."""

    watch_dir: str = "output"
    """Directory to watch for new .pt files."""

    watch_interval: int = 50
    """Check for new files every N steps (50 = 1s at 50Hz)."""

    simulate_deploy: bool = False
    """Run prepare phase even in simulation."""

    prepare_seconds: float | None = None
    """Duration of each prepare phase in seconds."""

    hold_seconds: float | None = None
    """Exit after motion ends + this many seconds."""


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


def patch_mujoco_viewer_compat() -> None:
    """Accept RoboJuDo's viewer kwarg typo across mujoco_viewer versions."""
    try:
        import inspect
        import mujoco_viewer
    except Exception:
        logger.debug("Skipping mujoco_viewer compatibility patch", exc_info=True)
        return

    viewer_class = mujoco_viewer.MujocoViewer
    if getattr(viewer_class, "_robojudo_key_callback_compat", False):
        return

    init = viewer_class.__init__
    try:
        parameters = inspect.signature(init).parameters
    except (TypeError, ValueError):
        return

    if "diable_key_callbacks" in parameters:
        viewer_class._robojudo_key_callback_compat = True
        return

    accepts_disable = "disable_key_callbacks" in parameters

    @wraps(init)
    def compat_init(self, *args, **kwargs):
        if "diable_key_callbacks" in kwargs:
            value = kwargs.pop("diable_key_callbacks")
            if accepts_disable and "disable_key_callbacks" not in kwargs:
                kwargs["disable_key_callbacks"] = value
        return init(self, *args, **kwargs)

    viewer_class.__init__ = compat_init
    viewer_class._robojudo_key_callback_compat = True
    logger.info("Applied mujoco_viewer compatibility patch for RoboJuDo")


def main() -> None:
    cfg = tyro.cli(DeployConfig)
    _enable_mujoco_viewer_compat()

    patch_mujoco_viewer_compat()

    # Import our policy and configs to trigger @register decorators
    import robojudo.pipeline
    from robojudo.config.config_manager import ConfigManager
    from robojudo.pipeline.rl_pipeline import RlPipeline

    import deploy.agent_tracker_policy
    import deploy.configs  # noqa: F401

    logger.info(f"Using config: {cfg.config}")
    config_manager = ConfigManager(config_name=cfg.config)
    pipeline_cfg = config_manager.get_cfg()

    # Override paths from CLI
    if hasattr(pipeline_cfg, "policy"):
        if cfg.onnx_path is not None:
            pipeline_cfg.policy.onnx_path = cfg.onnx_path
        if cfg.motion_path is not None:
            pipeline_cfg.policy.motion_path = cfg.motion_path
        if cfg.motion_index is not None:
            pipeline_cfg.policy.motion_index = cfg.motion_index

    if hasattr(pipeline_cfg, "mimic_policies"):
        for mp in pipeline_cfg.mimic_policies:
            if cfg.onnx_path is not None:
                mp.onnx_path = cfg.onnx_path
            if cfg.motion_path is not None:
                mp.motion_path = cfg.motion_path
            if cfg.motion_index is not None:
                mp.motion_index = cfg.motion_index

    # Build pipeline
    pipeline_class: type[RlPipeline] = getattr(
        robojudo.pipeline, pipeline_cfg.pipeline_type
    )
    pipeline = pipeline_class(cfg=pipeline_cfg)

    if not pipeline_cfg.env.is_sim or cfg.simulate_deploy:
        pipeline.prepare(prepare_seconds=cfg.prepare_seconds)

    # Get the mimic policy for motion loading
    mimic_policy = None
    if hasattr(pipeline, "policy_manager"):
        mimic_id = pipeline.policy_manager.policy_mimic_ids[0]
        mimic_policy = pipeline.policy_manager.policy_by_id(mimic_id).policy

    # File watcher state
    watch_dir = Path(cfg.watch_dir)
    watch_dir.mkdir(parents=True, exist_ok=True)
    last_seen_mtime = time.time()
    step_count = 0

    logger.info(f"Watching for motions in {watch_dir}/")

    # Main loop
    while True:
        t0 = time.time()
        pipeline.step()
        step_count += 1

        # File watcher — runs regardless of which policy is active
        if step_count % cfg.watch_interval == 0:
            new_file = find_latest_pt(watch_dir, last_seen_mtime)
            if new_file is not None:
                last_seen_mtime = new_file.stat().st_mtime
                try:
                    # Load new motion with blend edges
                    if mimic_policy is not None:
                        env_data = pipeline.env.get_data()
                        current_dof = np.asarray(env_data.dof_pos, dtype=np.float32)
                        mimic_policy.load_motion(
                            str(new_file),
                            current_dof_pos=current_dof,
                        )

                    # Switch to mimic if currently in loco
                    if (
                        hasattr(pipeline, "policy_locomotion_mimic_flag")
                        and pipeline.policy_locomotion_mimic_flag == 0
                    ):
                        pipeline.policy_manager.switch_to_mimic()
                        pipeline.policy_locomotion_mimic_flag = 1

                    logger.info(f"[Deploy] New motion: {new_file.name}")
                except Exception:
                    logger.warning(
                        "[Deploy] Failed to load %s",
                        new_file.name,
                        exc_info=True,
                    )

        # Maintain target frequency
        if not pipeline_cfg.run_fullspeed:
            elapsed = time.time() - t0
            sleep_time = pipeline.dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif not pipeline_cfg.env.is_sim and elapsed - pipeline.dt > 0.2:
                logger.critical("Exiting due to excessive frame drop")
                pipeline.env.shutdown()
                time.sleep(10)
                break


if __name__ == "__main__":
    main()
