# ruff: noqa: RUF012
# pyright: reportMissingImports=false
"""RoboJuDo configs for the agent tracker pipeline.

Registers configs with robojudo's cfg_registry so they can be used
with ConfigManager(config_name=...).
"""

from __future__ import annotations

from robojudo.config import cfg_registry
from robojudo.config.g1.env.g1_mujuco_env_cfg import G1MujocoEnvCfg
from robojudo.config.g1.env.g1_real_env_cfg import G1RealEnvCfg, G1UnitreeCfg
from robojudo.config.g1.pipeline.g1_locomimic_pipeline_cfg import (
    G1RlLocoMimicPipelineCfg,
)
from robojudo.config.g1.policy.g1_amo_policy_cfg import G1AmoPolicyCfg
from robojudo.config.g1.policy.g1_protomotions_bm_tracker_cfg import (
    ProtoMotionsBMTrackerPolicyCfg,
)
from robojudo.controller.ctrl_cfgs import UnitreeCtrlCfg


class AgentTrackerPolicyCfg(ProtoMotionsBMTrackerPolicyCfg):
    """Config for AgentTrackerPolicy with file watching."""

    policy_type: str = "AgentTrackerPolicy"
    watch_dir: str = ""
    """Directory to watch for new .pt files. Empty = disabled."""
    watch_interval: int = 50
    """Check for new files every N steps (50 = 1s at 50Hz)."""


@cfg_registry.register
class g1_agent_locomimic(G1RlLocoMimicPipelineCfg):
    """Agent tracker with AMO locomotion fallback (simulation).

    Starts in locomotion. File watcher auto-switches to mimic when a
    new .pt file appears. MOTION_DONE switches back to loco.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg(
        born_place_align=False,
        random_heading=False,
    )
    ctrl: list = []
    loco_policy: G1AmoPolicyCfg = G1AmoPolicyCfg()
    mimic_policies: list[AgentTrackerPolicyCfg] = [
        AgentTrackerPolicyCfg(),
    ]


@cfg_registry.register
class g1_agent_locomimic_real(g1_agent_locomimic):
    """Agent tracker with AMO locomotion on real G1 hardware.

    A = emergency shutdown, Select = loco, Start = mimic.
    """

    env: G1RealEnvCfg = G1RealEnvCfg(
        unitree=G1UnitreeCfg(
            net_if="enx806d97162a37",
        ),
        born_place_align=False,
    )
    ctrl: list[UnitreeCtrlCfg] = [
        UnitreeCtrlCfg(
            combination_init_buttons=[],
            triggers={
                "A": "[SHUTDOWN]",
                "Select": "[POLICY_LOCO]",
                "Start": "[POLICY_MIMIC]",
            },
        ),
    ]
    do_safety_check: bool = True
