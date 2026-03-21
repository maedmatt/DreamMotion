from __future__ import annotations

import importlib
import os

_initialized = False


def get_network_interface() -> str:
    """Read the robot-facing network interface from env."""
    iface = os.environ.get("UNITREE_NETWORK_INTERFACE")
    if not iface:
        raise RuntimeError(
            "Set UNITREE_NETWORK_INTERFACE to the robot-facing network interface "
            "before using Unitree SDK features, for example `eth0`."
        )
    return iface


def ensure_channel_initialized(network_interface: str | None = None) -> None:
    """Initialize Unitree DDS channel factory (idempotent).

    Safe to call from multiple modules (audio, odometry, locomotion).
    Only the first call actually initializes; subsequent calls are no-ops.

    Args:
        network_interface: Override interface name. If None, reads from
            UNITREE_NETWORK_INTERFACE env var.
    """
    global _initialized
    if _initialized:
        return

    if network_interface is None:
        network_interface = get_network_interface()

    channel_module = importlib.import_module("unitree_sdk2py.core.channel")
    channel_module.ChannelFactoryInitialize(0, network_interface)
    _initialized = True
