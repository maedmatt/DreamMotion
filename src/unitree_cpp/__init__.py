from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import List


def _candidate_build_dirs() -> list[Path]:
    env_dir = os.environ.get("UNITREE_CPP_BUILD_DIR")
    candidates: list[Path] = []
    if env_dir:
        candidates.append(Path(env_dir))

    repo_root = Path(__file__).resolve().parents[2]
    candidates.append(repo_root.parent / "RoboJuDo" / "packages" / "unitree_cpp" / "build")
    return candidates


def _find_extension() -> Path:
    for build_dir in _candidate_build_dirs():
        if not build_dir.is_dir():
            continue
        matches = sorted(build_dir.glob("unitree_cpp*.so"))
        if matches:
            return matches[0]
    searched = ", ".join(str(path) for path in _candidate_build_dirs())
    raise ModuleNotFoundError(
        "Could not find the local unitree_cpp extension. "
        f"Searched: {searched}"
    )


def _load_extension_module() -> object:
    module_name = __name__ + ".unitree_cpp"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    extension_path = _find_extension()
    spec = importlib.util.spec_from_file_location(module_name, extension_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load spec for {extension_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_ext = _load_extension_module()

UnitreeController = _ext.UnitreeController
_ImuState = _ext.ImuState
_MotorState = _ext.MotorState
_RobotState = _ext.RobotState
_SportState = _ext.SportState


class MotorState(_MotorState):
    @property
    def q(self) -> List[float]:
        return super().q

    @property
    def dq(self) -> List[float]:
        return super().dq

    @property
    def tau_est(self) -> List[float]:
        return super().tau_est


class ImuState(_ImuState):
    @property
    def rpy(self) -> List[float]:
        return super().rpy

    @property
    def quaternion(self) -> List[float]:
        return super().quaternion

    @property
    def gyroscope(self) -> List[float]:
        return super().gyroscope

    @property
    def accelerometer(self) -> List[float]:
        return super().accelerometer


class RobotState(_RobotState):
    @property
    def tick(self) -> int:
        return super().tick

    @property
    def motor_state(self) -> MotorState:
        return super().motor_state

    @property
    def imu_state(self) -> ImuState:
        return super().imu_state


class SportState(_SportState):
    @property
    def position(self) -> List[float]:
        return super().position

    @property
    def velocity(self) -> List[float]:
        return super().velocity
