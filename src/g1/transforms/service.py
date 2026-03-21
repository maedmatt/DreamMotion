from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from g1.transforms.math import compose, invert, transform_point
from g1.transforms.static import camera_to_base

if TYPE_CHECKING:
    from collections.abc import Sequence


class TransformService:
    """Compose static and dynamic transforms between coordinate frames.

    Supported frames:
      - ``"camera"``  — camera optical frame (Z forward, X right, Y down)
      - ``"base"``    — robot base_link (X forward, Y left, Z up)
      - ``"world"``   — odom frame (global)

    Transform chains:
      camera -> base  : static (camera_to_base, from calibration)
      camera -> world : camera_to_base @ base_to_odom
      base -> world   : base_to_odom (dynamic, from odometry)
    """

    _VALID_FRAMES: ClassVar[set[str]] = {"camera", "base", "world"}

    def __init__(self) -> None:
        self._T_camera_to_base = camera_to_base()
        self._odometry: object | None = None  # Lazy init

    def _get_odometry(self) -> object:
        """Lazy-load OdometrySubscriber (avoids requiring SDK for cam->base)."""
        if self._odometry is None:
            from g1.transforms.odometry import get_odometry

            self._odometry = get_odometry()
        return self._odometry

    def get_transform(self, from_frame: str, to_frame: str) -> np.ndarray:
        """Get the 4x4 transform matrix from one frame to another.

        Args:
            from_frame: Source frame name.
            to_frame: Target frame name.

        Returns:
            4x4 numpy array.
        """
        if from_frame not in self._VALID_FRAMES or to_frame not in self._VALID_FRAMES:
            msg = f"Unknown frame. Valid: {self._VALID_FRAMES}"
            raise ValueError(msg)

        if from_frame == to_frame:
            return np.eye(4)

        if from_frame == "camera" and to_frame == "base":
            return self._T_camera_to_base

        if from_frame == "base" and to_frame == "camera":
            return invert(self._T_camera_to_base)

        odom = self._get_odometry()
        T_base_to_odom = odom.base_to_odom()  # type: ignore[attr-defined]

        if from_frame == "base" and to_frame == "world":
            return T_base_to_odom

        if from_frame == "world" and to_frame == "base":
            return invert(T_base_to_odom)

        if from_frame == "camera" and to_frame == "world":
            return compose(self._T_camera_to_base, T_base_to_odom)

        # from_frame == "world" and to_frame == "camera"
        return compose(invert(T_base_to_odom), invert(self._T_camera_to_base))

    def transform_point_between(
        self,
        point: Sequence[float],
        from_frame: str,
        to_frame: str,
    ) -> np.ndarray:
        """Transform a 3D point between coordinate frames.

        Args:
            point: [x, y, z] in the source frame.
            from_frame: Source frame name.
            to_frame: Target frame name.

        Returns:
            numpy array [x, y, z] in the target frame.
        """
        T = self.get_transform(from_frame, to_frame)
        return transform_point(T, point)


@lru_cache(maxsize=1)
def get_transform_service() -> TransformService:
    """Return the singleton TransformService."""
    return TransformService()
