from __future__ import annotations

import atexit
import importlib
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True, slots=True)
class Intrinsics:
    """Pinhole camera intrinsics."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class CameraFrame:
    """A single aligned RGB + Depth capture."""

    color: np.ndarray  # (H, W, 3) uint8 BGR
    depth: np.ndarray  # (H, W) float32 meters
    intrinsics: Intrinsics


class OakCamera:
    """Luxonis OAK-D camera wrapper using DepthAI."""

    def __init__(self) -> None:
        dai = importlib.import_module("depthai")
        self._dai = dai

        pipeline = dai.Pipeline()

        # Color camera
        cam_rgb = pipeline.createColorCamera()
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)

        # Stereo depth
        mono_left = pipeline.createMonoCamera()
        mono_right = pipeline.createMonoCamera()
        mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

        stereo = pipeline.createStereoDepth()
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # Align depth to RGB
        stereo.setOutputSize(cam_rgb.getIspWidth(), cam_rgb.getIspHeight())
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        # XLink outputs
        xout_rgb = pipeline.createXLinkOut()
        xout_rgb.setStreamName("rgb")
        cam_rgb.isp.link(xout_rgb.input)

        xout_depth = pipeline.createXLinkOut()
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)

        # Start the device
        self._device = dai.Device(pipeline)
        self._q_rgb = self._device.getOutputQueue("rgb", maxSize=4, blocking=False)
        self._q_depth = self._device.getOutputQueue("depth", maxSize=4, blocking=False)

        # Cache intrinsics from the RGB camera
        calib = self._device.readCalibration()
        intrinsics_matrix = calib.getCameraIntrinsics(
            dai.CameraBoardSocket.CAM_A,
            cam_rgb.getIspWidth(),
            cam_rgb.getIspHeight(),
        )
        self._intrinsics = Intrinsics(
            fx=intrinsics_matrix[0][0],
            fy=intrinsics_matrix[1][1],
            cx=intrinsics_matrix[0][2],
            cy=intrinsics_matrix[1][2],
            width=cam_rgb.getIspWidth(),
            height=cam_rgb.getIspHeight(),
        )

        atexit.register(self.stop)

    @property
    def intrinsics(self) -> Intrinsics:
        return self._intrinsics

    def capture(self) -> CameraFrame:
        """Capture an aligned RGB + Depth frame.

        Returns:
            CameraFrame with color (H,W,3 uint8) and depth (H,W float32 meters).
        """
        rgb_packet = self._q_rgb.get()
        depth_packet = self._q_depth.get()

        color = rgb_packet.getCvFrame()
        # OAK-D depth is in millimeters (uint16), convert to meters
        depth_raw = depth_packet.getFrame()
        depth_m = depth_raw.astype(np.float32) / 1000.0

        return CameraFrame(
            color=color,
            depth=depth_m,
            intrinsics=self._intrinsics,
        )

    def deproject(self, u: int, v: int, depth_m: float) -> tuple[float, float, float]:
        """Deproject a pixel + depth to a 3D point in camera_link frame.

        Uses manual pinhole model:
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            Z = depth_m

        Args:
            u: Pixel column.
            v: Pixel row.
            depth_m: Depth in meters.

        Returns:
            (X, Y, Z) in camera optical frame (Z forward, X right, Y down).
        """
        intr = self._intrinsics
        z = depth_m
        x = (u - intr.cx) * z / intr.fx
        y = (v - intr.cy) * z / intr.fy
        return (x, y, z)

    def stop(self) -> None:
        """Stop the camera pipeline. Idempotent."""
        if hasattr(self, "_device") and self._device is not None:
            self._device.close()
            self._device = None  # type: ignore[assignment]


@lru_cache(maxsize=1)
def get_camera() -> OakCamera:
    """Return the singleton OakCamera instance."""
    return OakCamera()
