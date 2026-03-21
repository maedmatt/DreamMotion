from __future__ import annotations

import atexit
import importlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

_CALIB_FILE = Path(__file__).parent.parent.parent.parent / "config" / "calibration.json"


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
        self._pipeline = None
        self._device = None

        if hasattr(dai.Pipeline, "createColorCamera"):
            self._init_legacy_pipeline()
        else:
            self._init_modern_pipeline()

        atexit.register(self.stop)

    def _build_camera_nodes(self, pipeline):
        dai = self._dai
        if hasattr(dai.Pipeline, "createColorCamera"):
            cam_rgb = pipeline.createColorCamera()
            mono_left = pipeline.createMonoCamera()
            mono_right = pipeline.createMonoCamera()
            stereo = pipeline.createStereoDepth()
        else:
            cam_rgb = pipeline.create(dai.node.ColorCamera)
            mono_left = pipeline.create(dai.node.MonoCamera)
            mono_right = pipeline.create(dai.node.MonoCamera)
            stereo = pipeline.create(dai.node.StereoDepth)

        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)

        mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

        preset_mode = dai.node.StereoDepth.PresetMode
        for preset_name in ("HIGH_DENSITY", "DENSITY", "FAST_DENSITY", "DEFAULT"):
            preset = getattr(preset_mode, preset_name, None)
            if preset is not None:
                stereo.setDefaultProfilePreset(preset)
                break
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setOutputSize(cam_rgb.getIspWidth(), cam_rgb.getIspHeight())
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)
        return cam_rgb, stereo

    def _cache_intrinsics(self, cam_rgb) -> None:
        # Prefer calibration file (OpenCV checkerboard result) over device values
        if _CALIB_FILE.exists():
            with _CALIB_FILE.open() as f:
                data = json.load(f)
            m = data["camera_matrix"]
            size = data["image_size"]
            self._intrinsics = Intrinsics(
                fx=m[0][0],
                fy=m[1][1],
                cx=m[0][2],
                cy=m[1][2],
                width=size["width"],
                height=size["height"],
            )
            return

        dai = self._dai
        assert self._device is not None
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

    def _init_legacy_pipeline(self) -> None:
        dai = self._dai
        pipeline = dai.Pipeline()
        cam_rgb, stereo = self._build_camera_nodes(pipeline)

        xout_rgb = pipeline.createXLinkOut()
        xout_rgb.setStreamName("rgb")
        cam_rgb.isp.link(xout_rgb.input)

        xout_depth = pipeline.createXLinkOut()
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)

        self._device = dai.Device(pipeline)
        self._q_rgb = self._device.getOutputQueue("rgb", maxSize=4, blocking=False)
        self._q_depth = self._device.getOutputQueue("depth", maxSize=4, blocking=False)
        self._cache_intrinsics(cam_rgb)

    def _init_modern_pipeline(self) -> None:
        import time

        dai = self._dai
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                device = dai.Device()
                break
            except RuntimeError as exc:
                last_exc = exc
                time.sleep(2.0)
        else:
            raise RuntimeError(f"OAK-D failed to open after 3 attempts: {last_exc}") from last_exc

        pipeline = dai.Pipeline(device)
        cam_rgb, stereo = self._build_camera_nodes(pipeline)

        self._q_rgb = cam_rgb.isp.createOutputQueue(maxSize=4, blocking=False)
        self._q_depth = stereo.depth.createOutputQueue(maxSize=4, blocking=False)
        pipeline.start()

        self._pipeline = pipeline
        self._device = pipeline.getDefaultDevice()
        self._cache_intrinsics(cam_rgb)

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
        pipeline = getattr(self, "_pipeline", None)
        if pipeline is not None:
            pipeline.stop()
            self._pipeline = None

        device = getattr(self, "_device", None)
        if device is not None:
            device.close()
            self._device = None


@lru_cache(maxsize=1)
def get_camera() -> OakCamera:
    """Return the singleton OakCamera instance."""
    return OakCamera()
