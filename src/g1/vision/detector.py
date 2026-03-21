from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from g1.vision.camera import CameraFrame, get_camera

# Minimum reliable depth for OAK-D stereo (meters).
_MIN_DEPTH_M = 0.35


@dataclass(frozen=True, slots=True)
class Detection:
    """A single object detection with optional 3D position."""

    label: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]
    center_uv: tuple[int, int]
    point_camera: tuple[float, float, float] | None
    depth_valid: bool


class ObjectDetector:
    """Zero-shot object detector using YOLO-World + supervision."""

    def __init__(
        self, model_id: str = "yolov8l-worldv2.pt", confidence: float = 0.3
    ) -> None:
        yolo_mod = importlib.import_module("ultralytics")
        self._model = yolo_mod.YOLOWorld(model_id)
        self._confidence = confidence
        self._current_classes: list[str] = []

    def detect(
        self,
        frame: CameraFrame,
        classes: list[str],
    ) -> list[Detection]:
        """Detect objects matching *classes* in the given frame.

        Args:
            frame: An aligned RGB + Depth capture from the OAK-D.
            classes: List of text class names (e.g. ["box", "ball"]).

        Returns:
            List of Detection sorted by confidence (highest first).
        """
        sv = importlib.import_module("supervision")

        if classes != self._current_classes:
            self._model.set_classes(classes)
            self._current_classes = list(classes)
        results = self._model.predict(frame.color, conf=self._confidence, verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])

        camera = get_camera()
        output: list[Detection] = []

        for i in range(len(detections)):
            bbox = detections.xyxy[i].astype(int)
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Sample a 5x5 patch around center for robust depth
            depth_m = self._median_depth_patch(frame.depth, cx, cy, patch_size=5)

            depth_ok = depth_m is not None and depth_m >= _MIN_DEPTH_M
            point: tuple[float, float, float] | None = None
            if depth_ok and depth_m is not None:
                point = camera.deproject(cx, cy, depth_m)

            class_id = (
                int(detections.class_id[i]) if detections.class_id is not None else 0
            )
            label = classes[class_id] if class_id < len(classes) else "unknown"
            conf = (
                float(detections.confidence[i])
                if detections.confidence is not None
                else 0.0
            )

            output.append(
                Detection(
                    label=label,
                    confidence=conf,
                    bbox_xyxy=(x1, y1, x2, y2),
                    center_uv=(cx, cy),
                    point_camera=point,
                    depth_valid=depth_ok,
                )
            )

        output.sort(key=lambda d: d.confidence, reverse=True)
        return output

    @staticmethod
    def _median_depth_patch(
        depth: np.ndarray,
        u: int,
        v: int,
        patch_size: int = 5,
    ) -> float | None:
        """Sample a patch around (u, v) and return median non-zero depth.

        Returns None if no valid depth pixels exist in the patch.
        """
        h, w = depth.shape[:2]
        half = patch_size // 2
        y0 = max(0, v - half)
        y1 = min(h, v + half + 1)
        x0 = max(0, u - half)
        x1 = min(w, u + half + 1)

        patch = depth[y0:y1, x0:x1].flatten()
        valid = patch[patch > 0.0]
        if len(valid) == 0:
            return None
        return float(np.median(valid))


@lru_cache(maxsize=1)
def get_detector() -> ObjectDetector:
    """Return the singleton ObjectDetector instance."""
    return ObjectDetector()
