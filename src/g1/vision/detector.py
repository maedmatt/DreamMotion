from __future__ import annotations

import argparse
import importlib
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np

from g1.vision.camera import CameraFrame, get_camera

_MIN_DEPTH_M = 0.35
_WINDOW_NAME = "OAK-D Prompt Detection"


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
        self,
        model_id: str = "yolov8l-worldv2.pt",
        confidence: float = 0.3,
        device: str | None = None,
    ) -> None:
        yolo_mod = importlib.import_module("ultralytics")
        torch_mod = importlib.import_module("torch")
        self._torch = torch_mod
        self._device = device or ("cuda:0" if torch_mod.cuda.is_available() else "cpu")
        self._model = yolo_mod.YOLOWorld(model_id)
        self._confidence = confidence
        self._current_classes: list[str] = []
        self._sync_model_device()

    def _sync_model_device(self) -> None:
        self._model.to(self._device)
        txt_feats = getattr(self._model.model, "txt_feats", None)
        if txt_feats is not None and hasattr(txt_feats, "to"):
            self._model.model.txt_feats = txt_feats.to(self._device)

        predictor = getattr(self._model, "predictor", None)
        if predictor is not None and getattr(predictor, "model", None) is not None:
            predictor.model.to(self._device)

    def detect(self, frame: CameraFrame, classes: list[str]) -> list[Detection]:
        sv = importlib.import_module("supervision")

        if classes != self._current_classes:
            self._model.set_classes(classes)
            self._current_classes = list(classes)
            self._sync_model_device()

        try:
            results = self._model.predict(
                frame.color,
                conf=self._confidence,
                verbose=False,
                device=self._device,
            )
        except RuntimeError as exc:
            if "Expected all tensors to be on the same device" not in str(exc):
                raise
            if self._device == "cpu":
                raise
            self._device = "cpu"
            self._sync_model_device()
            results = self._model.predict(
                frame.color,
                conf=self._confidence,
                verbose=False,
                device=self._device,
            )

        detections = sv.Detections.from_ultralytics(results[0])

        output: list[Detection] = []
        for i in range(len(detections)):
            bbox = detections.xyxy[i].astype(int)
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            depth_m = self._median_depth_patch(frame.depth, cx, cy, patch_size=5)
            depth_ok = depth_m is not None and depth_m >= _MIN_DEPTH_M
            point: tuple[float, float, float] | None = None
            if depth_ok and depth_m is not None:
                point = self._deproject(frame, cx, cy, depth_m)

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

        output.sort(key=lambda item: item.confidence, reverse=True)
        return output

    @staticmethod
    def _median_depth_patch(
        depth: np.ndarray,
        u: int,
        v: int,
        patch_size: int = 5,
    ) -> float | None:
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

    @staticmethod
    def _deproject(
        frame: CameraFrame,
        u: int,
        v: int,
        depth_m: float,
    ) -> tuple[float, float, float]:
        intr = frame.intrinsics
        z = depth_m
        x = (u - intr.cx) * z / intr.fx
        y = (v - intr.cy) * z / intr.fy
        return (x, y, z)


@lru_cache(maxsize=1)
def get_detector() -> ObjectDetector:
    return ObjectDetector()


@dataclass(slots=True)
class PromptState:
    prompt: str | None = None
    running: bool = True
    lock: threading.Lock = field(default_factory=threading.Lock)

    def get_prompt(self) -> str | None:
        with self.lock:
            return self.prompt

    def set_prompt(self, prompt: str | None) -> None:
        with self.lock:
            self.prompt = prompt

    def stop(self) -> None:
        with self.lock:
            self.running = False

    def is_running(self) -> bool:
        with self.lock:
            return self.running


@dataclass(slots=True)
class FrameState:
    frame: CameraFrame | None = None
    frame_id: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def publish(self, frame: CameraFrame) -> None:
        with self.lock:
            self.frame = frame
            self.frame_id += 1

    def snapshot(self) -> tuple[int, CameraFrame | None]:
        with self.lock:
            return self.frame_id, self.frame


@dataclass(slots=True)
class DetectionState:
    prompt: str | None = None
    frame_id: int = 0
    detections: list[Detection] = field(default_factory=list)
    detector_status: str = "starting"
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def set_status(self, status: str) -> None:
        with self.lock:
            self.detector_status = status
            self.error = None

    def set_error(self, error: str) -> None:
        with self.lock:
            self.error = error
            self.detector_status = "error"

    def publish(self, prompt: str | None, frame_id: int, detections: list[Detection]) -> None:
        with self.lock:
            self.prompt = prompt
            self.frame_id = frame_id
            self.detections = list(detections)
            self.detector_status = "ready"
            self.error = None

    def snapshot(self) -> tuple[str | None, int, list[Detection], str, str | None]:
        with self.lock:
            return (
                self.prompt,
                self.frame_id,
                list(self.detections),
                self.detector_status,
                self.error,
            )


def _prompt_to_classes(prompt: str) -> list[str]:
    classes = [item.strip() for item in prompt.split(",") if item.strip()]
    return classes or [prompt]


def _camera_worker(prompt_state: PromptState, frame_state: FrameState) -> None:
    camera = get_camera()
    try:
        while prompt_state.is_running():
            frame_state.publish(camera.capture())
    finally:
        camera.stop()


def _prompt_worker(prompt_state: PromptState) -> None:
    print("Type a prompt in the terminal and press Enter to update detection.")
    print("Examples: bottle, chair, red box")
    print("Use blank input to clear detections. Type 'exit' to quit.\n")

    while prompt_state.is_running():
        try:
            prompt = input("Prompt> ").strip()
        except EOFError:
            prompt_state.stop()
            return
        except KeyboardInterrupt:
            prompt_state.stop()
            return

        if prompt.lower() in {"exit", "quit"}:
            prompt_state.stop()
            return

        prompt_state.set_prompt(prompt or None)
        if prompt:
            print(f"Tracking prompt: {prompt}")
        else:
            print("Cleared prompt. Preview will continue without detections.")


def _detector_worker(
    prompt_state: PromptState,
    frame_state: FrameState,
    detection_state: DetectionState,
    model_id: str,
    confidence: float,
) -> None:
    try:
        detection_state.set_status("loading model")
        detector = ObjectDetector(model_id=model_id, confidence=confidence)
        detection_state.set_status("ready")
    except Exception as exc:
        detection_state.set_error(f"model load failed: {exc}")
        return

    last_processed: tuple[int, str | None] | None = None
    while prompt_state.is_running():
        prompt = prompt_state.get_prompt()
        frame_id, frame = frame_state.snapshot()
        if frame is None:
            time.sleep(0.01)
            continue

        job_key = (frame_id, prompt)
        if job_key == last_processed:
            time.sleep(0.01)
            continue
        last_processed = job_key

        if not prompt:
            detection_state.publish(None, frame_id, [])
            continue

        try:
            detections = detector.detect(frame, _prompt_to_classes(prompt))
            detection_state.publish(prompt, frame_id, detections)
        except Exception as exc:
            detection_state.set_error(f"detect failed: {exc}")
            time.sleep(0.1)


def _format_pose(detection: Detection) -> str:
    if detection.point_camera is None:
        return "pose unavailable"

    x, y, z = detection.point_camera
    return f"x={x:.2f} y={y:.2f} z={z:.2f}m"


def _draw_detection(preview: np.ndarray, detection: Detection, *, show_pose: bool) -> None:
    x1, y1, x2, y2 = detection.bbox_xyxy
    cx, cy = detection.center_uv

    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.circle(preview, (cx, cy), 4, (0, 255, 0), -1)

    lines = [f"{detection.label} {detection.confidence:.2f}"]
    if show_pose:
        lines.append(_format_pose(detection))

    text_y = max(30, y1 - 12)
    for index, line in enumerate(lines):
        cv2.putText(
            preview,
            line,
            (x1, text_y + 24 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def parse_live_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live OAK-D + YOLO-World prompt detector."
    )
    parser.add_argument("--prompt", type=str, default=None, help="Optional initial prompt.")
    parser.add_argument(
        "--model-id",
        type=str,
        default="yolov8l-worldv2.pt",
        help="YOLO-World model to load.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--show-pose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay the estimated camera-frame pose when depth is valid.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_live_args()

    prompt_state = PromptState(prompt=args.prompt.strip() if args.prompt else None)
    frame_state = FrameState()
    detection_state = DetectionState()

    print("Starting OAK-D live detector...")
    print("Main thread: visualization")
    print("Worker thread 1: camera publisher")
    print("Worker thread 2: terminal prompt listener")
    print("Worker thread 3: YOLO detector\n")

    camera_thread = threading.Thread(
        target=_camera_worker,
        args=(prompt_state, frame_state),
        daemon=True,
        name="camera-publisher",
    )
    prompt_thread = threading.Thread(
        target=_prompt_worker,
        args=(prompt_state,),
        daemon=True,
        name="prompt-listener",
    )
    detector_thread = threading.Thread(
        target=_detector_worker,
        args=(prompt_state, frame_state, detection_state, args.model_id, args.confidence),
        daemon=True,
        name="yolo-detector",
    )

    camera_thread.start()
    prompt_thread.start()
    detector_thread.start()

    cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)
    print("Preview window opened. Press q in the preview window to quit.\n")

    try:
        while prompt_state.is_running():
            prompt = prompt_state.get_prompt()
            frame_id, frame = frame_state.snapshot()
            det_prompt, det_frame_id, detections, detector_status, detector_error = (
                detection_state.snapshot()
            )

            if frame is None:
                preview = np.zeros((720, 1280, 3), dtype=np.uint8)
                status = "Waiting for camera frames..."
            else:
                preview = frame.color.copy()
                if prompt and prompt == det_prompt and det_frame_id == frame_id:
                    for detection in detections:
                        _draw_detection(preview, detection, show_pose=args.show_pose)
                status = (
                    f"Prompt: {prompt if prompt else '<none>'} | "
                    f"detector: {detector_status} | detections: {len(detections) if prompt else 0}"
                )

            cv2.putText(
                preview,
                status,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                preview,
                "q: quit | type prompt in terminal",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if detector_error:
                cv2.putText(
                    preview,
                    detector_error,
                    (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(_WINDOW_NAME, preview)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        prompt_state.stop()
        cv2.destroyAllWindows()
        try:
            get_camera().stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
