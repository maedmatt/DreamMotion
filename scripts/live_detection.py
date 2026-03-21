"""Live OAK-D camera feed with YOLO-World detection overlay.

Validates camera.py + detector.py on a laptop with the OAK-D plugged in via USB.
Press Q to quit. Press S to save a snapshot.

Run with:

    uv run python scripts/live_detection.py
    uv run python scripts/live_detection.py --classes "box,bottle,person"
    uv run python scripts/live_detection.py --no-display   # headless, saves frames

What this validates:
  - OAK-D pipeline starts and delivers aligned RGB + depth
  - YOLO-World detects objects from a text prompt in real time
  - Depth values and 3D coordinates are printed for each detection
  - The intrinsics read from the device calibration look sane
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from g1.vision.detector import Detection


def _draw_detections(
    color_bgr: np.ndarray,
    detections: list[Detection],
    depth: np.ndarray,
) -> np.ndarray:
    """Draw bounding boxes + depth onto a copy of the frame."""
    import cv2

    canvas = color_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        color = (0, 255, 0) if det.depth_valid else (0, 100, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        label = f"{det.label} {det.confidence:.2f}"
        if det.point_camera is not None:
            _x, _y, z = det.point_camera
            label += f"  z={z:.2f}m"
        cv2.putText(
            canvas,
            label,
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

        # Mark center
        cx, cy = det.center_uv
        cv2.circle(canvas, (cx, cy), 4, (0, 0, 255), -1)

    return canvas


def run(classes: list[str], display: bool, save_dir: str | None) -> None:
    from pathlib import Path

    import cv2

    from g1.vision.camera import get_camera
    from g1.vision.detector import get_detector

    print("Starting OAK-D pipeline (~3s)...")
    cam = get_camera()
    print(
        f"  intrinsics: fx={cam.intrinsics.fx:.1f}  fy={cam.intrinsics.fy:.1f}"
        f"  cx={cam.intrinsics.cx:.1f}  cy={cam.intrinsics.cy:.1f}"
        f"  size={cam.intrinsics.width}x{cam.intrinsics.height}"
    )

    print("Loading YOLO-World model (first run downloads weights ~50MB)...")
    detector = get_detector()

    print(f"\nDetecting: {classes}")
    print("Press Q to quit, S to save snapshot.\n")

    save_path = Path(save_dir) if save_dir else Path("output/snapshots")
    frame_count = 0
    t0 = time.time()

    while True:
        frame = cam.capture()
        detections = detector.detect(frame, classes)

        fps = frame_count / max(time.time() - t0, 1e-6)
        frame_count += 1

        # Print detections to terminal
        if detections:
            for d in detections:
                xyz = f"  3D={d.point_camera}" if d.depth_valid else "  depth_invalid"
                print(f"  [{d.label}] conf={d.confidence:.2f}{xyz}")
        else:
            print(f"  (no detections)  fps={fps:.1f}", end="\r")

        canvas: np.ndarray | None = None
        if display or save_dir:
            canvas = _draw_detections(frame.color, detections, frame.depth)
            cv2.putText(
                canvas,
                f"fps={fps:.1f}  classes={classes}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        if display and canvas is not None:
            cv2.imshow("OAK-D  |  YOLO-World", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\nQuitting.")
                break
            if key == ord("s"):
                save_path.mkdir(parents=True, exist_ok=True)
                ts = int(time.time())
                cv2.imwrite(str(save_path / f"snap_{ts}.jpg"), canvas)
                print(f"\n  Saved {save_path}/snap_{ts}.jpg")

        if save_dir and canvas is not None and frame_count % 30 == 0:
            save_path.mkdir(parents=True, exist_ok=True)
            ts = int(time.time())
            cv2.imwrite(str(save_path / f"frame_{ts}.jpg"), canvas)

        if not display and frame_count >= 30:
            print(
                f"\nHeadless run complete ({frame_count} frames). "
                f"Frames saved to {save_path}."
            )
            break

    if display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live OAK-D detection demo")
    parser.add_argument(
        "--classes",
        default="box,person,bottle",
        help="Comma-separated class names for YOLO-World (default: box,person,bottle)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Headless mode: save frames to output/snapshots instead of showing window",
    )
    parser.add_argument(
        "--save",
        metavar="DIR",
        help="Also save annotated frames to this directory",
    )
    args = parser.parse_args()

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    display = not args.no_display

    try:
        run(classes=classes, display=display, save_dir=args.save)
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(0)
