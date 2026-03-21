from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import cv2

from g1.vision.camera import get_camera

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "img"
WINDOW_NAME = "OAK-D Chessboard Capture"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview the OAK-D color stream and save chessboard frames."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where captured images will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    camera = get_camera()
    print("OAK-D chessboard capture ready.")
    print("Press s to save the current frame, q to quit.")
    print(f"Saving images to: {output_dir}")

    try:
        while True:
            frame = camera.capture()
            preview = frame.color.copy()
            cv2.putText(
                preview,
                "Press s to save | q to quit",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW_NAME, preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
                image_path = output_dir / f"chessboard_{timestamp}.png"
                cv2.imwrite(str(image_path), frame.color)
                print(f"Saved {image_path}")
    finally:
        cv2.destroyAllWindows()
        camera.stop()


if __name__ == "__main__":
    main()
