from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

DEFAULT_IMAGE_DIR = Path(__file__).resolve().parent / "img"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "calibration.json"
TERMINATION_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate the OAK-D color camera from chessboard images."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Directory containing captured chessboard images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to write the calibration JSON output.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        required=True,
        help="Number of inner chessboard corners per row.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        required=True,
        help="Number of inner chessboard corners per column.",
    )
    parser.add_argument(
        "--square-size",
        type=float,
        default=1.0,
        help="Physical chessboard square size in your chosen unit.",
    )
    parser.add_argument(
        "--show-detections",
        action="store_true",
        help="Preview chessboard detections while calibrating.",
    )
    return parser.parse_args()


def _build_object_points(cols: int, rows: int, square_size: float) -> np.ndarray:
    objp = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = grid * square_size
    return objp


def _collect_corners(
    image_dir: Path,
    cols: int,
    rows: int,
    square_size: float,
    show_detections: bool,
) -> tuple[list[np.ndarray], list[np.ndarray], tuple[int, int]]:
    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    pattern_size = (cols, rows)
    object_template = _build_object_points(cols, rows, square_size)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray,
            pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            print(f"No chessboard found in {image_path.name}")
            continue

        refined = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            TERMINATION_CRITERIA,
        )
        object_points.append(object_template.copy())
        image_points.append(refined)
        image_size = (gray.shape[1], gray.shape[0])
        print(f"Accepted {image_path.name}")

        if show_detections:
            preview = image.copy()
            cv2.drawChessboardCorners(preview, pattern_size, refined, found)
            cv2.imshow("Chessboard Detections", preview)
            cv2.waitKey(250)

    if show_detections:
        cv2.destroyAllWindows()

    if not object_points or image_size is None:
        raise RuntimeError(
            "No valid chessboard detections found. "
            "Capture more images with the full pattern visible."
        )

    return object_points, image_points, image_size


def _compute_mean_reprojection_error(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> float:
    total_error = 0.0
    total_points = 0
    for objp, imgp, rvec, tvec in zip(
        object_points, image_points, rvecs, tvecs, strict=True
    ):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, distortion)
        error = cv2.norm(imgp, projected, cv2.NORM_L2)
        total_error += error * error
        total_points += len(objp)
    return float(np.sqrt(total_error / total_points))


def main() -> None:
    args = parse_args()
    object_points, image_points, image_size = _collect_corners(
        image_dir=args.image_dir,
        cols=args.cols,
        rows=args.rows,
        square_size=args.square_size,
        show_detections=args.show_detections,
    )

    import numpy as np

    rms, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        np.zeros((3, 3), dtype=np.float64),
        np.zeros(5, dtype=np.float64),
    )
    mean_error = _compute_mean_reprojection_error(
        object_points,
        image_points,
        list(rvecs),
        list(tvecs),
        camera_matrix,
        distortion,
    )

    result = {
        "image_dir": str(args.image_dir),
        "num_images_used": len(object_points),
        "pattern": {
            "cols": args.cols,
            "rows": args.rows,
            "square_size": args.square_size,
        },
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "rms_reprojection_error": float(rms),
        "mean_reprojection_error": mean_error,
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": distortion.reshape(-1).tolist(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))

    print(f"Calibration complete. Used {len(object_points)} images.")
    print(f"RMS reprojection error: {rms:.6f}")
    print(f"Mean reprojection error: {mean_error:.6f}")
    print(f"Saved calibration to {args.output}")


if __name__ == "__main__":
    main()
