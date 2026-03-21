"""Unit tests for g1.transforms.math — pure 4x4 matrix utilities.

No hardware required. Run with:

    uv run python tests/unit/test_transforms_math.py

All tests use only numpy and scipy (no robot, no camera, no SDK).
"""

from __future__ import annotations

import sys

import numpy as np

from g1.transforms.math import (
    compose,
    extract_quaternion,
    extract_translation,
    invert,
    mat_from_rpy,
    mat_from_translation_quat,
    transform_point,
)


def _pass(name: str) -> None:
    print(f"  ✓ {name}")


def _fail(name: str, detail: str) -> None:
    print(f"  ✗ {name}: {detail}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# mat_from_translation_quat
# ---------------------------------------------------------------------------


def test_identity_quat() -> None:
    T = mat_from_translation_quat([0, 0, 0], [0, 0, 0, 1])
    if not np.allclose(T, np.eye(4)):
        _fail("identity_quat", f"got\n{T}")
    _pass("identity quaternion → identity matrix")


def test_pure_translation() -> None:
    T = mat_from_translation_quat([1, 2, 3], [0, 0, 0, 1])
    expected = np.eye(4)
    expected[:3, 3] = [1, 2, 3]
    if not np.allclose(T, expected):
        _fail("pure_translation", f"got\n{T}")
    _pass("pure translation quaternion")


def test_90deg_yaw() -> None:
    from scipy.spatial.transform import Rotation

    q = Rotation.from_euler("z", 90, degrees=True).as_quat()  # [x,y,z,w]
    T = mat_from_translation_quat([0, 0, 0], q)
    # x-axis should now point in -y direction
    x_axis = T[:3, 0]
    if not np.allclose(x_axis, [0, 1, 0], atol=1e-6):
        _fail("90deg_yaw", f"x-axis mapped to {x_axis}, expected [0,1,0]")
    _pass("90° yaw rotation")


# ---------------------------------------------------------------------------
# mat_from_rpy
# ---------------------------------------------------------------------------


def test_rpy_zero() -> None:
    T = mat_from_rpy([0, 0, 0], [0, 0, 0])
    if not np.allclose(T, np.eye(4)):
        _fail("rpy_zero", f"got\n{T}")
    _pass("zero RPY → identity")


def test_rpy_translation_only() -> None:
    T = mat_from_rpy([3, -1, 2], [0, 0, 0])
    p = transform_point(T, [0, 0, 0])
    if not np.allclose(p, [3, -1, 2]):
        _fail("rpy_translation_only", f"got {p}")
    _pass("RPY pure translation")


# ---------------------------------------------------------------------------
# invert
# ---------------------------------------------------------------------------


def test_invert_identity() -> None:
    T = mat_from_rpy([1, 2, 3], [0.1, 0.2, 0.3])
    result = T @ invert(T)
    if not np.allclose(result, np.eye(4), atol=1e-10):
        _fail("invert_identity", f"T @ inv(T) =\n{result}")
    _pass("T @ inv(T) == I")


def test_invert_pure_translation() -> None:
    T = mat_from_rpy([5, 0, 0], [0, 0, 0])
    T_inv = invert(T)
    p = transform_point(T_inv, [5, 0, 0])
    if not np.allclose(p, [0, 0, 0]):
        _fail("invert_pure_translation", f"got {p}")
    _pass("invert pure translation")


# ---------------------------------------------------------------------------
# compose
# ---------------------------------------------------------------------------


def test_compose_two_translations() -> None:
    T1 = mat_from_rpy([1, 0, 0], [0, 0, 0])
    T2 = mat_from_rpy([0, 2, 0], [0, 0, 0])
    p = transform_point(compose(T1, T2), [0, 0, 0])
    if not np.allclose(p, [1, 2, 0]):
        _fail("compose_two_translations", f"got {p}")
    _pass("compose two translations")


def test_compose_rotation_then_translation() -> None:
    # 90° yaw then move 1m in new x-direction (which is original -y)
    T_rot = mat_from_rpy([0, 0, 0], [0, 0, np.pi / 2])
    T_trans = mat_from_rpy([1, 0, 0], [0, 0, 0])
    result = transform_point(compose(T_rot, T_trans), [0, 0, 0])
    if not np.allclose(result, [0, 1, 0], atol=1e-6):
        _fail("compose_rotation_then_translation", f"got {result}")
    _pass("compose: rotate then translate")


# ---------------------------------------------------------------------------
# transform_point
# ---------------------------------------------------------------------------


def test_transform_point_basic() -> None:
    T = mat_from_rpy([10, 0, 0], [0, 0, 0])
    p = transform_point(T, [1, 0, 0])
    if not np.allclose(p, [11, 0, 0]):
        _fail("transform_point_basic", f"got {p}")
    _pass("transform_point basic offset")


# ---------------------------------------------------------------------------
# extract helpers
# ---------------------------------------------------------------------------


def test_extract_translation_roundtrip() -> None:
    t = [1.5, -2.0, 0.7]
    T = mat_from_rpy(t, [0.1, 0.2, 0.3])
    extracted = extract_translation(T)
    if not np.allclose(extracted, t):
        _fail("extract_translation_roundtrip", f"got {extracted}")
    _pass("extract_translation roundtrip")


def test_extract_quaternion_roundtrip() -> None:
    from scipy.spatial.transform import Rotation

    q_orig = Rotation.from_euler("xyz", [0.1, 0.2, 0.3]).as_quat()
    T = mat_from_translation_quat([0, 0, 0], q_orig)
    q_extracted = extract_quaternion(T)
    # Quaternions can be sign-flipped and still be equal rotations
    if not (np.allclose(q_extracted, q_orig, atol=1e-6) or
            np.allclose(q_extracted, -q_orig, atol=1e-6)):
        _fail("extract_quaternion_roundtrip", f"orig={q_orig} got={q_extracted}")
    _pass("extract_quaternion roundtrip")


# ---------------------------------------------------------------------------
# Camera optical frame → robot base frame convention
# (tests the rotation logic in static.py)
# ---------------------------------------------------------------------------


def test_optical_to_robot_convention() -> None:
    """Camera Z-forward must map to robot X-forward after the optical correction."""
    from g1.transforms.static import _optical_to_robot_rotation

    R = _optical_to_robot_rotation()
    camera_z_forward = np.array([0.0, 0.0, 1.0])
    in_robot = R @ camera_z_forward
    if not np.allclose(in_robot, [1.0, 0.0, 0.0], atol=1e-9):
        _fail("optical_to_robot_z_forward", f"camera Z mapped to {in_robot}, expected robot X")
    _pass("camera Z-forward maps to robot X-forward")

    camera_x_right = np.array([1.0, 0.0, 0.0])
    in_robot = R @ camera_x_right
    if not np.allclose(in_robot, [0.0, -1.0, 0.0], atol=1e-9):
        _fail("optical_to_robot_x_right", f"camera X mapped to {in_robot}, expected robot -Y")
    _pass("camera X-right maps to robot -Y-left")


if __name__ == "__main__":
    print("=== transforms/math.py unit tests ===\n")

    test_identity_quat()
    test_pure_translation()
    test_90deg_yaw()
    test_rpy_zero()
    test_rpy_translation_only()
    test_invert_identity()
    test_invert_pure_translation()
    test_compose_two_translations()
    test_compose_rotation_then_translation()
    test_transform_point_basic()
    test_extract_translation_roundtrip()
    test_extract_quaternion_roundtrip()
    test_optical_to_robot_convention()

    print("\nAll tests passed.")
