from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence
from scipy.spatial.transform import Rotation


def mat_from_translation_quat(
    translation: Sequence[float],
    quaternion: Sequence[float],
) -> np.ndarray:
    """Build a 4x4 homogeneous matrix from translation and quaternion.

    Args:
        translation: [x, y, z] position.
        quaternion: [x, y, z, w] rotation (scipy convention).

    Returns:
        4x4 numpy array (dtype float64).
    """
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    T[:3, 3] = translation
    return T


def mat_from_rpy(
    translation: Sequence[float],
    rpy: Sequence[float],
) -> np.ndarray:
    """Build a 4x4 homogeneous matrix from translation and roll-pitch-yaw.

    Args:
        translation: [x, y, z] position.
        rpy: [roll, pitch, yaw] in radians (intrinsic XYZ).

    Returns:
        4x4 numpy array (dtype float64).
    """
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = translation
    return T


def invert(T: np.ndarray) -> np.ndarray:
    """Invert a rigid-body 4x4 homogeneous matrix efficiently.

    Uses R^T and -R^T @ t instead of general matrix inverse.
    """
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def compose(*transforms: np.ndarray) -> np.ndarray:
    """Compose multiple 4x4 transforms left-to-right: T_final = T0 @ T1 @ ... @ Tn."""
    result = np.eye(4)
    for T in transforms:
        result = result @ T
    return result


def transform_point(T: np.ndarray, point: Sequence[float]) -> np.ndarray:
    """Apply a 4x4 transform to a 3D point.

    Returns:
        numpy array of shape (3,).
    """
    p_hom = np.array([*point, 1.0])
    return (T @ p_hom)[:3]


def extract_translation(T: np.ndarray) -> np.ndarray:
    """Extract translation vector [x, y, z] from a 4x4 matrix."""
    return T[:3, 3].copy()


def extract_quaternion(T: np.ndarray) -> np.ndarray:
    """Extract quaternion [x, y, z, w] from the rotation part of a 4x4 matrix."""
    return Rotation.from_matrix(T[:3, :3]).as_quat()
