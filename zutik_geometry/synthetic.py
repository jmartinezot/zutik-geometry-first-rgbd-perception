"""Deterministic synthetic registration fixture used by tests and the CLI."""

from __future__ import annotations

import math

import numpy as np

from .registration import make_transform, transform_points
from .types import ModelCatalogEntry


def _rotation_xyz(angles: tuple[float, float, float]) -> np.ndarray:
    x, y, z = angles
    rx = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(x), -math.sin(x)], [0.0, math.sin(x), math.cos(x)]])
    ry = np.array([[math.cos(y), 0.0, math.sin(y)], [0.0, 1.0, 0.0], [-math.sin(y), 0.0, math.cos(y)]])
    rz = np.array([[math.cos(z), -math.sin(z), 0.0], [math.sin(z), math.cos(z), 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def make_asymmetric_model(seed: int = 42, count: int = 240) -> np.ndarray:
    """Sample an asymmetric, box-like CAD surface in metres."""

    rng = np.random.default_rng(seed)
    points: list[np.ndarray] = []
    for _ in range(count):
        face = int(rng.integers(0, 6))
        point = np.array(
            [rng.uniform(-0.12, 0.12), rng.uniform(-0.08, 0.08), rng.uniform(-0.06, 0.06)],
            dtype=float,
        )
        if face == 0:
            point[0] = -0.12
        elif face == 1:
            point[0] = 0.12
        elif face == 2:
            point[1] = -0.08
        elif face == 3:
            point[1] = 0.08
        elif face == 4:
            point[2] = -0.06
        else:
            point[2] = 0.06
        points.append(point)
    # A raised feature breaks box symmetries and makes the catalogue identity
    # observable from geometry alone.
    raised = rng.uniform([-0.02, -0.035, 0.06], [0.08, 0.035, 0.105], size=(count // 4, 3))
    return np.vstack((np.asarray(points), raised))


def make_synthetic_case() -> tuple[ModelCatalogEntry, np.ndarray, np.ndarray]:
    """Return model, noisy/outlier scene, and the known T_camera_object."""

    rng = np.random.default_rng(123)
    model = ModelCatalogEntry("synthetic-asymmetric-part", make_asymmetric_model(), "synthetic part")
    transform = make_transform(
        _rotation_xyz((0.19, -0.27, 0.34)),
        np.array([0.31, -0.09, 0.78]),
    )
    visible = transform_points(transform, model.points)
    noisy = visible + rng.normal(0.0, 0.0008, visible.shape)
    outliers = rng.uniform([-0.4, -0.35, 0.3], [0.55, 0.35, 1.2], size=(90, 3))
    scene = np.vstack((noisy, outliers))
    return model, scene, transform


def make_synthetic_scene_case() -> tuple[ModelCatalogEntry, np.ndarray, np.ndarray]:
    """Return a non-cropped RGB-D-like scene with support geometry.

    The target is mixed with a dominant support plane, a background wall, and
    isolated outliers. The detector must remove/cluster scene geometry; it is
    not handed the target-only cloud.
    """

    rng = np.random.default_rng(321)
    model = ModelCatalogEntry("synthetic-asymmetric-part", make_asymmetric_model(), "synthetic part")
    transform = make_transform(
        _rotation_xyz((0.23, -0.31, 0.39)),
        np.array([0.28, -0.08, 0.88]),
    )
    target = transform_points(transform, model.points)
    target += rng.normal(0.0, 0.0007, target.shape)

    support_x = np.linspace(-0.55, 0.75, 36)
    support_y = np.linspace(-0.50, 0.45, 30)
    support = np.array(
        [[x, y, 0.70] for x in support_x for y in support_y],
        dtype=float,
    )
    wall_y = np.linspace(-0.50, 0.45, 26)
    wall_z = np.linspace(0.70, 1.30, 18)
    wall = np.array(
        [[-0.52, y, z] for y in wall_y for z in wall_z],
        dtype=float,
    )
    outliers = rng.uniform([-0.65, -0.60, 0.65], [0.85, 0.60, 1.40], size=(140, 3))
    return model, np.vstack((support, wall, target, outliers)), transform


def rotation_error_degrees(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Return the geodesic SO(3) error between two homogeneous transforms."""

    relative = estimate[:3, :3] @ truth[:3, :3].T
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))
