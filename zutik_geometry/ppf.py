"""Narrow subprocess contract for the experimental PCL PPF spike.

This module intentionally contains no registration algorithm.  It owns the
repository-specific contract around the experimental C++ executable:

* point coordinates and translations are metres;
* the bridge returns ``T_camera_object`` (target/scene <- source/model);
* point clouds are serialized as six-column XYZ + normal text; and
* malformed output and subprocess failures are surfaced rather than converted
  into an apparently valid empty result.

The production Open3D registration path does not import this module.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


_MAGIC = "ZUTIK_PPF_XYZ_NORMALS_V1"


@dataclass(frozen=True)
class PPFBridgeConfig:
    """Explicit configuration passed to the experimental PCL executable."""

    executable: str | Path
    angle_bin_deg: float = 10.0
    distance_bin_m: float = 0.005
    scene_reference_sampling_rate: int = 5
    position_cluster_m: float = 0.03
    rotation_cluster_deg: float = 25.0
    top_k: int = 20
    timeout_s: float = 180.0

    def __post_init__(self) -> None:
        if self.angle_bin_deg <= 0 or self.distance_bin_m <= 0:
            raise ValueError("PPF bins must be positive and expressed in metres/degrees")
        if self.scene_reference_sampling_rate <= 0:
            raise ValueError("scene_reference_sampling_rate must be positive")
        if self.position_cluster_m <= 0 or self.rotation_cluster_deg <= 0:
            raise ValueError("PPF clustering thresholds must be positive")
        if self.top_k <= 0 or self.timeout_s <= 0:
            raise ValueError("top_k and timeout_s must be positive")


@dataclass(frozen=True)
class PPFHypothesis:
    rank: int
    votes: int | None
    T_camera_object: np.ndarray


@dataclass(frozen=True)
class PPFRunResult:
    runtime_s: float
    model_points: int
    scene_points: int
    hypotheses: tuple[PPFHypothesis, ...]
    diagnostics: dict[str, Any]


def _validate_cloud(points: np.ndarray, normals: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} points must have shape (N, 3)")
    if normals.shape != points.shape:
        raise ValueError(f"{name} normals must have shape {points.shape}")
    if points.shape[0] == 0:
        raise ValueError(f"{name} point cloud is empty")
    if not np.isfinite(points).all() or not np.isfinite(normals).all():
        raise ValueError(f"{name} points and normals must be finite")
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 1e-8):
        raise ValueError(f"{name} contains a zero-length normal")
    normals = normals / lengths[:, None]
    return points, normals


def serialize_point_normals(path: str | Path, points: np.ndarray, normals: np.ndarray, *, name: str) -> None:
    """Write the bridge format without any implicit unit conversion."""

    points, normals = _validate_cloud(points, normals, name)
    path = Path(path)
    lines = [f"{_MAGIC} m {points.shape[0]}\n"]
    for point, normal in zip(points, normals, strict=True):
        values = np.concatenate((point, normal))
        lines.append(" ".join(f"{value:.17g}" for value in values) + "\n")
    path.write_text("".join(lines), encoding="utf-8")


def _validate_transform(value: Any, *, context: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{context} must be a finite 4x4 transform")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
        raise ValueError(f"{context} has an invalid homogeneous bottom row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
        raise ValueError(f"{context} rotation is not orthonormal")
    if np.linalg.det(rotation) < 0.0:
        raise ValueError(f"{context} rotation is reflected")
    return transform


def parse_ppf_result(path: str | Path) -> PPFRunResult:
    """Parse and validate a bridge JSON result."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to parse PPF bridge output: {exc}") from exc
    if payload.get("status") != "ok":
        raise RuntimeError(payload.get("error", "PPF bridge reported failure"))
    if payload.get("translation_units") != "m":
        raise ValueError("PPF bridge output must declare translation_units=m")
    if payload.get("transform_convention") != "T_target_source":
        raise ValueError("PPF bridge output must declare T_target_source")

    hypotheses: list[PPFHypothesis] = []
    for index, raw in enumerate(payload.get("hypotheses", []), start=1):
        transform = _validate_transform(raw.get("transform"), context=f"hypotheses[{index - 1}].transform")
        raw_votes = raw.get("votes")
        votes = None if raw_votes is None else int(raw_votes)
        if votes is not None and votes < 0:
            raise ValueError("PPF vote count cannot be negative")
        hypotheses.append(PPFHypothesis(int(raw.get("rank", index)), votes, transform))
    return PPFRunResult(
        runtime_s=float(payload.get("runtime_s", 0.0)),
        model_points=int(payload.get("model_points", 0)),
        scene_points=int(payload.get("scene_points", 0)),
        hypotheses=tuple(hypotheses),
        diagnostics=dict(payload.get("diagnostics", {})),
    )


def run_ppf_bridge(
    model_points: np.ndarray,
    model_normals: np.ndarray,
    scene_points: np.ndarray,
    scene_normals: np.ndarray,
    config: PPFBridgeConfig,
) -> PPFRunResult:
    """Run the experimental executable with explicit metre-based inputs."""

    model_points, model_normals = _validate_cloud(model_points, model_normals, "model")
    scene_points, scene_normals = _validate_cloud(scene_points, scene_normals, "scene")
    executable = Path(config.executable)
    if not executable.exists():
        raise FileNotFoundError(f"PPF executable does not exist: {executable}")

    with tempfile.TemporaryDirectory(prefix="zutik-ppf-") as temp_dir:
        temp = Path(temp_dir)
        model_path = temp / "model.xyzn"
        scene_path = temp / "scene.xyzn"
        result_path = temp / "result.json"
        serialize_point_normals(model_path, model_points, model_normals, name="model")
        serialize_point_normals(scene_path, scene_points, scene_normals, name="scene")
        command: Sequence[str] = [
            str(executable),
            "--model",
            str(model_path),
            "--scene",
            str(scene_path),
            "--output",
            str(result_path),
            "--angle-bin-deg",
            str(config.angle_bin_deg),
            "--distance-bin-m",
            str(config.distance_bin_m),
            "--scene-reference-sampling-rate",
            str(config.scene_reference_sampling_rate),
            "--position-cluster-m",
            str(config.position_cluster_m),
            "--rotation-cluster-deg",
            str(config.rotation_cluster_deg),
            "--top-k",
            str(config.top_k),
        ]
        try:
            completed = subprocess.run(
                list(command),
                check=False,
                capture_output=True,
                text=True,
                timeout=config.timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"PPF bridge timed out after {config.timeout_s:.1f}s") from exc
        if completed.returncode != 0:
            stderr = completed.stderr.strip()[-2000:]
            stdout = completed.stdout.strip()[-1000:]
            raise RuntimeError(
                f"PPF bridge failed with exit code {completed.returncode}; stderr={stderr!r}; stdout={stdout!r}"
            )
        if not result_path.exists():
            raise RuntimeError("PPF bridge exited successfully but did not create its JSON result")
        return parse_ppf_result(result_path)
