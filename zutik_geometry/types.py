"""Public data contracts for the geometry-first branch.

Frame convention:
    ``T_A_B`` maps a point expressed in frame B into frame A.  A catalogue
    model is expressed in its object frame, so registration returns
    ``T_camera_object``.  If the input camera pose is available,
    ``T_world_object = T_world_camera @ T_camera_object``.

All distances and translations in this package are metres.  Rotation is a
right-handed 3x3 matrix embedded in a 4x4 homogeneous transform.  BOP output
conversion to millimetres is handled only by the BOP adapter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


def _as_transform(value: np.ndarray | list[list[float]], name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 4x4 homogeneous transform")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise ValueError(f"{name} must have homogeneous last row [0, 0, 0, 1]")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation must be orthonormal")
    if np.linalg.det(rotation) < 0.0:
        raise ValueError(f"{name} rotation must have positive determinant")
    return matrix


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics for depth-to-camera conversion, in pixels."""

    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if not all(np.isfinite(v) and v > 0.0 for v in (self.fx, self.fy)):
            raise ValueError("fx and fy must be positive finite values")
        if not all(np.isfinite(v) for v in (self.cx, self.cy)):
            raise ValueError("cx and cy must be finite values")

    @classmethod
    def from_matrix(cls, matrix: np.ndarray | list[list[float]]) -> "CameraIntrinsics":
        camera_matrix = np.asarray(matrix, dtype=float)
        if camera_matrix.shape != (3, 3):
            raise ValueError("camera matrix must be 3x3")
        return cls(
            fx=float(camera_matrix[0, 0]),
            fy=float(camera_matrix[1, 1]),
            cx=float(camera_matrix[0, 2]),
            cy=float(camera_matrix[1, 2]),
        )

    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=float,
        )


@dataclass(frozen=True)
class ModelCatalogEntry:
    """Known object model sampled in the object/model frame."""

    model_id: str
    points: np.ndarray
    semantic_label: str | None = None
    units: str = "m"
    representation: str = "custom"
    normals: np.ndarray | None = None

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
            raise ValueError("model points must have shape (N, 3) with N >= 3")
        if not np.all(np.isfinite(points)):
            raise ValueError("model points must be finite")
        if self.units != "m":
            raise ValueError("ModelCatalogEntry.points must be converted to metres")
        if self.representation not in {"custom", "cad", "reconst"}:
            raise ValueError("model representation must be custom, cad, or reconst")
        normals = None if self.normals is None else np.asarray(self.normals, dtype=float)
        if normals is not None:
            if normals.shape != points.shape:
                raise ValueError("model normals must have the same shape as model points")
            if not np.all(np.isfinite(normals)):
                raise ValueError("model normals must be finite")
            lengths = np.linalg.norm(normals, axis=1)
            if np.any(lengths <= 1e-12):
                raise ValueError("model normals must be non-zero")
            normals = normals / lengths[:, None]
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "normals", normals)

    @property
    def bounds_m(self) -> tuple[np.ndarray, np.ndarray]:
        return np.min(self.points, axis=0), np.max(self.points, axis=0)


@dataclass(frozen=True)
class DetectionConfig:
    """Open3D registration, filtering, and scene-candidate parameters."""

    voxel_size_m: float = 0.005
    normal_radius_factor: float = 2.0
    normal_neighbors: int = 30
    feature_radius_factor: float = 5.0
    feature_neighbors: int = 100
    max_registration_points: int = 2500
    ransac_iterations: int = 20000
    ransac_confidence: float = 0.999
    ransac_inlier_threshold_m: float = 0.015
    icp_iterations: int = 35
    icp_max_correspondence_m: float = 0.02
    min_ransac_inliers: int = 6
    min_icp_inliers: int = 12
    min_fitness: float = 0.20
    max_rmse_m: float = 0.02
    dedup_translation_m: float = 0.025
    dedup_rotation_deg: float = 15.0
    random_seed: int = 7
    candidate_mode: str = "geometry_candidates"
    model_representation: str = "custom"
    model_normal_source: str = "auto"
    observed_normal_orientation: str = "towards_camera"
    verification_mode: str = "bidirectional"
    verification_distance_m: float = 0.02
    # A 20 mm, two-sided overlap gate is deliberately broad enough for partial
    # RGB-D visibility and sensor/model discrepancy.  It is not a pose score.
    min_bidirectional_overlap: float = 0.20
    verification_max_points: int = 2500
    min_candidate_points: int = 30
    max_candidate_extent_m: float = 0.75
    cluster_eps_m: float = 0.04
    cluster_min_points: int = 20
    remove_support_plane: bool = False
    support_plane_distance_m: float = 0.008
    support_plane_min_ratio: float = 0.45
    support_plane_iterations: int = 1000
    remove_statistical_outliers: bool = True
    outlier_neighbors: int = 20
    outlier_std_ratio: float = 2.0

    def __post_init__(self) -> None:
        positive = {
            "voxel_size_m": self.voxel_size_m,
            "normal_radius_factor": self.normal_radius_factor,
            "normal_neighbors": self.normal_neighbors,
            "feature_radius_factor": self.feature_radius_factor,
            "feature_neighbors": self.feature_neighbors,
            "max_registration_points": self.max_registration_points,
            "ransac_iterations": self.ransac_iterations,
            "ransac_confidence": self.ransac_confidence,
            "ransac_inlier_threshold_m": self.ransac_inlier_threshold_m,
            "icp_iterations": self.icp_iterations,
            "icp_max_correspondence_m": self.icp_max_correspondence_m,
            "min_ransac_inliers": self.min_ransac_inliers,
            "min_icp_inliers": self.min_icp_inliers,
            "cluster_eps_m": self.cluster_eps_m,
            "cluster_min_points": self.cluster_min_points,
            "support_plane_iterations": self.support_plane_iterations,
            "outlier_neighbors": self.outlier_neighbors,
            "outlier_std_ratio": self.outlier_std_ratio,
            "verification_distance_m": self.verification_distance_m,
            "verification_max_points": self.verification_max_points,
            "min_candidate_points": self.min_candidate_points,
            "max_candidate_extent_m": self.max_candidate_extent_m,
        }
        if any(float(value) <= 0.0 for value in positive.values()):
            raise ValueError("registration parameters must be positive")
        if not 0.0 < self.ransac_confidence < 1.0:
            raise ValueError("ransac_confidence must be in (0, 1)")
        if not 0.0 < self.min_fitness <= 1.0:
            raise ValueError("min_fitness must be in (0, 1]")
        if self.max_rmse_m <= 0.0 or self.dedup_translation_m <= 0.0:
            raise ValueError("distance thresholds must be positive")
        if self.dedup_rotation_deg <= 0.0:
            raise ValueError("dedup_rotation_deg must be positive")
        if not 0.0 < self.support_plane_min_ratio <= 1.0:
            raise ValueError("support_plane_min_ratio must be in (0, 1]")
        if self.candidate_mode not in {"full", "dbscan", "full_and_dbscan", "geometry_candidates"}:
            raise ValueError(
                "candidate_mode must be full, dbscan, full_and_dbscan, or geometry_candidates"
            )
        if self.model_representation not in {"custom", "cad", "reconst"}:
            raise ValueError("model_representation must be custom, cad, or reconst")
        if self.model_normal_source not in {"auto", "mesh", "estimate"}:
            raise ValueError("model_normal_source must be auto, mesh, or estimate")
        if self.observed_normal_orientation not in {"none", "towards_camera", "away_from_camera"}:
            raise ValueError(
                "observed_normal_orientation must be none, towards_camera, or away_from_camera"
            )
        if self.verification_mode not in {"none", "bidirectional"}:
            raise ValueError("verification_mode must be none or bidirectional")
        if not 0.0 < self.min_bidirectional_overlap <= 1.0:
            raise ValueError("min_bidirectional_overlap must be in (0, 1]")
        if self.outlier_std_ratio <= 0.0:
            raise ValueError("outlier_std_ratio must be positive")


@dataclass(frozen=True)
class RegistrationDiagnostics:
    """Raw Open3D evidence retained alongside the acceptance decision.

    ``inlier_ratio`` is the accepted RANSAC correspondence count divided by
    the pre-RANSAC feature correspondence count. ``global_fitness`` is
    Open3D's downsampled RANSAC source coverage and ``fitness`` is the ICP
    source coverage.
    """

    method: str
    accepted: bool
    failure_reason: str | None
    model_points: int
    scene_points: int
    candidate_points: int
    candidate_source: str
    candidate_index: int
    feature_correspondences: int
    ransac_inliers: int
    icp_inliers: int
    inlier_ratio: float
    global_fitness: float
    fitness: float
    ransac_rmse_m: float | None
    rmse_m: float | None
    icp_iterations: int
    downsample_voxel_m: float
    model_representation: str = "custom"
    model_normal_source: str = "estimate"
    observed_normal_orientation: str = "none"
    verification_mode: str = "none"
    model_to_candidate_overlap: float | None = None
    candidate_to_model_overlap: float | None = None
    bidirectional_overlap: float | None = None
    model_to_candidate_rmse_m: float | None = None
    candidate_to_model_rmse_m: float | None = None
    model_to_candidate_trimmed_rmse_m: float | None = None
    candidate_to_model_trimmed_rmse_m: float | None = None
    extent_ratio_min: float | None = None
    extent_ratio_max: float | None = None
    candidate_generation: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeometricVerification:
    """Bidirectional model/candidate evidence after pose refinement.

    These values are geometric coverage and residual measurements, not
    probabilities.  Both directions are retained because one-sided model
    coverage can accept a small model embedded in a much larger scene.
    """

    distance_threshold_m: float
    model_to_candidate_overlap: float
    candidate_to_model_overlap: float
    bidirectional_overlap: float
    model_to_candidate_rmse_m: float
    candidate_to_model_rmse_m: float
    model_to_candidate_trimmed_rmse_m: float
    candidate_to_model_trimmed_rmse_m: float
    extent_ratio: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "distance_threshold_m": self.distance_threshold_m,
            "model_to_candidate_overlap": self.model_to_candidate_overlap,
            "candidate_to_model_overlap": self.candidate_to_model_overlap,
            "bidirectional_overlap": self.bidirectional_overlap,
            "model_to_candidate_rmse_m": self.model_to_candidate_rmse_m,
            "candidate_to_model_rmse_m": self.candidate_to_model_rmse_m,
            "model_to_candidate_trimmed_rmse_m": self.model_to_candidate_trimmed_rmse_m,
            "candidate_to_model_trimmed_rmse_m": self.candidate_to_model_trimmed_rmse_m,
            "extent_ratio": self.extent_ratio.tolist(),
        }


@dataclass(frozen=True)
class GraphReadyObjectRecord:
    """Stable object record suitable for an existing world-model adapter.

    This package does not implement a scene graph.  The record deliberately
    carries only one observation and its geometric evidence, allowing an
    existing World Semantic Model to decide association and graph updates.
    """

    object_id: str
    model_id: str
    semantic_label: str | None
    pose_frame: str
    pose_matrix: np.ndarray
    position_m: np.ndarray
    size_m: np.ndarray
    confidence: float
    accepted: bool
    geometric_evidence: RegistrationDiagnostics

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "model_id": self.model_id,
            "semantic_label": self.semantic_label,
            "pose_frame": self.pose_frame,
            "pose_matrix": self.pose_matrix.tolist(),
            "position_m": self.position_m.tolist(),
            "size_m": self.size_m.tolist(),
            "confidence": self.confidence,
            "accepted": self.accepted,
            "geometric_evidence": asdict(self.geometric_evidence),
        }


@dataclass(frozen=True)
class PoseHypothesis:
    """One model-to-camera pose and its optional world-frame composition."""

    model_id: str
    semantic_label: str | None
    T_camera_object: np.ndarray | None
    T_world_object: np.ndarray | None
    camera_frame: str
    world_frame: str | None
    timestamp: float | None
    confidence: float
    accepted: bool
    diagnostics: RegistrationDiagnostics
    model_bounds_m: tuple[np.ndarray, np.ndarray] | None = None

    def __post_init__(self) -> None:
        if self.T_camera_object is not None:
            object.__setattr__(
                self,
                "T_camera_object",
                _as_transform(self.T_camera_object, "T_camera_object"),
            )
        if self.T_world_object is not None:
            object.__setattr__(
                self,
                "T_world_object",
                _as_transform(self.T_world_object, "T_world_object"),
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.accepted and self.T_camera_object is None:
            raise ValueError("accepted hypotheses must contain T_camera_object")

    @property
    def object_id(self) -> str:
        return self.model_id

    def to_graph_record(self) -> GraphReadyObjectRecord:
        if self.T_camera_object is None:
            raise ValueError("rejected hypotheses do not have a graph-ready pose")
        transform = self.T_world_object if self.T_world_object is not None else self.T_camera_object
        pose_frame = self.world_frame if self.T_world_object is not None else self.camera_frame
        if self.model_bounds_m is None:
            size = np.zeros(3, dtype=float)
        else:
            size = self.model_bounds_m[1] - self.model_bounds_m[0]
        return GraphReadyObjectRecord(
            object_id=self.object_id,
            model_id=self.model_id,
            semantic_label=self.semantic_label,
            pose_frame=pose_frame,
            pose_matrix=transform.copy(),
            position_m=transform[:3, 3].copy(),
            size_m=size,
            confidence=self.confidence,
            accepted=self.accepted,
            geometric_evidence=self.diagnostics,
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "model_id": self.model_id,
            "semantic_label": self.semantic_label,
            "T_camera_object": None
            if self.T_camera_object is None
            else self.T_camera_object.tolist(),
            "T_world_object": None
            if self.T_world_object is None
            else self.T_world_object.tolist(),
            "camera_frame": self.camera_frame,
            "world_frame": self.world_frame,
            "timestamp": self.timestamp,
            "translation_units": "m",
            "rotation_format": "3x3 matrix in homogeneous transform",
            "confidence": self.confidence,
            "accepted": self.accepted,
            "diagnostics": asdict(self.diagnostics),
        }
        if self.model_bounds_m is not None:
            result["model_bounds_m"] = [
                self.model_bounds_m[0].tolist(),
                self.model_bounds_m[1].tolist(),
            ]
        return result


def validate_world_camera_transform(value: np.ndarray | list[list[float]]) -> np.ndarray:
    """Validate and normalize a camera-to-world transform."""

    return _as_transform(value, "T_world_camera")
