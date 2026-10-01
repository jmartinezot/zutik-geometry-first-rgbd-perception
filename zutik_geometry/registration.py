"""Open3D-backed geometry-first RGB-D registration.

The low-level registration is intentionally delegated to Open3D's established
implementation: voxel downsampling, hybrid-search normal estimation, FPFH,
feature-based RANSAC, and point-to-plane ICP. This module owns orchestration,
scene candidate generation, contracts, thresholds, and diagnostics.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np

try:  # Keep import-time errors actionable for callers that forgot the extra.
    import open3d as o3d
except ModuleNotFoundError:  # pragma: no cover - exercised only without install
    o3d = None  # type: ignore[assignment]

from .types import (
    CameraIntrinsics,
    DetectionConfig,
    GeometricVerification,
    ModelCatalogEntry,
    PoseHypothesis,
    RegistrationDiagnostics,
    validate_world_camera_transform,
)


METHOD_NAME = "open3d-fpfh-ransac-point-to-plane-icp"


def _require_open3d() -> object:
    if o3d is None:
        raise RuntimeError(
            "Open3D is required for geometry registration; install the project with "
            "python -m pip install -e ."
        )
    return o3d


def validate_point_cloud(points: np.ndarray | Sequence[Sequence[float]], name: str) -> np.ndarray:
    """Return finite float points or raise a useful input error."""

    cloud = np.asarray(points, dtype=float)
    if cloud.ndim != 2 or cloud.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    if len(cloud) < 3:
        raise ValueError(f"{name} must contain at least three points")
    if not np.all(np.isfinite(cloud)):
        raise ValueError(f"{name} must contain only finite points")
    return cloud


def depth_to_point_cloud(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    depth_scale_m: float = 1.0,
    min_depth_m: float = 0.05,
    max_depth_m: float = 10.0,
) -> np.ndarray:
    """Project a depth image into a camera-frame cloud in metres.

    ``depth_scale_m`` explicitly converts one stored depth unit to metres;
    e.g. a uint16 image in millimetres uses ``0.001``. The returned points
    satisfy x-right, y-down, z-forward camera coordinates.
    """

    image = np.asarray(depth)
    if image.ndim != 2:
        raise ValueError("depth must be a 2-D array")
    if not np.isfinite(depth_scale_m) or depth_scale_m <= 0.0:
        raise ValueError("depth_scale_m must be positive")
    if not 0.0 < min_depth_m < max_depth_m:
        raise ValueError("depth range must satisfy 0 < min_depth_m < max_depth_m")
    depth_m = image.astype(float) * depth_scale_m
    valid = np.isfinite(depth_m) & (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
    v, u = np.nonzero(valid)
    z = depth_m[v, u]
    x = (u.astype(float) - intrinsics.cx) * z / intrinsics.fx
    y = (v.astype(float) - intrinsics.cy) * z / intrinsics.fy
    return np.column_stack((x, y, z))


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a homogeneous transform whose convention is T_target_source."""

    return points @ transform[:3, :3].T + transform[:3, 3]


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=float)
    result[:3, :3] = np.asarray(rotation, dtype=float)
    result[:3, 3] = np.asarray(translation, dtype=float)
    return result


def _to_point_cloud(points: np.ndarray, normals: np.ndarray | None = None) -> object:
    open3d = _require_open3d()
    cloud = open3d.geometry.PointCloud()
    cloud.points = open3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    if normals is not None:
        cloud.normals = open3d.utility.Vector3dVector(np.asarray(normals, dtype=float))
    return cloud


def _cloud_points(cloud: object) -> np.ndarray:
    return np.asarray(cloud.points, dtype=float)


def _limit_points(cloud: object, maximum: int) -> object:
    points = _cloud_points(cloud)
    if len(points) <= maximum:
        return cloud
    indices = np.linspace(0, len(points) - 1, maximum, dtype=np.int64)
    normals = None
    if cloud.has_normals():
        normals = np.asarray(cloud.normals, dtype=float)[indices]
    return _to_point_cloud(points[indices], normals)


def _voxel_cloud(
    points: np.ndarray,
    config: DetectionConfig,
    *,
    normals: np.ndarray | None = None,
) -> object:
    cloud = _to_point_cloud(points, normals)
    downsampled = cloud.voxel_down_sample(config.voxel_size_m)
    if len(downsampled.points) < 3:
        return downsampled
    return _limit_points(downsampled, config.max_registration_points)


def _estimate_normals(
    cloud: object,
    config: DetectionConfig,
    radius_factor: float,
    *,
    orientation: str = "none",
) -> None:
    if len(cloud.points) < 3:
        raise ValueError("at least three points are required for normal estimation")
    radius = config.voxel_size_m * radius_factor
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=config.normal_neighbors,
        )
    )
    if orientation in {"towards_camera", "away_from_camera"}:
        # Open3D's local normal signs are otherwise arbitrary.  Establish the
        # same camera-facing convention first, then invert it when explicitly
        # requested; simply negating an unoriented estimate is not stable.
        cloud.orient_normals_towards_camera_location(np.zeros(3))
        if orientation == "away_from_camera":
            cloud.normals = o3d.utility.Vector3dVector(-np.asarray(cloud.normals))
    elif orientation != "none":
        raise ValueError(f"unsupported normal orientation: {orientation}")


def _normalize_cloud_normals(cloud: object) -> None:
    if not cloud.has_normals():
        return
    normals = np.asarray(cloud.normals, dtype=float)
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 1e-12) or not np.all(np.isfinite(normals)):
        raise ValueError("point-cloud normals must be finite and non-zero")
    cloud.normals = o3d.utility.Vector3dVector(normals / lengths[:, None])


def _compute_fpfh(cloud: object, config: DetectionConfig) -> object:
    radius = config.voxel_size_m * config.feature_radius_factor
    return o3d.pipelines.registration.compute_fpfh_feature(
        cloud,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=config.feature_neighbors,
        ),
    )


def _remove_support_plane(cloud: object, config: DetectionConfig) -> tuple[object, dict[str, Any]]:
    info: dict[str, Any] = {
        "enabled": bool(config.remove_support_plane),
        "attempted": False,
        "removed": False,
        "distance_threshold_m": config.support_plane_distance_m,
        "minimum_inlier_fraction": config.support_plane_min_ratio,
        "iterations": config.support_plane_iterations,
        "input_points": len(cloud.points),
        "inlier_count": 0,
        "inlier_fraction": 0.0,
        "plane_equation": None,
        "reason": "disabled",
    }
    if not config.remove_support_plane:
        return cloud, info
    if len(cloud.points) < 30:
        info["reason"] = "too_few_points"
        return cloud, info
    info["attempted"] = True
    plane, inliers = cloud.segment_plane(
        distance_threshold=config.support_plane_distance_m,
        ransac_n=3,
        num_iterations=config.support_plane_iterations,
    )
    inlier_fraction = len(inliers) / max(len(cloud.points), 1)
    equation = np.asarray(plane, dtype=float)
    norm = float(np.linalg.norm(equation[:3]))
    if norm > 1e-12:
        equation = equation / norm
    info.update(
        {
            "inlier_count": len(inliers),
            "inlier_fraction": inlier_fraction,
            "plane_equation": equation.tolist(),
        }
    )
    if inlier_fraction < config.support_plane_min_ratio:
        info["reason"] = "inlier_fraction_below_threshold"
        return cloud, info
    info["removed"] = True
    info["reason"] = "removed"
    return cloud.select_by_index(inliers, invert=True), info


def _remove_statistical_outliers(cloud: object, config: DetectionConfig) -> object:
    if not config.remove_statistical_outliers or len(cloud.points) <= config.outlier_neighbors:
        return cloud
    filtered, _ = cloud.remove_statistical_outlier(
        nb_neighbors=config.outlier_neighbors,
        std_ratio=config.outlier_std_ratio,
    )
    return filtered


def _candidate_extent(points: np.ndarray) -> list[float]:
    return np.ptp(np.asarray(points, dtype=float), axis=0).tolist() if len(points) else [0.0, 0.0, 0.0]


def _scene_candidates(
    scene: np.ndarray,
    config: DetectionConfig,
) -> tuple[list[tuple[np.ndarray, str, int, dict[str, Any]]], dict[str, Any]]:
    """Generate full-scene or geometry-only candidates without semantic masks."""

    _require_open3d()
    input_cloud = _to_point_cloud(scene)
    filtered_cloud, plane_info = _remove_support_plane(input_cloud, config)
    after_plane = _cloud_points(filtered_cloud)
    filtered_cloud = _remove_statistical_outliers(filtered_cloud, config)
    filtered = _cloud_points(filtered_cloud)
    generation: dict[str, Any] = {
        "mode": config.candidate_mode,
        "input_points": len(scene),
        "after_support_plane_points": len(after_plane),
        "after_statistical_outlier_points": len(filtered),
        "support_plane": plane_info,
        "cluster_eps_m": config.cluster_eps_m,
        "cluster_min_points": config.cluster_min_points,
        "min_candidate_points": config.min_candidate_points,
        "max_candidate_extent_m": config.max_candidate_extent_m,
        "clusters_before_filtering": 0,
        "clusters": [],
        "noise_points": 0,
        "candidate_count": 0,
    }
    if len(filtered) < 3:
        generation["reason"] = "too_few_filtered_points"
        return [], generation

    candidates: list[tuple[np.ndarray, str, int, dict[str, Any]]] = []
    if config.candidate_mode in {"full", "full_and_dbscan"}:
        candidates.append(
            (
                filtered,
                "full",
                0,
                {"candidate_points": len(filtered), "extent_m": _candidate_extent(filtered)},
            )
        )

    cluster_modes = {"dbscan", "full_and_dbscan", "geometry_candidates"}
    if config.candidate_mode in cluster_modes:
        labels = np.asarray(
            filtered_cloud.cluster_dbscan(
                eps=config.cluster_eps_m,
                min_points=config.cluster_min_points,
                print_progress=False,
            )
        )
        labels_unique = sorted(int(value) for value in np.unique(labels) if value >= 0)
        generation["clusters_before_filtering"] = len(labels_unique)
        generation["noise_points"] = int(np.sum(labels < 0))
        cluster_index = 0
        for label in labels_unique:
            cluster = filtered[labels == label]
            extent = np.ptp(cluster, axis=0)
            rejection: str | None = None
            minimum_points = config.min_candidate_points if config.candidate_mode == "geometry_candidates" else 3
            if len(cluster) < minimum_points:
                rejection = "below_min_candidate_points"
            elif config.candidate_mode == "geometry_candidates" and float(np.max(extent)) > config.max_candidate_extent_m:
                rejection = "exceeds_max_candidate_extent"
            cluster_info: dict[str, Any] = {
                "label": label,
                "points": len(cluster),
                "extent_m": extent.tolist(),
                "centroid_m": np.mean(cluster, axis=0).tolist(),
                "accepted": rejection is None,
                "rejection_reason": rejection,
            }
            generation["clusters"].append(cluster_info)
            if rejection is None:
                source = "geometry_cluster" if config.candidate_mode == "geometry_candidates" else "dbscan"
                candidates.append(
                    (
                        cluster,
                        source,
                        cluster_index,
                        {"cluster_label": label, **cluster_info},
                    )
                )
                cluster_index += 1
    generation["candidate_count"] = len(candidates)
    if not candidates:
        generation["reason"] = "no_plausible_candidates"
    return candidates, generation


def _feature_correspondence_count(source_feature: object, target_feature: object) -> int:
    """Count Open3D feature matches before RANSAC pruning.

    Open3D may fall back from mutual matching to the original correspondence
    set when too few mutual pairs remain. The total non-mutual set is a stable
    denominator for diagnostics in either case.
    """

    registration = o3d.pipelines.registration
    correspondences_from_features = getattr(registration, "correspondences_from_features", None)
    if correspondences_from_features is None:
        return 0
    try:
        return len(correspondences_from_features(source_feature, target_feature, False))
    except TypeError:
        return len(correspondences_from_features(source_feature, target_feature))


def _nearest_neighbor_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute finite nearest-neighbor distances with bounded memory."""

    source_cloud = _to_point_cloud(source)
    target_cloud = _to_point_cloud(target)
    tree = o3d.geometry.KDTreeFlann(target_cloud)
    distances: list[float] = []
    for point in np.asarray(source, dtype=float):
        count, _, squared = tree.search_knn_vector_3d(point, 1)
        if count != 1 or not squared:
            distances.append(float("inf"))
        else:
            distances.append(math.sqrt(float(squared[0])))
    return np.asarray(distances, dtype=float)


def _trimmed_rmse(distances: np.ndarray, keep_fraction: float = 0.9) -> float:
    finite = np.asarray(distances, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("inf")
    keep = max(1, int(math.ceil(len(finite) * keep_fraction)))
    selected = np.sort(finite)[:keep]
    return float(math.sqrt(np.mean(selected * selected)))


def compute_bidirectional_verification(
    model_points: np.ndarray | Sequence[Sequence[float]],
    candidate_points: np.ndarray | Sequence[Sequence[float]],
    transform: np.ndarray,
    *,
    distance_threshold_m: float = 0.02,
    max_points: int = 2500,
) -> GeometricVerification:
    """Measure two-sided geometric agreement after applying ``transform``.

    The transform is ``T_camera_object``/``T_target_source``.  The model and
    candidate are voxel-reduced and deterministically capped only to bound
    verification cost.  The returned overlaps are geometric fractions, not
    confidence or probability values.
    """

    if distance_threshold_m <= 0.0 or max_points < 3:
        raise ValueError("verification threshold must be positive and max_points must be at least three")
    model = validate_point_cloud(model_points, "model_points")
    candidate = validate_point_cloud(candidate_points, "candidate_points")
    model_cloud = _limit_points(_to_point_cloud(model), max_points)
    candidate_cloud = _limit_points(_to_point_cloud(candidate), max_points)
    model_sample = transform_points(transform, _cloud_points(model_cloud))
    candidate_sample = _cloud_points(candidate_cloud)
    model_distances = _nearest_neighbor_distances(model_sample, candidate_sample)
    candidate_distances = _nearest_neighbor_distances(candidate_sample, model_sample)
    model_extent = np.ptp(model_sample, axis=0)
    candidate_extent = np.ptp(candidate_sample, axis=0)
    extent_ratio = candidate_extent / np.maximum(model_extent, 1e-12)
    return GeometricVerification(
        distance_threshold_m=float(distance_threshold_m),
        model_to_candidate_overlap=float(np.mean(model_distances <= distance_threshold_m)),
        candidate_to_model_overlap=float(np.mean(candidate_distances <= distance_threshold_m)),
        bidirectional_overlap=float(
            min(
                np.mean(model_distances <= distance_threshold_m),
                np.mean(candidate_distances <= distance_threshold_m),
            )
        ),
        model_to_candidate_rmse_m=float(math.sqrt(np.mean(model_distances**2))),
        candidate_to_model_rmse_m=float(math.sqrt(np.mean(candidate_distances**2))),
        model_to_candidate_trimmed_rmse_m=_trimmed_rmse(model_distances),
        candidate_to_model_trimmed_rmse_m=_trimmed_rmse(candidate_distances),
        extent_ratio=extent_ratio,
    )


def _registration_failure(
    model: ModelCatalogEntry,
    scene_count: int,
    config: DetectionConfig,
    reason: str,
    *,
    candidate_points: int = 0,
    candidate_source: str = "none",
    candidate_index: int = -1,
    feature_correspondences: int = 0,
    ransac_inliers: int = 0,
    icp_inliers: int = 0,
    inlier_ratio: float = 0.0,
    global_fitness: float = 0.0,
    fitness: float = 0.0,
    ransac_rmse_m: float | None = None,
    rmse_m: float | None = None,
    transform: np.ndarray | None = None,
    icp_iterations: int = 0,
    camera_frame: str = "camera",
    world_frame: str | None = None,
    timestamp: float | None = None,
    model_representation: str = "custom",
    model_normal_source: str = "estimate",
    observed_normal_orientation: str = "none",
    verification: GeometricVerification | None = None,
    candidate_generation: dict[str, Any] | None = None,
) -> PoseHypothesis:
    diagnostics = RegistrationDiagnostics(
        method=METHOD_NAME,
        accepted=False,
        failure_reason=reason,
        model_points=len(model.points),
        scene_points=scene_count,
        candidate_points=candidate_points,
        candidate_source=candidate_source,
        candidate_index=candidate_index,
        feature_correspondences=feature_correspondences,
        ransac_inliers=ransac_inliers,
        icp_inliers=icp_inliers,
        inlier_ratio=float(np.clip(inlier_ratio, 0.0, 1.0)),
        global_fitness=float(np.clip(global_fitness, 0.0, 1.0)),
        fitness=float(np.clip(fitness, 0.0, 1.0)),
        ransac_rmse_m=ransac_rmse_m,
        rmse_m=rmse_m,
        icp_iterations=icp_iterations,
        downsample_voxel_m=config.voxel_size_m,
        model_representation=model_representation,
        model_normal_source=model_normal_source,
        observed_normal_orientation=observed_normal_orientation,
        verification_mode=config.verification_mode,
        model_to_candidate_overlap=None
        if verification is None
        else verification.model_to_candidate_overlap,
        candidate_to_model_overlap=None
        if verification is None
        else verification.candidate_to_model_overlap,
        bidirectional_overlap=None if verification is None else verification.bidirectional_overlap,
        model_to_candidate_rmse_m=None
        if verification is None
        else verification.model_to_candidate_rmse_m,
        candidate_to_model_rmse_m=None
        if verification is None
        else verification.candidate_to_model_rmse_m,
        model_to_candidate_trimmed_rmse_m=None
        if verification is None
        else verification.model_to_candidate_trimmed_rmse_m,
        candidate_to_model_trimmed_rmse_m=None
        if verification is None
        else verification.candidate_to_model_trimmed_rmse_m,
        extent_ratio_min=None if verification is None else float(np.min(verification.extent_ratio)),
        extent_ratio_max=None if verification is None else float(np.max(verification.extent_ratio)),
        candidate_generation=candidate_generation or {},
    )
    confidence = float(np.clip(0.35 * fitness + 0.35 * global_fitness + 0.3 * inlier_ratio, 0.0, 1.0))
    return PoseHypothesis(
        model_id=model.model_id,
        semantic_label=model.semantic_label,
        T_camera_object=transform,
        T_world_object=None,
        camera_frame=camera_frame,
        world_frame=world_frame,
        timestamp=timestamp,
        confidence=confidence,
        accepted=False,
        diagnostics=diagnostics,
        model_bounds_m=model.bounds_m,
    )


def _set_open3d_seed(seed: int) -> None:
    random_module = getattr(getattr(o3d, "utility", None), "random", None)
    seed_function = getattr(random_module, "seed", None)
    if seed_function is not None:
        seed_function(seed)


def _register_candidate(
    model: ModelCatalogEntry,
    candidate: np.ndarray,
    scene_count: int,
    candidate_source: str,
    candidate_index: int,
    config: DetectionConfig,
    *,
    candidate_generation: dict[str, Any] | None = None,
    camera_frame: str,
    world_frame: str | None,
    timestamp: float | None,
) -> PoseHypothesis:
    open3d = _require_open3d()
    model_points = validate_point_cloud(model.points, "model.points")
    candidate = validate_point_cloud(candidate, "scene candidate")
    use_model_normals = config.model_normal_source == "mesh" or (
        config.model_normal_source == "auto" and model.normals is not None
    )
    actual_model_normal_source = "mesh" if use_model_normals else "estimate"
    model_down = _voxel_cloud(
        model_points,
        config,
        normals=model.normals if use_model_normals else None,
    )
    scene_down = _voxel_cloud(candidate, config)
    if len(model_down.points) < 3 or len(scene_down.points) < 3:
        return _registration_failure(
            model,
            scene_count,
            config,
            "too_few_points_after_downsampling",
            candidate_points=len(candidate),
            candidate_source=candidate_source,
            candidate_index=candidate_index,
            camera_frame=camera_frame,
            world_frame=world_frame,
            timestamp=timestamp,
            model_representation=config.model_representation,
            model_normal_source=actual_model_normal_source,
            observed_normal_orientation=config.observed_normal_orientation,
            candidate_generation=candidate_generation,
        )
    try:
        if model_down.has_normals():
            _normalize_cloud_normals(model_down)
        else:
            _estimate_normals(model_down, config, config.normal_radius_factor)
        _estimate_normals(
            scene_down,
            config,
            config.normal_radius_factor,
            orientation=config.observed_normal_orientation,
        )
        model_feature = _compute_fpfh(model_down, config)
        scene_feature = _compute_fpfh(scene_down, config)
        feature_count = _feature_correspondence_count(model_feature, scene_feature)
        _set_open3d_seed(config.random_seed)
        registration = open3d.pipelines.registration
        distance = config.ransac_inlier_threshold_m
        global_result = registration.registration_ransac_based_on_feature_matching(
            model_down,
            scene_down,
            model_feature,
            scene_feature,
            True,
            distance,
            registration.TransformationEstimationPointToPoint(False),
            3,
            [
                registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                registration.CorrespondenceCheckerBasedOnDistance(distance),
            ],
            registration.RANSACConvergenceCriteria(
                config.ransac_iterations,
                config.ransac_confidence,
            ),
        )
    except (RuntimeError, ValueError) as exc:
        return _registration_failure(
            model,
            scene_count,
            config,
            f"global_registration_failed: {exc}",
            candidate_points=len(candidate),
            candidate_source=candidate_source,
            candidate_index=candidate_index,
            camera_frame=camera_frame,
            world_frame=world_frame,
            timestamp=timestamp,
            model_representation=config.model_representation,
            model_normal_source=actual_model_normal_source,
            observed_normal_orientation=config.observed_normal_orientation,
            candidate_generation=candidate_generation,
        )
    ransac_inliers = len(global_result.correspondence_set)
    global_fitness = float(global_result.fitness)
    ransac_rmse = float(global_result.inlier_rmse) if ransac_inliers else None
    if feature_count == 0:
        feature_count = ransac_inliers
    inlier_ratio = ransac_inliers / max(feature_count, 1)
    if ransac_inliers < config.min_ransac_inliers:
        return _registration_failure(
            model,
            scene_count,
            config,
            "ransac_consensus_below_threshold",
            candidate_points=len(candidate),
            candidate_source=candidate_source,
            candidate_index=candidate_index,
            feature_correspondences=feature_count,
            ransac_inliers=ransac_inliers,
            inlier_ratio=inlier_ratio,
            global_fitness=global_fitness,
            ransac_rmse_m=ransac_rmse,
            transform=np.asarray(global_result.transformation),
            camera_frame=camera_frame,
            world_frame=world_frame,
            timestamp=timestamp,
            model_representation=config.model_representation,
            model_normal_source=actual_model_normal_source,
            observed_normal_orientation=config.observed_normal_orientation,
            candidate_generation=candidate_generation,
        )

    model_refine = _limit_points(
        _to_point_cloud(model_points, model.normals if use_model_normals else None),
        config.max_registration_points,
    )
    scene_refine = _limit_points(_to_point_cloud(candidate), config.max_registration_points)
    try:
        _estimate_normals(
            scene_refine,
            config,
            config.normal_radius_factor,
            orientation=config.observed_normal_orientation,
        )
        icp_result = registration.registration_icp(
            model_refine,
            scene_refine,
            config.icp_max_correspondence_m,
            global_result.transformation,
            registration.TransformationEstimationPointToPlane(),
            registration.ICPConvergenceCriteria(max_iteration=config.icp_iterations),
        )
    except (RuntimeError, ValueError) as exc:
        return _registration_failure(
            model,
            scene_count,
            config,
            f"icp_failed: {exc}",
            candidate_points=len(candidate),
            candidate_source=candidate_source,
            candidate_index=candidate_index,
            feature_correspondences=feature_count,
            ransac_inliers=ransac_inliers,
            inlier_ratio=inlier_ratio,
            global_fitness=global_fitness,
            ransac_rmse_m=ransac_rmse,
            transform=np.asarray(global_result.transformation),
            camera_frame=camera_frame,
            world_frame=world_frame,
            timestamp=timestamp,
            model_representation=config.model_representation,
            model_normal_source=actual_model_normal_source,
            observed_normal_orientation=config.observed_normal_orientation,
            candidate_generation=candidate_generation,
        )
    icp_inliers = len(icp_result.correspondence_set)
    fitness = float(icp_result.fitness)
    rmse = float(icp_result.inlier_rmse) if icp_inliers else None
    verification: GeometricVerification | None = None
    if config.verification_mode == "bidirectional":
        try:
            verification = compute_bidirectional_verification(
                model_points,
                candidate,
                np.asarray(icp_result.transformation),
                distance_threshold_m=config.verification_distance_m,
                max_points=config.verification_max_points,
            )
        except (RuntimeError, ValueError) as exc:
            return _registration_failure(
                model,
                scene_count,
                config,
                f"verification_failed: {exc}",
                candidate_points=len(candidate),
                candidate_source=candidate_source,
                candidate_index=candidate_index,
                feature_correspondences=feature_count,
                ransac_inliers=ransac_inliers,
                icp_inliers=icp_inliers,
                inlier_ratio=inlier_ratio,
                global_fitness=global_fitness,
                fitness=fitness,
                ransac_rmse_m=ransac_rmse,
                rmse_m=rmse,
                transform=np.asarray(icp_result.transformation),
                camera_frame=camera_frame,
                world_frame=world_frame,
                timestamp=timestamp,
                model_representation=config.model_representation,
                model_normal_source=actual_model_normal_source,
                observed_normal_orientation=config.observed_normal_orientation,
                candidate_generation=candidate_generation,
            )
    failure_reason: str | None = None
    if icp_inliers < config.min_icp_inliers:
        failure_reason = "icp_inliers_below_threshold"
    elif fitness < config.min_fitness:
        failure_reason = "icp_fitness_below_threshold"
    elif rmse is None or rmse > config.max_rmse_m:
        failure_reason = "icp_rmse_above_threshold"
    elif verification is not None and (
        verification.model_to_candidate_overlap < config.min_bidirectional_overlap
    ):
        failure_reason = "model_to_candidate_overlap_below_threshold"
    elif verification is not None and (
        verification.candidate_to_model_overlap < config.min_bidirectional_overlap
    ):
        failure_reason = "candidate_to_model_overlap_below_threshold"
    # The two overlap gates are the conservative acceptance evidence.  Raw
    # nearest-neighbour RMSE is retained above as a diagnostic, but using it as
    # a hard gate here would reject otherwise valid candidates containing
    # clutter/outliers.  The candidate->model overlap still requires that a
    # meaningful fraction of the observed candidate be explained by the
    # model, preventing a small model from accepting a huge scene.
    accepted = failure_reason is None
    transform = np.asarray(icp_result.transformation, dtype=float)
    rmse_score = 0.0 if rmse is None else math.exp(-rmse / max(config.max_rmse_m, 1e-12))
    verification_score = 0.0 if verification is None else verification.bidirectional_overlap
    confidence = float(
        np.clip(
            0.25 * fitness
            + 0.20 * global_fitness
            + 0.15 * min(inlier_ratio, 1.0)
            + 0.20 * rmse_score
            + 0.20 * verification_score,
            0.0,
            1.0,
        )
    )
    diagnostics = RegistrationDiagnostics(
        method=METHOD_NAME,
        accepted=accepted,
        failure_reason=failure_reason,
        model_points=len(model.points),
        scene_points=scene_count,
        candidate_points=len(candidate),
        candidate_source=candidate_source,
        candidate_index=candidate_index,
        feature_correspondences=feature_count,
        ransac_inliers=ransac_inliers,
        icp_inliers=icp_inliers,
        inlier_ratio=float(np.clip(inlier_ratio, 0.0, 1.0)),
        global_fitness=float(np.clip(global_fitness, 0.0, 1.0)),
        fitness=float(np.clip(fitness, 0.0, 1.0)),
        ransac_rmse_m=ransac_rmse,
        rmse_m=rmse,
        icp_iterations=config.icp_iterations,
        downsample_voxel_m=config.voxel_size_m,
        model_representation=config.model_representation,
        model_normal_source=actual_model_normal_source,
        observed_normal_orientation=config.observed_normal_orientation,
        verification_mode=config.verification_mode,
        model_to_candidate_overlap=None
        if verification is None
        else verification.model_to_candidate_overlap,
        candidate_to_model_overlap=None
        if verification is None
        else verification.candidate_to_model_overlap,
        bidirectional_overlap=None if verification is None else verification.bidirectional_overlap,
        model_to_candidate_rmse_m=None
        if verification is None
        else verification.model_to_candidate_rmse_m,
        candidate_to_model_rmse_m=None
        if verification is None
        else verification.candidate_to_model_rmse_m,
        model_to_candidate_trimmed_rmse_m=None
        if verification is None
        else verification.model_to_candidate_trimmed_rmse_m,
        candidate_to_model_trimmed_rmse_m=None
        if verification is None
        else verification.candidate_to_model_trimmed_rmse_m,
        extent_ratio_min=None if verification is None else float(np.min(verification.extent_ratio)),
        extent_ratio_max=None if verification is None else float(np.max(verification.extent_ratio)),
        candidate_generation=candidate_generation or {},
    )
    return PoseHypothesis(
        model_id=model.model_id,
        semantic_label=model.semantic_label,
        T_camera_object=transform,
        T_world_object=None,
        camera_frame=camera_frame,
        world_frame=world_frame,
        timestamp=timestamp,
        confidence=confidence,
        accepted=accepted,
        diagnostics=diagnostics,
        model_bounds_m=model.bounds_m,
    )


def register_model_to_scene(
    model: ModelCatalogEntry,
    scene_points: np.ndarray | Sequence[Sequence[float]],
    *,
    config: DetectionConfig | None = None,
    T_world_camera: np.ndarray | None = None,
    camera_frame: str = "camera",
    world_frame: str | None = None,
    timestamp: float | None = None,
) -> PoseHypothesis:
    """Find one known model instance in a camera-frame scene cloud.

    The scene is not assumed to be an object crop. It is optionally filtered
    for a dominant support plane/statistical outliers and split into a full
    scene candidate plus DBSCAN clusters. The best accepted candidate is
    returned; this API intentionally emits at most one instance per model.
    """

    cfg = config or DetectionConfig()
    validate_point_cloud(model.points, "model.points")
    scene = validate_point_cloud(scene_points, "scene_points")
    if T_world_camera is not None:
        world_camera = validate_world_camera_transform(T_world_camera)
        if world_frame is None:
            raise ValueError("world_frame is required when T_world_camera is supplied")
    else:
        world_camera = None
    _require_open3d()
    _set_open3d_seed(cfg.random_seed)
    candidates, candidate_generation = _scene_candidates(scene, cfg)
    if not candidates:
        return _registration_failure(
            model,
            len(scene),
            cfg,
            "no_scene_candidates",
            camera_frame=camera_frame,
            world_frame=world_frame,
            timestamp=timestamp,
            model_representation=cfg.model_representation,
            model_normal_source=(
                "mesh"
                if cfg.model_normal_source == "mesh"
                or (cfg.model_normal_source == "auto" and model.normals is not None)
                else "estimate"
            ),
            observed_normal_orientation=cfg.observed_normal_orientation,
            candidate_generation=candidate_generation,
        )
    hypotheses = [
        _register_candidate(
            model,
            candidate,
            len(scene),
            source,
            index,
            cfg,
            candidate_generation={**candidate_generation, "selected_candidate": candidate_info},
            camera_frame=camera_frame,
            world_frame=world_frame,
            timestamp=timestamp,
        )
        for candidate, source, index, candidate_info in candidates
    ]
    hypotheses.sort(
        key=lambda hypothesis: (
            hypothesis.accepted,
            hypothesis.diagnostics.bidirectional_overlap
            if hypothesis.diagnostics.bidirectional_overlap is not None
            else 0.0,
            hypothesis.diagnostics.fitness,
            hypothesis.diagnostics.global_fitness,
            hypothesis.diagnostics.inlier_ratio,
            hypothesis.confidence,
        ),
        reverse=True,
    )
    best = hypotheses[0]
    if best.accepted and world_camera is not None and best.T_camera_object is not None:
        return PoseHypothesis(
            model_id=best.model_id,
            semantic_label=best.semantic_label,
            T_camera_object=best.T_camera_object,
            T_world_object=world_camera @ best.T_camera_object,
            camera_frame=best.camera_frame,
            world_frame=world_frame,
            timestamp=best.timestamp,
            confidence=best.confidence,
            accepted=True,
            diagnostics=best.diagnostics,
            model_bounds_m=best.model_bounds_m,
        )
    return best


def _rotation_distance_degrees(first: np.ndarray, second: np.ndarray) -> float:
    relative = first[:3, :3].T @ second[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


class GeometryFirstDetector:
    """Run Open3D model-based registration against a catalogue."""

    def __init__(
        self,
        catalogue: Iterable[ModelCatalogEntry],
        *,
        config: DetectionConfig | None = None,
    ) -> None:
        self.catalogue = tuple(catalogue)
        if not self.catalogue:
            raise ValueError("catalogue must contain at least one model")
        model_ids = [model.model_id for model in self.catalogue]
        if len(set(model_ids)) != len(model_ids):
            raise ValueError("catalogue model_id values must be unique")
        self.config = config or DetectionConfig()

    def detect(
        self,
        scene_points: np.ndarray | Sequence[Sequence[float]],
        *,
        T_world_camera: np.ndarray | None = None,
        camera_frame: str = "camera",
        world_frame: str | None = None,
        timestamp: float | None = None,
    ) -> list[PoseHypothesis]:
        hypotheses = [
            register_model_to_scene(
                model,
                scene_points,
                config=self.config,
                T_world_camera=T_world_camera,
                camera_frame=camera_frame,
                world_frame=world_frame,
                timestamp=timestamp,
            )
            for model in self.catalogue
        ]
        accepted = [hypothesis for hypothesis in hypotheses if hypothesis.accepted]
        accepted.sort(key=lambda item: item.confidence, reverse=True)
        kept: list[PoseHypothesis] = []
        for hypothesis in accepted:
            assert hypothesis.T_camera_object is not None
            is_duplicate = any(
                np.linalg.norm(hypothesis.T_camera_object[:3, 3] - other.T_camera_object[:3, 3])
                <= self.config.dedup_translation_m
                and _rotation_distance_degrees(hypothesis.T_camera_object, other.T_camera_object)
                <= self.config.dedup_rotation_deg
                for other in kept
            )
            if not is_duplicate:
                kept.append(hypothesis)
        return kept + [item for item in hypotheses if not item.accepted]
