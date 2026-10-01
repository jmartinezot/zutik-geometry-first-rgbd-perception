#!/usr/bin/env python3
"""Run the frozen-set, post-hoc T-LESS registration architecture diagnosis.

This script is intentionally an experiment adapter, not a production detector.
It reads the immutable v1 manifest/results, uses GT only to construct oracle
clouds or evaluate hypotheses, and writes all new material below
``experiments/tless_registration_diagnosis_v2``.

The inference configuration is copied from the frozen B geometry variant.  No
per-case parameter selection is performed.  The three bakeoff variants are:

* Open3D feature-RANSAC + point-to-plane ICP (current production initializer);
* Open3D Fast Global Registration + point-to-plane ICP;
* Open3D feature-RANSAC + Generalized ICP.

All distances and transforms inside this script are metres and
``T_camera_object`` unless a field is explicitly labelled millimetres or
pixels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import open3d as o3d
except ModuleNotFoundError as exc:  # pragma: no cover - experiment dependency
    raise SystemExit("Open3D is required; install the project environment first") from exc

from zutik_geometry.bop import bop_pose_to_transform
from zutik_geometry.evaluation import (
    classify_acceptance,
    diagnostic_accuracy,
    error_bands,
    rotation_error_degrees,
    translation_error_m,
)
from zutik_geometry.io import _load_ply, load_model_catalog_entry, load_point_cloud
from zutik_geometry.registration import (
    _compute_fpfh,
    _estimate_normals,
    _feature_correspondence_count,
    _limit_points,
    _nearest_neighbor_distances,
    _scene_candidates,
    _to_point_cloud,
    _voxel_cloud,
    compute_bidirectional_verification,
    depth_to_point_cloud,
    transform_points,
)
from zutik_geometry.types import CameraIntrinsics, DetectionConfig, ModelCatalogEntry


DEFAULT_ROOT = Path("/home/datasets/bop/tless")
DEFAULT_MANIFEST = Path("experiments/tless_validation_manifest_scene_conditioned_v1.json")
DEFAULT_V1 = Path("experiments/tless_validation_v1")
DEFAULT_OUTPUT = Path("experiments/tless_registration_diagnosis_v2")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def to_json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_value(v) for v in value]
    return value


def pose_metrics(estimate: np.ndarray | None, gt: np.ndarray) -> dict[str, float | None]:
    if estimate is None:
        return {
            "translation_error_m": None,
            "translation_error_mm": None,
            "rotation_error_deg": None,
        }
    te = translation_error_m(estimate, gt)
    re = rotation_error_degrees(estimate, gt)
    return {
        "translation_error_m": te,
        "translation_error_mm": te * 1000.0,
        "rotation_error_deg": re,
    }


def config_for_representation(representation: str) -> DetectionConfig:
    """Exact frozen B geometry settings, with only the representation changed."""

    return DetectionConfig(
        voxel_size_m=0.005,
        normal_radius_factor=2.0,
        normal_neighbors=30,
        feature_radius_factor=5.0,
        feature_neighbors=100,
        max_registration_points=2500,
        ransac_iterations=20000,
        ransac_confidence=0.999,
        ransac_inlier_threshold_m=0.015,
        icp_iterations=35,
        icp_max_correspondence_m=0.02,
        min_ransac_inliers=6,
        min_icp_inliers=12,
        min_fitness=0.20,
        max_rmse_m=0.02,
        random_seed=7,
        candidate_mode="geometry_candidates",
        model_representation=representation,
        model_normal_source="auto",
        observed_normal_orientation="towards_camera",
        verification_mode="bidirectional",
        verification_distance_m=0.02,
        min_bidirectional_overlap=0.20,
        verification_max_points=2500,
        min_candidate_points=30,
        max_candidate_extent_m=0.75,
        cluster_eps_m=0.04,
        cluster_min_points=20,
        remove_support_plane=True,
        support_plane_distance_m=0.008,
        support_plane_min_ratio=0.45,
        support_plane_iterations=1000,
        remove_statistical_outliers=True,
        outlier_neighbors=20,
        outlier_std_ratio=2.0,
    )


def load_case_frame(root: Path, case: dict[str, Any]) -> dict[str, Any]:
    split = "test_primesense"
    scene_id = int(case["scene_id"])
    image_id = int(case["image_id"])
    scene_root = root / split / f"{scene_id:06d}"
    camera_all = read_json(scene_root / "scene_camera.json")
    camera = camera_all.get(str(image_id), camera_all.get(image_id))
    if not isinstance(camera, dict):
        raise ValueError(f"no camera entry for scene {scene_id}, image {image_id}")
    K = np.asarray(camera["cam_K"], dtype=float).reshape(3, 3)
    intrinsics = CameraIntrinsics.from_matrix(K)
    depth_path = scene_root / "depth" / f"{image_id:06d}.png"
    rgb_path = scene_root / "rgb" / f"{image_id:06d}.png"
    mask_path = scene_root / "mask_visib" / f"{image_id:06d}_{int(case['gt_instance_index']):06d}.png"
    from PIL import Image

    with Image.open(depth_path) as image:
        depth = np.asarray(image)
    with Image.open(mask_path) as image:
        visible_mask = np.asarray(image) > 0
    depth_scale_m = float(camera["depth_scale"]) * 1e-3
    depth_m = depth.astype(float) * depth_scale_m
    valid = np.isfinite(depth_m) & (depth_m >= 0.05) & (depth_m <= 10.0)
    full_scene = depth_to_point_cloud(
        depth,
        intrinsics,
        depth_scale_m=depth_scale_m,
        min_depth_m=0.05,
        max_depth_m=10.0,
    )
    target_valid = valid & visible_mask
    v, u = np.nonzero(target_valid)
    z = depth_m[v, u]
    target = np.column_stack(
        (
            (u.astype(float) - intrinsics.cx) * z / intrinsics.fx,
            (v.astype(float) - intrinsics.cy) * z / intrinsics.fy,
            z,
        )
    )
    gt_entries = read_json(scene_root / "scene_gt.json")[str(image_id)]
    gt_index = int(case["gt_instance_index"])
    gt_entry = gt_entries[gt_index]
    if int(gt_entry["obj_id"]) != int(case["model_id"]):
        raise ValueError(f"manifest GT index does not contain model {case['model_id']}: {gt_entry}")
    gt = bop_pose_to_transform(gt_entry["cam_R_m2c"], gt_entry["cam_t_m2c"])
    info_entries = read_json(scene_root / "scene_gt_info.json")[str(image_id)]
    info = info_entries[gt_index]
    return {
        "scene_id": scene_id,
        "image_id": image_id,
        "scene_root": scene_root,
        "rgb_path": rgb_path,
        "depth_path": depth_path,
        "mask_path": mask_path,
        "intrinsics": intrinsics,
        "K": K,
        "depth_scale_m": depth_scale_m,
        "depth": depth,
        "full_scene": full_scene,
        "oracle_target": target,
        "visible_mask": visible_mask,
        "gt": gt,
        "gt_entry": gt_entry,
        "gt_info": info,
    }


def point_stats(points: np.ndarray, normals: np.ndarray | None = None) -> dict[str, Any]:
    points = np.asarray(points, dtype=float)
    cloud = _to_point_cloud(points, normals)
    distances = np.asarray(cloud.compute_nearest_neighbor_distance(), dtype=float) if len(points) > 1 else np.array([])
    result: dict[str, Any] = {
        "count": int(len(points)),
        "centroid_m": np.mean(points, axis=0).tolist(),
        "aabb_min_m": np.min(points, axis=0).tolist(),
        "aabb_max_m": np.max(points, axis=0).tolist(),
        "dimensions_m": np.ptp(points, axis=0).tolist(),
        "point_spacing_m": {
            "median": finite_float(np.percentile(distances, 50)) if len(distances) else None,
            "p75": finite_float(np.percentile(distances, 75)) if len(distances) else None,
            "p95": finite_float(np.percentile(distances, 95)) if len(distances) else None,
            "maximum": finite_float(np.max(distances)) if len(distances) else None,
        },
        "normals_available": bool(normals is not None),
    }
    try:
        obb = cloud.get_oriented_bounding_box()
        result["obb_dimensions_m"] = np.asarray(obb.extent).tolist()
        result["obb_center_m"] = np.asarray(obb.center).tolist()
    except RuntimeError:
        result["obb_dimensions_m"] = None
        result["obb_center_m"] = None
    if normals is not None:
        n = np.asarray(normals, dtype=float)
        result["normal_mean"] = np.mean(n, axis=0).tolist()
        result["normal_norm_min"] = float(np.min(np.linalg.norm(n, axis=1)))
        result["normal_norm_max"] = float(np.max(np.linalg.norm(n, axis=1)))
    return result


def distribution(distances: np.ndarray, thresholds_m: Iterable[float] = ()) -> dict[str, Any]:
    values = np.asarray(distances, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    result: dict[str, Any] = {
        "count": int(len(values)),
        "median_m": float(np.percentile(values, 50)),
        "p75_m": float(np.percentile(values, 75)),
        "p90_m": float(np.percentile(values, 90)),
        "p95_m": float(np.percentile(values, 95)),
        "maximum_m": float(np.max(values)),
        "robust_max_p99_m": float(np.percentile(values, 99)),
    }
    result["coverage"] = {
        f"{int(threshold * 1000)}mm": float(np.mean(values <= threshold))
        for threshold in thresholds_m
    }
    return result


def model_geometry(root: Path, model_id: int, representation: str) -> ModelCatalogEntry:
    path = root / f"models_{representation}" / f"obj_{model_id:06d}.ply"
    return load_model_catalog_entry(path, model_id=str(model_id), units="mm", representation=representation)


def nearest_metrics(model: np.ndarray, target: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    model_camera = transform_points(gt, model)
    return {
        "model_to_observed": distribution(
            _nearest_neighbor_distances(model_camera, target),
            thresholds_m=(0.002, 0.005, 0.010, 0.015, 0.020),
        ),
        "observed_to_model": distribution(
            _nearest_neighbor_distances(target, model_camera),
            thresholds_m=(0.002, 0.005, 0.010, 0.015, 0.020),
        ),
    }


def normal_preparation(
    model: ModelCatalogEntry,
    target: np.ndarray,
    config: DetectionConfig,
    *,
    model_normal_source: str = "auto",
    observed_orientation: str = "towards_camera",
) -> tuple[Any, Any, str]:
    use_model_normals = model_normal_source == "mesh" or (
        model_normal_source == "auto" and model.normals is not None
    )
    source = _voxel_cloud(
        model.points,
        config,
        normals=model.normals if use_model_normals else None,
    )
    destination = _voxel_cloud(target, config)
    if len(source.points) < 3 or len(destination.points) < 3:
        raise ValueError("too few points after voxel downsampling")
    if source.has_normals():
        source_normals = np.asarray(source.normals, dtype=float)
        source.normals = o3d.utility.Vector3dVector(
            source_normals / np.maximum(np.linalg.norm(source_normals, axis=1, keepdims=True), 1e-12)
        )
    else:
        _estimate_normals(source, config, config.normal_radius_factor)
    _estimate_normals(destination, config, config.normal_radius_factor, orientation=observed_orientation)
    return source, destination, "mesh" if use_model_normals else "estimate"


def correspondence_diagnostics(
    source: Any,
    destination: Any,
    source_feature: Any,
    destination_feature: Any,
    gt: np.ndarray,
) -> dict[str, Any]:
    registration = o3d.pipelines.registration
    raw = registration.correspondences_from_features(source_feature, destination_feature, False)
    # Open3D's default mutual filter intentionally falls back to all nearest
    # correspondences when its mutual set is too small.  A strict diagnostic
    # set is obtained with mutual_consistency_ratio=0.0; the default call is
    # retained to expose what the production RANSAC call actually receives.
    strict_mutual = registration.correspondences_from_features(
        source_feature, destination_feature, True, 0.0
    )
    current_effective = registration.correspondences_from_features(
        source_feature, destination_feature, True
    )

    def match_stats(matches: Any) -> dict[str, Any]:
        pairs = np.asarray(matches, dtype=int)
        if pairs.size == 0:
            return {"count": 0, "distance": {"count": 0}}
        pairs = pairs.reshape(-1, 2)
        source_points = np.asarray(source.points)[pairs[:, 0]]
        destination_points = np.asarray(destination.points)[pairs[:, 1]]
        distances = np.linalg.norm(transform_points(gt, source_points) - destination_points, axis=1)
        return {
            "count": int(len(pairs)),
            "distance": distribution(distances, thresholds_m=(0.005, 0.010, 0.020)),
        }

    raw_set = {tuple(pair) for pair in np.asarray(raw, dtype=int).reshape(-1, 2)}
    strict_mutual_set = {
        tuple(pair) for pair in np.asarray(strict_mutual, dtype=int).reshape(-1, 2)
    }
    current_effective_set = {
        tuple(pair) for pair in np.asarray(current_effective, dtype=int).reshape(-1, 2)
    }
    return {
        "source_feature_count": int(np.asarray(source_feature.data).shape[1]),
        "destination_feature_count": int(np.asarray(destination_feature.data).shape[1]),
        "raw_nearest_correspondences": match_stats(raw),
        "strict_mutual_correspondences": match_stats(strict_mutual),
        "current_open3d_effective_correspondences": match_stats(current_effective),
        "strict_mutual_ratio_of_raw": float(len(strict_mutual_set) / max(len(raw_set), 1)),
        "current_effective_ratio_of_raw": float(len(current_effective_set) / max(len(raw_set), 1)),
        "non_mutual_count": int(len(raw_set - strict_mutual_set)),
        "open3d_fallback_expected": bool(len(strict_mutual_set) < 0.1 * len(raw_set)),
        "evaluator_note": "GT is used only after feature correspondences are generated.",
    }


def method_review() -> dict[str, Any]:
    return {
        "open3d_version": o3d.__version__,
        "available_api": {
            "feature_ransac": hasattr(o3d.pipelines.registration, "registration_ransac_based_on_feature_matching"),
            "fast_global_registration": hasattr(o3d.pipelines.registration, "registration_fgr_based_on_feature_matching"),
            "generalized_icp": hasattr(o3d.pipelines.registration, "registration_generalized_icp"),
            "point_to_plane_icp": hasattr(o3d.pipelines.registration, "registration_icp"),
        },
        "variants": [
            {
                "name": "ransac_point_to_plane_icp",
                "global_initialization": "Open3D feature-based RANSAC using FPFH correspondences",
                "refinement": "Open3D point-to-plane ICP",
                "tradeoff": "current reference; robust correspondence sampling but expensive and descriptor-dependent",
            },
            {
                "name": "fgr_point_to_plane_icp",
                "global_initialization": "Open3D Fast Global Registration using the same FPFH features",
                "refinement": "Open3D point-to-plane ICP",
                "tradeoff": "faster global optimizer, but still requires useful FPFH correspondences and is sensitive to outliers/partial overlap",
            },
            {
                "name": "ransac_generalized_icp",
                "global_initialization": "Open3D feature-based RANSAC using FPFH correspondences",
                "refinement": "Open3D Generalized ICP",
                "tradeoff": "tests a local refinement that models local surface covariance; does not solve a bad global initializer",
            },
        ],
        "not_selected": [
            "learned features and learned segmentation are outside this geometry-only decision",
            "large new C++/PCL subsystem was not justified in a Python/Open3D repository",
            "Open3D tensor multi-scale ICP was not added to the bakeoff because it would introduce a separate wrapper and a larger comparison surface",
        ],
        "official_references": {
            "open3d_global_registration": "https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html",
            "open3d_icp": "https://www.open3d.org/docs/latest/tutorial/pipelines/icp_registration.html",
            "bop_toolkit": "https://github.com/thodan/bop_toolkit",
        },
    }


def run_global_variant(
    model: ModelCatalogEntry,
    target: np.ndarray,
    gt: np.ndarray,
    config: DetectionConfig,
    method: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "method": method,
        "model_representation": model.representation,
        "voxel_size_m": config.voxel_size_m,
        "global_initializer": method.split("_", 1)[0],
        "refinement": "generalized_icp" if method.endswith("generalized_icp") else "point_to_plane_icp",
        "status": "failed",
    }
    try:
        source, destination, normal_source = normal_preparation(model, target, config)
        source_feature = _compute_fpfh(source, config)
        destination_feature = _compute_fpfh(destination, config)
        feature_info = correspondence_diagnostics(source, destination, source_feature, destination_feature, gt)
        result["model_normal_source"] = normal_source
        result["feature_diagnostics"] = feature_info
        _set_seed(config.random_seed)
        registration = o3d.pipelines.registration
        distance = config.ransac_inlier_threshold_m
        if method.startswith("ransac"):
            global_result = registration.registration_ransac_based_on_feature_matching(
                source,
                destination,
                source_feature,
                destination_feature,
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
        elif method.startswith("fgr"):
            global_result = registration.registration_fgr_based_on_feature_matching(
                source,
                destination,
                source_feature,
                destination_feature,
                registration.FastGlobalRegistrationOption(
                    maximum_correspondence_distance=distance,
                    iteration_number=64,
                ),
            )
        else:
            raise ValueError(f"unknown global method {method}")
        global_transform = np.asarray(global_result.transformation, dtype=float)
        initial_metrics = pose_metrics(global_transform, gt)
        result.update(
            {
                "global_status": "ok",
                "global_fitness": float(global_result.fitness),
                "global_rmse_m": float(global_result.inlier_rmse) if len(global_result.correspondence_set) else None,
                "global_inliers": int(len(global_result.correspondence_set)),
                "global_transform": global_transform.tolist(),
                "global_pose": initial_metrics,
            }
        )
        model_refine = _limit_points(
            _to_point_cloud(model.points, model.normals if normal_source == "mesh" else None),
            config.max_registration_points,
        )
        target_refine = _limit_points(_to_point_cloud(target), config.max_registration_points)
        _estimate_normals(
            target_refine,
            config,
            config.normal_radius_factor,
            orientation=config.observed_normal_orientation,
        )
        if method.endswith("generalized_icp"):
            if not model_refine.has_normals():
                _estimate_normals(model_refine, config, config.normal_radius_factor)
            refined = registration.registration_generalized_icp(
                model_refine,
                target_refine,
                config.icp_max_correspondence_m,
                global_transform,
                registration.TransformationEstimationForGeneralizedICP(),
                registration.ICPConvergenceCriteria(max_iteration=config.icp_iterations),
            )
        else:
            refined = registration.registration_icp(
                model_refine,
                target_refine,
                config.icp_max_correspondence_m,
                global_transform,
                registration.TransformationEstimationPointToPlane(),
                registration.ICPConvergenceCriteria(max_iteration=config.icp_iterations),
            )
        final_transform = np.asarray(refined.transformation, dtype=float)
        final_metrics = pose_metrics(final_transform, gt)
        verification = compute_bidirectional_verification(
            model.points,
            target,
            final_transform,
            distance_threshold_m=config.verification_distance_m,
            max_points=config.verification_max_points,
        )
        accepted = bool(
            len(global_result.correspondence_set) >= config.min_ransac_inliers
            and len(refined.correspondence_set) >= config.min_icp_inliers
            and float(refined.fitness) >= config.min_fitness
            and len(refined.correspondence_set) > 0
            and float(refined.inlier_rmse) <= config.max_rmse_m
            and verification.model_to_candidate_overlap >= config.min_bidirectional_overlap
            and verification.candidate_to_model_overlap >= config.min_bidirectional_overlap
        )
        final_accuracy = diagnostic_accuracy(
            float(final_metrics["translation_error_m"]),
            float(final_metrics["rotation_error_deg"]),
        )
        initial_accuracy = diagnostic_accuracy(
            float(initial_metrics["translation_error_m"]),
            float(initial_metrics["rotation_error_deg"]),
        )
        result.update(
            {
                "status": "ok",
                "final_transform": final_transform.tolist(),
                "final_pose": final_metrics,
                "accepted_by_current_geometry_gates": accepted,
                "final_diagnostic_accurate_non_symmetric": final_accuracy,
                "initial_diagnostic_accurate_non_symmetric": initial_accuracy,
                "refinement_improved_translation": final_metrics["translation_error_m"] < initial_metrics["translation_error_m"],
                "refinement_improved_rotation": final_metrics["rotation_error_deg"] < initial_metrics["rotation_error_deg"],
                "icp_fitness": float(refined.fitness),
                "icp_rmse_m": float(refined.inlier_rmse) if len(refined.correspondence_set) else None,
                "icp_inliers": int(len(refined.correspondence_set)),
                "verification": {
                    "distance_threshold_m": verification.distance_threshold_m,
                    "model_to_candidate_overlap": verification.model_to_candidate_overlap,
                    "candidate_to_model_overlap": verification.candidate_to_model_overlap,
                    "bidirectional_overlap": verification.bidirectional_overlap,
                    "model_to_candidate_rmse_m": verification.model_to_candidate_rmse_m,
                    "candidate_to_model_rmse_m": verification.candidate_to_model_rmse_m,
                    "model_to_candidate_trimmed_rmse_m": verification.model_to_candidate_trimmed_rmse_m,
                    "candidate_to_model_trimmed_rmse_m": verification.candidate_to_model_trimmed_rmse_m,
                    "extent_ratio": verification.extent_ratio.tolist(),
                },
            }
        )
    except Exception as exc:  # diagnostics must retain failures rather than abort the set
        result.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    result["runtime_s"] = time.perf_counter() - started
    return result


def _set_seed(seed: int) -> None:
    random_module = getattr(o3d.utility, "random", None)
    seed_fn = getattr(random_module, "seed", None)
    if seed_fn is not None:
        seed_fn(seed)


def candidate_oracle_metrics(candidate: np.ndarray, target: np.ndarray, threshold_m: float = 0.005) -> dict[str, Any]:
    target_distances = _nearest_neighbor_distances(target, candidate)
    candidate_distances = _nearest_neighbor_distances(candidate, target)
    return {
        "points": int(len(candidate)),
        "extent_m": np.ptp(candidate, axis=0).tolist(),
        "centroid_m": np.mean(candidate, axis=0).tolist(),
        "target_centroid_distance_m": float(np.linalg.norm(np.mean(candidate, axis=0) - np.mean(target, axis=0))),
        "target_recall_at_threshold": float(np.mean(target_distances <= threshold_m)),
        "candidate_purity_at_threshold": float(np.mean(candidate_distances <= threshold_m)),
        "distance_threshold_m": threshold_m,
    }


def run_candidate_analysis(frame: dict[str, Any], config: DetectionConfig) -> dict[str, Any]:
    candidates, generation = _scene_candidates(frame["full_scene"], config)
    target = frame["oracle_target"]
    records = []
    for points, source, index, metadata in candidates:
        records.append(
            {
                "source": source,
                "candidate_index": index,
                "metadata": metadata,
                **candidate_oracle_metrics(points, target),
            }
        )
    best_recall = max(records, key=lambda item: item["target_recall_at_threshold"]) if records else None
    best_purity = max(records, key=lambda item: item["candidate_purity_at_threshold"]) if records else None
    usable = [
        item
        for item in records
        if item["target_recall_at_threshold"] >= 0.20
        and item["candidate_purity_at_threshold"] >= 0.50
        and item["points"] >= 30
    ]
    if not records:
        separability = "target_classified_as_noise_or_no_accepted_cluster"
    elif best_recall["target_recall_at_threshold"] < 0.20:
        separability = "target_fragmented_or_classified_as_noise"
    elif best_recall["candidate_purity_at_threshold"] < 0.50:
        separability = "target_merged_with_support_or_background"
    elif len([item for item in records if item["target_recall_at_threshold"] >= 0.10]) > 1:
        separability = "target_fragmented_across_clusters"
    else:
        separability = "target_has_a_geometrically_useful_cluster"
    return {
        "config": {
            "cluster_eps_m": config.cluster_eps_m,
            "cluster_min_points": config.cluster_min_points,
            "min_candidate_points": config.min_candidate_points,
            "max_candidate_extent_m": config.max_candidate_extent_m,
            "support_plane": config.support_plane_distance_m,
            "outlier_removal": config.remove_statistical_outliers,
        },
        "generation": generation,
        "candidate_records": records,
        "best_by_target_recall": best_recall,
        "best_by_candidate_purity": best_purity,
        "usable_candidate_rule_diagnostic_only": "target_recall_at_5mm >= 0.20 AND candidate_purity_at_5mm >= 0.50 AND points >= 30",
        "usable_candidate_count_diagnostic_only": len(usable),
        "target_separability_classification": separability,
        "gt_only_note": "GT mask labels are used after candidate generation and never select an inference candidate.",
    }


def axis_angle(axis: Iterable[float], angle_deg: float) -> np.ndarray:
    axis_array = np.asarray(list(axis), dtype=float)
    axis_array /= np.linalg.norm(axis_array)
    angle = math.radians(angle_deg)
    x, y, z = axis_array
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        [
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
        ]
    )


def run_icp_diagnostic(model: ModelCatalogEntry, target: np.ndarray, gt: np.ndarray, config: DetectionConfig, perturb: bool) -> dict[str, Any]:
    started = time.perf_counter()
    delta = np.eye(4)
    if perturb:
        delta[:3, :3] = axis_angle((1.0, -2.0, 1.0), 4.0)
        delta[:3, 3] = [0.004, -0.003, 0.005]
        initial = gt @ delta
    else:
        initial = gt.copy()
    result: dict[str, Any] = {
        "model_representation": model.representation,
        "initialization": "GT-INITIALIZED DIAGNOSTIC ONLY — NOT INFERENCE",
        "perturbation": "deterministic_small" if perturb else "exact_gt",
        "initial_pose": pose_metrics(initial, gt),
    }
    try:
        use_normals = model.normals is not None
        source = _limit_points(_to_point_cloud(model.points, model.normals if use_normals else None), config.max_registration_points)
        destination = _limit_points(_to_point_cloud(target), config.max_registration_points)
        if not source.has_normals():
            _estimate_normals(source, config, config.normal_radius_factor)
        _estimate_normals(destination, config, config.normal_radius_factor, orientation=config.observed_normal_orientation)
        icp = o3d.pipelines.registration.registration_icp(
            source,
            destination,
            config.icp_max_correspondence_m,
            initial,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=config.icp_iterations),
        )
        final = np.asarray(icp.transformation)
        result.update(
            {
                "status": "ok",
                "final_pose": pose_metrics(final, gt),
                "final_transform": final.tolist(),
                "icp_fitness": float(icp.fitness),
                "icp_rmse_m": float(icp.inlier_rmse) if len(icp.correspondence_set) else None,
                "icp_inliers": int(len(icp.correspondence_set)),
                "improved_translation": pose_metrics(final, gt)["translation_error_m"] < result["initial_pose"]["translation_error_m"],
                "improved_rotation": pose_metrics(final, gt)["rotation_error_deg"] < result["initial_pose"]["rotation_error_deg"],
            }
        )
    except Exception as exc:
        result.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    result["runtime_s"] = time.perf_counter() - started
    return result


def symmetry_info(root: Path, model_id: int) -> dict[str, Any]:
    data = read_json(root / "models_cad" / "models_info.json")[str(model_id)]
    discrete = data.get("symmetries_discrete", [])
    continuous = data.get("symmetries_continuous", [])
    return {
        "status": "continuous" if continuous else "discrete" if discrete else "none",
        "continuous": bool(continuous),
        "discrete_count": len(discrete),
        "raw": {"symmetries_discrete": discrete, "symmetries_continuous": continuous},
    }


def posthoc_correctness(v1_dir: Path, root: Path, manifest: list[dict[str, Any]]) -> dict[str, Any]:
    rows = read_json(v1_dir / "per_case_results.json")
    manifest_by_case = {item["case_id"]: item for item in manifest}
    output = []
    for row in rows:
        estimate = np.asarray(row["T_camera_object_estimated"], dtype=float)
        gt = np.asarray(row["T_camera_object_ground_truth"], dtype=float)
        errors = pose_metrics(estimate, gt)
        sym = symmetry_info(root, int(row["model_id"]))
        applicable = sym["status"] == "none"
        accurate = diagnostic_accuracy(
            float(errors["translation_error_m"]),
            float(errors["rotation_error_deg"]),
            rotation_applicable=applicable,
        )
        output.append(
            {
                "case_id": row["case_id"],
                "variant": row["variant"],
                "accepted_internal": bool(row["accepted"]),
                "diagnostic_accurate": accurate,
                "evaluation_class": classify_acceptance(bool(row["accepted"]), accurate),
                **errors,
                **error_bands(errors["translation_error_m"], errors["rotation_error_deg"], rotation_applicable=applicable),
                "symmetry": sym["status"],
                "naive_rotation_applicable": applicable,
                "official_bop_metrics": official_pose_metrics(root, row, manifest_by_case[row["case_id"]]),
            }
        )
    counts = Counter(item["evaluation_class"] for item in output)
    return {
        "accuracy_definition": {
            "translation_tolerance_m": 0.020,
            "non_symmetric_rotation_tolerance_deg": 5.0,
            "symmetric_objects": "translation-only diagnostic classification; official BOP symmetry-aware values retained separately",
        },
        "rows": output,
        "counts": dict(counts),
        "note": "This is post-inference diagnostic labeling, not object-detection precision/recall.",
    }


def official_symmetries(root: Path, model_id: int) -> tuple[list[dict[str, np.ndarray]], str]:
    info = read_json(root / "models_eval" / "models_info.json")[str(model_id)]
    syms = [{"R": np.eye(3), "t": np.zeros((3, 1))}]
    if info.get("symmetries_continuous"):
        return syms, "continuous_symmetry_not_discretized"
    for flat in info.get("symmetries_discrete", []):
        matrix = np.asarray(flat, dtype=float).reshape(4, 4)
        syms.append({"R": matrix[:3, :3], "t": matrix[:3, 3].reshape(3, 1)})
    return syms, "identity_plus_discrete" if len(syms) > 1 else "identity_only"


def official_pose_metrics(root: Path, row: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    try:
        from bop_toolkit_lib import pose_error
    except ImportError:
        return {"available": False, "reason": "bop_toolkit_lib not installed in this environment"}
    try:
        model_id = int(row["model_id"])
        model_path = root / "models_eval" / f"obj_{model_id:06d}.ply"
        pts = np.asarray(load_point_cloud(model_path), dtype=float)  # BOP evaluation units: mm.
        estimate = np.asarray(row["T_camera_object_estimated"], dtype=float)
        gt = np.asarray(row["T_camera_object_ground_truth"], dtype=float)
        R_est, t_est = estimate[:3, :3], (estimate[:3, 3] * 1000.0).reshape(3, 1)
        R_gt, t_gt = gt[:3, :3], (gt[:3, 3] * 1000.0).reshape(3, 1)
        syms, symmetry_status = official_symmetries(root, model_id)
        camera_path = root / "test_primesense" / f"{int(case['scene_id']):06d}" / "scene_camera.json"
        camera = read_json(camera_path)[str(int(case["image_id"]))]
        K = np.asarray(camera["cam_K"], dtype=float).reshape(3, 3)
        result = {
            "available": True,
            "model_source": "models_eval",
            "translation_units": "mm",
            "mssd_mm": float(pose_error.mssd(R_est, t_est, R_gt, t_gt, pts, syms)),
            "re_deg": float(pose_error.re(R_est, R_gt)),
            "te_mm": float(pose_error.te(t_est, t_gt)),
            "symmetry_status": symmetry_status,
        }
        if K is not None:
            mspd = float(pose_error.mspd(R_est, t_est, R_gt, t_gt, K, pts, syms))
            result["mspd_px"] = mspd if math.isfinite(mspd) else None
        return result
    except Exception as exc:
        return {"available": False, "reason": f"official metric error: {type(exc).__name__}: {exc}"}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def write_bop_csvs(output: Path, v1_dir: Path, cases: list[dict[str, Any]]) -> dict[str, Any]:
    rows = read_json(v1_dir / "per_case_results.json")
    expected_cases = {item["case_id"]: item for item in cases}
    audit: dict[str, Any] = {}
    for variant in sorted({row["variant"] for row in rows}):
        variant_rows = [row for row in rows if row["variant"] == variant]
        csv_path = output / "bop_predictions" / f"{variant}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["scene_id", "im_id", "obj_id", "score", "R", "t", "time"])
            writer.writeheader()
            for row in variant_rows:
                transform = np.asarray(row["T_camera_object_estimated"], dtype=float)
                writer.writerow(
                    {
                        "scene_id": int(row["scene_id"]),
                        "im_id": int(row["image_id"]),
                        "obj_id": int(row["model_id"]),
                        "score": 1.0 if row["accepted"] else 0.0,
                        "R": " ".join(f"{value:.9g}" for value in transform[:3, :3].reshape(-1)),
                        "t": " ".join(f"{value * 1000.0:.9g}" for value in transform[:3, 3]),
                        "time": float(row["runtime_s"]),
                    }
                )
        keys = [(int(row["scene_id"]), int(row["image_id"]), int(row["model_id"])) for row in variant_rows]
        audit[variant] = {
            "path": str(csv_path),
            "fields": ["scene_id", "im_id", "obj_id", "score", "R", "t", "time"],
            "row_count": len(variant_rows),
            "duplicate_keys": len(keys) - len(set(keys)),
            "all_frozen_manifest_cases_present": {case_id: case_id in {row["case_id"] for row in variant_rows} for case_id in expected_cases},
            "translation_units": "mm",
            "runtime_units": "seconds",
            "model_conditioned_note": "One hypothesis is emitted for each frozen model-conditioned case; this is not a complete BOP detector result file.",
        }
        try:
            from bop_toolkit_lib import inout

            parsed = inout.load_bop_results(str(csv_path), version="bop19")
            audit[variant]["official_parser"] = {"available": True, "parsed_count": len(parsed)}
        except ImportError:
            audit[variant]["official_parser"] = {"available": False, "reason": "bop_toolkit_lib not installed"}
        except Exception as exc:
            audit[variant]["official_parser"] = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    return audit


def rank_hypotheses(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            by_case[row["case_id"]].append(row)
    metrics = {
        "bidirectional_overlap": False,
        "model_to_candidate_overlap": False,
        "candidate_to_model_overlap": False,
        "icp_rmse_m": True,
        "icp_fitness": False,
        "global_inliers": False,
        "model_to_candidate_trimmed_rmse_m": True,
        "candidate_to_model_trimmed_rmse_m": True,
    }
    output = {}
    for case_id, case_rows in by_case.items():
        output[case_id] = {}
        for metric, ascending in metrics.items():
            usable = [row for row in case_rows if row.get("verification", {}).get(metric) is not None or row.get(metric) is not None]
            usable.sort(key=lambda row: row.get("verification", {}).get(metric, row.get(metric)), reverse=not ascending)
            correct = [row for row in usable if row.get("final_diagnostic_accurate_non_symmetric")]
            output[case_id][metric] = {
                "ranking": [f"{row['model_representation']}:{row['method']}" for row in usable],
                "top_is_diagnostic_accurate": bool(usable and usable[0].get("final_diagnostic_accurate_non_symmetric")),
                "best_accurate_rank": next((index + 1 for index, row in enumerate(usable) if row.get("final_diagnostic_accurate_non_symmetric")), None),
                "available_hypotheses": len(usable),
                "accurate_hypotheses": len(correct),
            }
    return {
        "metrics": "Post-hoc descriptive ranking only; no learned score or production threshold is fit.",
        "by_case": output,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--v1-dir", type=Path, default=DEFAULT_V1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest = read_json(args.manifest)
    cases = manifest["cases"]
    write_json(output / "manifest_copy.json", manifest)
    write_json(output / "method_review.json", method_review())

    # Post-hoc audit of the immutable v1 predictions.
    posthoc = posthoc_correctness(args.v1_dir, root, cases)
    write_json(output / "posthoc_acceptance_correctness.json", posthoc)
    write_csv(output / "posthoc_acceptance_correctness.csv", posthoc["rows"])
    write_json(output / "acceptance_confusion.json", posthoc["counts"])

    bop_audit = write_bop_csvs(output, args.v1_dir, cases)
    # Add explicit target-file compatibility information without using targets
    # to change any inference input.
    target_path = root / "tless" / "test_targets_bop19.json"
    targets = read_json(target_path) if target_path.is_file() else []
    target_keys = {(int(item["scene_id"]), int(item["im_id"]), int(item["obj_id"])) for item in targets}
    frozen_keys = {(int(c["scene_id"]), int(c["image_id"]), int(c["model_id"])) for c in cases}
    bop_audit["target_definition"] = {
        "path": str(target_path),
        "available": target_path.is_file(),
        "frozen_case_keys_in_targets": sorted(frozen_keys & target_keys),
        "frozen_case_keys_missing_from_targets": sorted(frozen_keys - target_keys),
        "inst_count_for_frozen_cases": [item for item in targets if (int(item["scene_id"]), int(item["im_id"]), int(item["obj_id"])) in frozen_keys],
        "not_a_valid_aggregate_note": "The frozen file is model-conditioned and contains only selected cases; no BOP aggregate score is reported.",
    }
    write_json(output / "bop_compatibility.json", bop_audit)

    candidate_rows: list[dict[str, Any]] = []
    registration_rows: list[dict[str, Any]] = []
    icp_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    config = config_for_representation("cad")
    print(f"Running frozen-set diagnosis for {len(cases)} cases", flush=True)
    for case_number, case in enumerate(cases, start=1):
        case_id = case["case_id"]
        print(f"[{case_number}/{len(cases)}] {case_id}", flush=True)
        frame = load_case_frame(root, case)
        case_base = {
            "case_id": case_id,
            "scene_id": case["scene_id"],
            "image_id": case["image_id"],
            "model_id": case["model_id"],
            "visibility_fraction": case["visibility_fraction"],
            "oracle_target_points": len(frame["oracle_target"]),
            "oracle_target_stats": point_stats(frame["oracle_target"]),
            "depth_valid_scene_points": len(frame["full_scene"]),
            "gt_pose": frame["gt"].tolist(),
            "gt_info": frame["gt_info"],
        }
        candidate = run_candidate_analysis(frame, config)
        candidate_rows.append({**case_base, **candidate})
        for representation in ("cad", "reconst"):
            model = model_geometry(root, int(case["model_id"]), representation)
            geometry_rows.append(
                {
                    **case_base,
                    "representation": representation,
                    "model_path": str(root / f"models_{representation}" / f"obj_{int(case['model_id']):06d}.ply"),
                    "model_stats": point_stats(model.points, model.normals),
                    "model_normals_loaded": model.normals is not None,
                    "gt_aligned_model_vs_oracle": nearest_metrics(model.points, frame["oracle_target"], frame["gt"]),
                    "physical_scale_audit": {
                        "voxel_size_m": config.voxel_size_m,
                        "normal_radius_m": config.voxel_size_m * config.normal_radius_factor,
                        "fpfh_radius_m": config.voxel_size_m * config.feature_radius_factor,
                        "ransac_threshold_m": config.ransac_inlier_threshold_m,
                        "icp_threshold_m": config.icp_max_correspondence_m,
                        "voxel_to_smallest_dimension": config.voxel_size_m / max(float(np.min(np.ptp(model.points, axis=0))), 1e-12),
                        "fpfh_radius_to_dimensions": (config.voxel_size_m * config.feature_radius_factor / np.maximum(np.ptp(model.points, axis=0), 1e-12)).tolist(),
                        "ransac_threshold_to_dimensions": (config.ransac_inlier_threshold_m / np.maximum(np.ptp(model.points, axis=0), 1e-12)).tolist(),
                    },
                }
            )
            for method in ("ransac_point_to_plane_icp", "fgr_point_to_plane_icp", "ransac_generalized_icp"):
                row = run_global_variant(model, frame["oracle_target"], frame["gt"], config_for_representation(representation), method)
                row.update({"case_id": case_id, "scene_id": case["scene_id"], "image_id": case["image_id"], "model_id": case["model_id"]})
                registration_rows.append(row)
                if row.get("status") == "ok":
                    feature_rows.append({
                        "case_id": case_id,
                        "model_id": case["model_id"],
                        "representation": representation,
                        "method": method,
                        **row.get("feature_diagnostics", {}),
                    })
            for perturb in (False, True):
                icp = run_icp_diagnostic(model, frame["oracle_target"], frame["gt"], config_for_representation(representation), perturb)
                icp.update({"case_id": case_id, "scene_id": case["scene_id"], "image_id": case["image_id"], "model_id": case["model_id"]})
                icp_rows.append(icp)

    write_json(output / "candidate_oracle_analysis.json", candidate_rows)
    write_csv(output / "candidate_oracle_analysis.csv", candidate_rows)
    write_json(output / "model_geometry_and_scale_audit.json", geometry_rows)
    write_json(output / "registration_oracle_bakeoff.json", registration_rows)
    write_csv(output / "registration_oracle_bakeoff.csv", registration_rows)
    write_json(output / "icp_oracle_study.json", icp_rows)
    write_csv(output / "icp_oracle_study.csv", icp_rows)
    write_json(output / "feature_correspondence_diagnostics.json", feature_rows)
    write_csv(output / "feature_correspondence_diagnostics.csv", feature_rows)
    write_json(output / "hypothesis_ranking.json", rank_hypotheses(registration_rows))

    summary = {
        "dataset_root": str(root),
        "split": "test_primesense",
        "manifest": str(args.manifest),
        "v1_preserved": True,
        "case_count": len(cases),
        "registration_methods": sorted({row["method"] for row in registration_rows}),
        "registration_ok_count": sum(row.get("status") == "ok" for row in registration_rows),
        "candidate_usable_oracle_count": sum(item["usable_candidate_count_diagnostic_only"] > 0 for item in candidate_rows),
        "candidate_any_count": sum(item["generation"].get("candidate_count", 0) > 0 for item in candidate_rows),
        "registration_accuracy_by_method": {
            method: {
                "rows": [row for row in registration_rows if row["method"] == method],
                "accurate_count_non_symmetric": sum(
                    row.get("final_diagnostic_accurate_non_symmetric", False)
                    for row in registration_rows
                    if row["method"] == method
                ),
            }
            for method in sorted({row["method"] for row in registration_rows})
        },
        "production_decision": "No production defaults changed. The bakeoff is diagnostic and uses only frozen oracle clouds.",
    }
    # Keep summary compact; full rows live in the method-specific files.
    for value in summary["registration_accuracy_by_method"].values():
        value.pop("rows", None)
    write_json(output / "summary.json", summary)
    (output / "README.md").write_text(
        """# T-LESS registration diagnosis v2

This directory is a new diagnostic record. `experiments/tless_validation_v1/`
and the frozen manifest are preserved unchanged. Ground truth is used only
after candidate generation for labels, explicitly marked oracle clouds, or
post-hoc errors. It is never an inference initialization, crop, mask,
candidate choice, or threshold.

The bakeoff compares Open3D feature-RANSAC plus point-to-plane ICP, FGR plus
point-to-plane ICP, and feature-RANSAC plus Generalized ICP with unchanged
frozen settings. It is an oracle-candidate study, not a detector benchmark.
Inference uses CAD and reconstructed models separately; `models_eval` is only
for the official BOP pose-error audit.

Units are metres internally and millimetres in BOP CSV/official pose errors.
The official parser and MSSD audit details are recorded in
`bop_compatibility.json`. No aggregate BOP score is reported for these
model-conditioned selected cases.
""",
        encoding="utf-8",
    )
    print(f"Wrote diagnosis to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
