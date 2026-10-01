#!/usr/bin/env python3
"""Run the isolated PCL/Drost PPF architecture spike.

This is an experiment adapter, not a production detector.  It consumes the
same ten-case manifest as the frozen v2 diagnosis, reconstructs the same
GT-visible oracle clouds for diagnosis, and never supplies GT poses to PPF.
Ground truth is used only after PPF returns hypotheses to calculate official
BOP errors and to label diagnostic plots/tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import open3d as o3d
except ModuleNotFoundError as exc:  # pragma: no cover - experiment dependency
    raise SystemExit("Open3D is required by the experiment adapter") from exc

from zutik_geometry.evaluation import diagnostic_accuracy, rotation_error_degrees, translation_error_m
from zutik_geometry.io import load_model_catalog_entry
from zutik_geometry.ppf import PPFBridgeConfig, PPFRunResult, run_ppf_bridge
from zutik_geometry.registration import (
    _estimate_normals,
    _normalize_cloud_normals,
    _to_point_cloud,
    _voxel_cloud,
    compute_bidirectional_verification,
    transform_points,
)
from zutik_geometry.types import DetectionConfig

from scripts.run_tless_registration_diagnosis_v2 import (
    DEFAULT_MANIFEST,
    DEFAULT_ROOT,
    config_for_representation,
    load_case_frame,
    model_geometry,
    official_pose_metrics,
    pose_metrics,
    read_json,
    to_json_value,
    write_json,
)


DEFAULT_OUTPUT = Path("experiments/tless_ppf_spike_v1")
DEFAULT_EXECUTABLE = Path("/usr/local/bin/zutik_ppf_bridge")
PPF_VOXEL_SIZE_M = 0.008
PPF_MAX_MODEL_POINTS = 500
PPF_MAX_SCENE_POINTS = 800


def ppf_config(executable: Path) -> PPFBridgeConfig:
    """One global configuration fixed before the frozen oracle experiment."""

    return PPFBridgeConfig(
        executable=executable,
        angle_bin_deg=10.0,
        distance_bin_m=0.005,
        scene_reference_sampling_rate=5,
        position_cluster_m=0.03,
        rotation_cluster_deg=25.0,
        top_k=20,
        timeout_s=180.0,
    )


def ppf_config_record(config: PPFBridgeConfig) -> dict[str, Any]:
    record = dict(config.__dict__)
    record["executable"] = str(record["executable"])
    return record


def ppf_preprocess(model: Any, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Prepare bounded oriented clouds without using GT or changing production settings."""

    config = DetectionConfig(
        voxel_size_m=PPF_VOXEL_SIZE_M,
        max_registration_points=PPF_MAX_MODEL_POINTS,
        model_representation=model.representation,
        model_normal_source="auto",
        observed_normal_orientation="towards_camera",
    )
    source = _voxel_cloud(model.points, config, normals=model.normals)
    destination_config = DetectionConfig(
        voxel_size_m=PPF_VOXEL_SIZE_M,
        max_registration_points=PPF_MAX_SCENE_POINTS,
        model_representation=model.representation,
        model_normal_source="estimate",
        observed_normal_orientation="towards_camera",
    )
    destination = _voxel_cloud(target, destination_config)
    if len(source.points) < 3 or len(destination.points) < 3:
        raise ValueError("PPF preprocessing left fewer than three points")
    if source.has_normals():
        _normalize_cloud_normals(source)
        model_normal_source = "mesh"
    else:
        _estimate_normals(source, config, 2.0, orientation="none")
        model_normal_source = "estimated"
    _estimate_normals(destination, destination_config, 2.0, orientation="towards_camera")
    model_points = np.asarray(source.points, dtype=float)
    model_normals = np.asarray(source.normals, dtype=float)
    scene_points = np.asarray(destination.points, dtype=float)
    scene_normals = np.asarray(destination.normals, dtype=float)
    return model_points, model_normals, scene_points, scene_normals, {
        "voxel_size_m": PPF_VOXEL_SIZE_M,
        "max_model_points": PPF_MAX_MODEL_POINTS,
        "max_scene_points": PPF_MAX_SCENE_POINTS,
        "model_normal_source": model_normal_source,
        "observed_normal_orientation": "towards_camera",
        "model_points_after_preprocessing": len(model_points),
        "scene_points_after_preprocessing": len(scene_points),
    }


def hypothesis_metrics(
    root: Path,
    case: dict[str, Any],
    model: Any,
    target: np.ndarray,
    transform: np.ndarray,
) -> dict[str, Any]:
    row = {
        "case_id": case["case_id"],
        "scene_id": int(case["scene_id"]),
        "image_id": int(case["image_id"]),
        "model_id": int(case["model_id"]),
        "T_camera_object_estimated": transform.tolist(),
        "T_camera_object_ground_truth": case["gt"].tolist(),
    }
    verification = compute_bidirectional_verification(
        model.points,
        target,
        transform,
        distance_threshold_m=0.02,
        max_points=2500,
    )
    naive = pose_metrics(transform, case["gt"])
    official = official_pose_metrics(root, row, case)
    return {
        "naive_pose": naive,
        "official_bop": official,
        "diagnostic_accuracy_20mm_5deg": diagnostic_accuracy(
            float(naive["translation_error_m"]), float(naive["rotation_error_deg"])
        ),
        "verification": verification.to_dict(),
    }


def _refine_one(
    source_points: np.ndarray,
    source_normals: np.ndarray,
    target_points: np.ndarray,
    target_normals: np.ndarray,
    initial: np.ndarray,
    config: DetectionConfig,
    method: str,
) -> dict[str, Any]:
    registration = o3d.pipelines.registration
    source = _to_point_cloud(source_points, source_normals)
    target = _to_point_cloud(target_points, target_normals)
    started = time.perf_counter()
    if method == "gicp":
        result = registration.registration_generalized_icp(
            source,
            target,
            config.icp_max_correspondence_m,
            initial,
            registration.TransformationEstimationForGeneralizedICP(),
            registration.ICPConvergenceCriteria(max_iteration=config.icp_iterations),
        )
    elif method == "point_to_plane_icp":
        result = registration.registration_icp(
            source,
            target,
            config.icp_max_correspondence_m,
            initial,
            registration.TransformationEstimationPointToPlane(),
            registration.ICPConvergenceCriteria(max_iteration=config.icp_iterations),
        )
    else:
        raise ValueError(method)
    return {
        "status": "ok",
        "method": method,
        "transform": np.asarray(result.transformation).tolist(),
        "fitness": float(result.fitness),
        "rmse_m": float(result.inlier_rmse) if len(result.correspondence_set) else None,
        "inliers": int(len(result.correspondence_set)),
        "runtime_s": time.perf_counter() - started,
    }


def synthetic_model() -> tuple[np.ndarray, np.ndarray]:
    """Asymmetric box surface with deterministic oriented points."""

    axes = np.linspace(-1.0, 1.0, 9)
    points: list[list[float]] = []
    normals: list[list[float]] = []
    # Dimensions are 120 x 80 x 50 mm; each face has a different extent.
    for u in axes:
        for v in axes:
            points.extend(
                [
                    [0.06, 0.04 * u, 0.025 * v],
                    [-0.06, 0.04 * u, 0.025 * v],
                    [0.06 * u, 0.04, 0.025 * v],
                    [0.06 * u, -0.04, 0.025 * v],
                    [0.06 * u, 0.04 * v, 0.025],
                    [0.06 * u, 0.04 * v, -0.025],
                ]
            )
            normals.extend([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    # Break the rectangular-box 180-degree symmetries with an offset raised
    # rectangular feature. Without this, a rotation error near 180 degrees is
    # a fixture ambiguity rather than evidence about the bridge convention.
    bump_u = np.linspace(-0.010, 0.010, 5)
    bump_v = np.linspace(-0.008, 0.008, 5)
    for u in bump_u:
        for v in bump_v:
            points.append([0.025 + u, 0.012 + v, 0.035])
            normals.append([0, 0, 1])
    for u in bump_u:
        for z in np.linspace(0.025, 0.035, 3):
            points.extend([[0.015 + u, 0.004, z], [0.035 + u, 0.004, z]])
            normals.extend([[0, -1, 0], [0, 1, 0]])
    for v in bump_v:
        for z in np.linspace(0.025, 0.035, 3):
            points.extend([[0.015, 0.012 + v, z], [0.035, 0.012 + v, z]])
            normals.extend([[-1, 0, 0], [1, 0, 0]])
    return np.asarray(points, dtype=float), np.asarray(normals, dtype=float)


def axis_angle(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    theta = math.radians(angle_deg)
    x, y, z = axis
    c, s = math.cos(theta), math.sin(theta)
    return np.array(
        [
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
        ]
    )


def run_synthetic(executable: Path, output: Path) -> dict[str, Any]:
    rng = np.random.default_rng(20261001)
    model, model_normals = synthetic_model()
    gt = np.eye(4)
    gt[:3, :3] = axis_angle(np.array([1.0, 2.0, -1.0]), 37.0)
    gt[:3, 3] = [0.23, -0.11, 0.72]
    transformed = transform_points(gt, model)
    transformed_normals = model_normals @ gt[:3, :3].T
    keep = np.arange(len(model)) % 7 != 0  # deterministic partial visibility
    observed = transformed[keep] + rng.normal(0.0, 0.0007, size=(int(np.sum(keep)), 3))
    observed_normals = transformed_normals[keep]
    clutter = rng.uniform([-0.20, -0.20, 0.45], [0.45, 0.20, 1.0], size=(120, 3))
    clutter_normals = rng.normal(size=(len(clutter), 3))
    clutter_normals /= np.linalg.norm(clutter_normals, axis=1, keepdims=True)
    scene = np.vstack([observed, clutter])
    scene_normals = np.vstack([observed_normals, clutter_normals])
    config = ppf_config(executable)
    started = time.perf_counter()
    result = run_ppf_bridge(model, model_normals, scene, scene_normals, config)
    hypotheses = []
    for hypothesis in result.hypotheses:
        te = translation_error_m(hypothesis.T_camera_object, gt)
        re = rotation_error_degrees(hypothesis.T_camera_object, gt)
        hypotheses.append({"rank": hypothesis.rank, "votes": hypothesis.votes, "translation_error_m": te, "rotation_error_deg": re})
    best = min(hypotheses, key=lambda item: (item["translation_error_m"], item["rotation_error_deg"])) if hypotheses else None
    record = {
        "status": "ok",
        "label": "deterministic synthetic PPF bridge validation",
        "gt_pose_not_provided_to_bridge": True,
        "model_points": len(model),
        "scene_points": len(scene),
        "partial_visibility_fraction": float(np.mean(keep)),
        "noise_sigma_m": 0.0007,
        "outlier_points": len(clutter),
        "bridge_runtime_s": result.runtime_s,
        "wall_runtime_s": time.perf_counter() - started,
        "hypotheses": hypotheses,
        "best_of_k_posthoc": best,
        "success_within_20mm_5deg": bool(best and best["translation_error_m"] <= 0.02 and best["rotation_error_deg"] <= 5.0),
        "configuration": ppf_config_record(config),
    }
    write_json(output / "synthetic_result.json", record)
    return record


def run_real(root: Path, manifest_path: Path, executable: Path, output: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    cases = manifest["cases"]
    rows: list[dict[str, Any]] = []
    refinements: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    bridge_config = ppf_config(executable)
    for case_number, case in enumerate(cases, start=1):
        print(f"[{case_number}/{len(cases)}] {case['case_id']}", flush=True)
        frame = load_case_frame(root, case)
        case_for_metrics = {**case, "gt": frame["gt"]}
        for representation in ("cad", "reconst"):
            model = model_geometry(root, int(case["model_id"]), representation)
            model_points, model_normals, scene_points, scene_normals, prep = ppf_preprocess(model, frame["oracle_target"])
            started = time.perf_counter()
            status = "ok"
            error = None
            result: PPFRunResult | None = None
            try:
                result = run_ppf_bridge(model_points, model_normals, scene_points, scene_normals, bridge_config)
            except Exception as exc:
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
            case_record: dict[str, Any] = {
                "case_id": case["case_id"],
                "scene_id": int(case["scene_id"]),
                "image_id": int(case["image_id"]),
                "model_id": int(case["model_id"]),
                "representation": representation,
                "oracle_points_before_preprocessing": len(frame["oracle_target"]),
                "preprocessing": prep,
                "status": status,
                "error": error,
                "bridge_runtime_s": result.runtime_s if result else None,
                "wall_runtime_s": time.perf_counter() - started,
                "bridge_diagnostics": result.diagnostics if result else {},
                "hypotheses": [],
            }
            if result is not None:
                for hypothesis in result.hypotheses:
                    metrics = hypothesis_metrics(root, case_for_metrics, model, frame["oracle_target"], hypothesis.T_camera_object)
                    case_record["hypotheses"].append({
                        "rank": hypothesis.rank,
                        "votes": hypothesis.votes,
                        "T_camera_object": hypothesis.T_camera_object.tolist(),
                        **metrics,
                    })
                if result.hypotheses:
                    top = result.hypotheses[0]
                    for method in ("gicp", "point_to_plane_icp"):
                        refinement = {
                            "case_id": case["case_id"],
                            "representation": representation,
                            "initial_rank": 1,
                            "initial_metrics": hypothesis_metrics(root, case_for_metrics, model, frame["oracle_target"], top.T_camera_object),
                            "initialization": "raw PPF top-1; no GT initialization",
                            "method": method,
                        }
                        try:
                            refined = _refine_one(
                                model_points,
                                model_normals,
                                scene_points,
                                scene_normals,
                                top.T_camera_object,
                                config_for_representation(representation),
                                method,
                            )
                            refined_transform = np.asarray(refined["transform"], dtype=float)
                            refinement.update(refined)
                            refinement["final_metrics"] = hypothesis_metrics(root, case_for_metrics, model, frame["oracle_target"], refined_transform)
                            initial_pose = refinement["initial_metrics"]["naive_pose"]
                            final_pose = refinement["final_metrics"]["naive_pose"]
                            refinement["improved_translation"] = final_pose["translation_error_m"] < initial_pose["translation_error_m"]
                            refinement["improved_rotation"] = final_pose["rotation_error_deg"] < initial_pose["rotation_error_deg"]
                        except Exception as exc:
                            refinement.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
                        refinements.append(refinement)
            rows.append(case_record)

    write_json(output / "ppf_hypotheses.json", rows)
    with (output / "ppf_hypotheses.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(to_json_value(row), sort_keys=True) + "\n")
    write_json(output / "refinement_comparison.json", refinements)
    write_csv(
        output / "ppf_top1_and_best_of_k.csv",
        summarize_rows(rows),
    )
    write_json(output / "ranking_analysis.json", ranking_analysis(rows))
    summary = summarize_experiment(rows, refinements, root, bridge_config, time.perf_counter() - started_all)
    write_json(output / "summary.json", summary)
    return summary


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        hypotheses = row.get("hypotheses", [])
        top = hypotheses[0] if hypotheses else {}
        best = min(hypotheses, key=lambda item: item["official_bop"].get("mssd_mm", float("inf"))) if hypotheses else {}
        result.append({
            "case_id": row["case_id"],
            "representation": row["representation"],
            "status": row["status"],
            "hypothesis_count": len(hypotheses),
            "top1_votes": top.get("votes"),
            "top1_translation_error_mm": top.get("naive_pose", {}).get("translation_error_mm"),
            "top1_rotation_error_deg": top.get("naive_pose", {}).get("rotation_error_deg"),
            "top1_mssd_mm": top.get("official_bop", {}).get("mssd_mm"),
            "top1_mspd_px": top.get("official_bop", {}).get("mspd_px"),
            "top1_bidirectional_overlap": top.get("verification", {}).get("bidirectional_overlap"),
            "best_of_k_mssd_mm_posthoc_oracle": best.get("official_bop", {}).get("mssd_mm"),
            "best_of_k_translation_error_mm_posthoc_oracle": best.get("naive_pose", {}).get("translation_error_mm"),
            "best_of_k_rotation_error_deg_posthoc_oracle": best.get("naive_pose", {}).get("rotation_error_deg"),
        })
    return result


def ranking_analysis(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare available non-GT geometric rankings without selecting inference output."""

    result: list[dict[str, Any]] = []
    for row in rows:
        hypotheses = row.get("hypotheses", [])
        if not hypotheses:
            result.append({"case_id": row["case_id"], "representation": row["representation"], "hypothesis_count": 0})
            continue
        by_overlap = sorted(
            range(len(hypotheses)),
            key=lambda index: hypotheses[index]["verification"]["bidirectional_overlap"],
            reverse=True,
        )
        by_trimmed = sorted(
            range(len(hypotheses)),
            key=lambda index: (
                hypotheses[index]["verification"]["model_to_candidate_trimmed_rmse_m"]
                + hypotheses[index]["verification"]["candidate_to_model_trimmed_rmse_m"]
            ),
        )
        by_mssd = sorted(
            range(len(hypotheses)),
            key=lambda index: hypotheses[index]["official_bop"].get("mssd_mm", float("inf")),
        )
        result.append(
            {
                "case_id": row["case_id"],
                "representation": row["representation"],
                "hypothesis_count": len(hypotheses),
                "ppf_ranked_pose_rank": 1,
                "bidirectional_best_rank": by_overlap[0] + 1,
                "trimmed_symmetric_best_rank": by_trimmed[0] + 1,
                "posthoc_mssd_best_rank": by_mssd[0] + 1,
                "ranking_is_informative": len(hypotheses) > 1,
                "note": "One PCL 1.13 hypothesis was exposed; ranking agreement is therefore inconclusive, not evidence of a useful ranker.",
            }
        )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_experiment(rows: list[dict[str, Any]], refinements: list[dict[str, Any]], root: Path, config: PPFBridgeConfig, runtime_s: float) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "dataset_root": str(root),
        "split": "test_primesense",
        "case_count": len({row["case_id"] for row in rows}),
        "representation_rows": len(rows),
        "runtime_s": runtime_s,
        "configuration": ppf_config_record(config),
        "preprocessing": {
            "voxel_size_m": PPF_VOXEL_SIZE_M,
            "max_model_points": PPF_MAX_MODEL_POINTS,
            "max_scene_points": PPF_MAX_SCENE_POINTS,
        },
        "refinement_rows": len(refinements),
        "results": {},
        "ranking_note": "best_of_k values are POST-HOC ORACLE RANKING — NOT INFERENCE; PCL internally returns its highest-vote pose, but PCL 1.13 exposes neither the vote count nor the remaining ranked candidates.",
    }
    for representation in ("cad", "reconst"):
        selected = [row for row in rows if row["representation"] == representation]
        top = [row["hypotheses"][0] for row in selected if row["hypotheses"]]
        best = [min(row["hypotheses"], key=lambda item: item["official_bop"].get("mssd_mm", float("inf"))) for row in selected if row["hypotheses"]]
        mssd_top = [item["official_bop"]["mssd_mm"] for item in top if "mssd_mm" in item["official_bop"]]
        mssd_best = [item["official_bop"]["mssd_mm"] for item in best if "mssd_mm" in item["official_bop"]]
        official_top = [item["official_bop"] for item in top if item["official_bop"].get("available")]
        summary["results"][representation] = {
            "rows": len(selected),
            "failed_runs": sum(row["status"] != "ok" for row in selected),
            "empty_hypothesis_runs": sum(not row["hypotheses"] for row in selected),
            "top1_within_20mm_5deg": sum(item["diagnostic_accuracy_20mm_5deg"] for item in top),
            "best_of_k_within_20mm_5deg_posthoc_oracle": sum(item["diagnostic_accuracy_20mm_5deg"] for item in best),
            "official_metrics_available_count": len(official_top),
            "top1_median_mssd_mm": float(np.median(mssd_top)) if mssd_top else None,
            "best_of_k_median_mssd_mm_posthoc_oracle": float(np.median(mssd_best)) if mssd_best else None,
            "top1_median_translation_error_mm": float(np.median([item["naive_pose"]["translation_error_mm"] for item in top])) if top else None,
            "top1_median_rotation_error_deg": float(np.median([item["naive_pose"]["rotation_error_deg"] for item in top])) if top else None,
            "top1_median_bidirectional_overlap": float(np.median([item["verification"]["bidirectional_overlap"] for item in top])) if top else None,
        }
    summary["top_k"] = {
        "requested": config.top_k,
        "available": False,
        "reason": "PCL 1.13.0 public PPFRegistration API exposes getFinalTransformation only; getBestPoseCandidates is absent.",
        "best_of_k_note": "best_of_k fields equal the only exposed top-1 hypothesis in this run and are not a genuine top-K result.",
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic", action="store_true")
    mode.add_argument("--all", action="store_true")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--executable", type=Path, default=DEFAULT_EXECUTABLE)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "manifest_copy.json", read_json(args.manifest))
    write_json(args.output_dir / "configuration.json", {
        "bridge": ppf_config_record(ppf_config(args.executable)),
        "preprocessing": {
            "voxel_size_m": PPF_VOXEL_SIZE_M,
            "max_model_points": PPF_MAX_MODEL_POINTS,
            "max_scene_points": PPF_MAX_SCENE_POINTS,
            "units": "metres",
        },
        "inference_contract": "model source -> camera/oracle target; output T_camera_object; GT pose never passed to PPF",
        "frozen_manifest": str(args.manifest),
    })
    synthetic = run_synthetic(args.executable, args.output_dir)
    if not synthetic["success_within_20mm_5deg"]:
        print("Synthetic PPF validation failed; T-LESS run was not started.", file=sys.stderr)
        return 2
    print("Synthetic PPF validation passed.", flush=True)
    if args.synthetic:
        return 0
    summary = run_real(args.dataset_root.resolve(), args.manifest, args.executable, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
