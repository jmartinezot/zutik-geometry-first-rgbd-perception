"""Run the frozen, model-conditioned T-LESS engineering validation set.

Ground truth is read only after each detector invocation, for error reporting and
visualization.  It is never passed to scene construction, candidate generation,
registration, or initialization.  This is deliberately not a BOP benchmark
runner: the model identity is supplied by the manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np

from zutik_geometry.bop import _load_frame, resolve_model_directory
from zutik_geometry.io import load_model_catalog_entry
from zutik_geometry.registration import GeometryFirstDetector, depth_to_point_cloud
from zutik_geometry.synthetic import rotation_error_degrees
from zutik_geometry.types import DetectionConfig


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _ground_truth_entry(root: Path, split: str, case: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    scene_root = root / split / f"{int(case['scene_id']):06d}"
    entries = _read_json(scene_root / "scene_gt.json").get(str(int(case["image_id"])))
    index = int(case["gt_instance_index"])
    if not isinstance(entries, list) or index >= len(entries):
        raise ValueError(f"GT instance index {index} is absent for {case['case_id']}")
    entry = entries[index]
    if int(entry["obj_id"]) != int(case["model_id"]):
        raise ValueError(
            f"manifest model {case['model_id']} does not match GT instance {index}: {entry['obj_id']}"
        )
    from zutik_geometry.bop import bop_pose_to_transform

    return bop_pose_to_transform(entry["cam_R_m2c"], entry["cam_t_m2c"]), entry


def _symmetry_status(root: Path, case: dict[str, Any]) -> dict[str, Any]:
    info_path = root / "models_cad" / "models_info.json"
    info = _read_json(info_path).get(str(int(case["model_id"])), {}) if info_path.is_file() else {}
    discrete = info.get("symmetries_discrete") or []
    continuous = info.get("symmetries_continuous") or []
    if continuous:
        status = "continuous"
    elif discrete:
        status = "discrete"
    else:
        status = "none"
    return {"status": status, "discrete_count": len(discrete), "continuous": bool(continuous)}


def _config(variant: str) -> DetectionConfig:
    common = dict(
        voxel_size_m=0.005,
        ransac_inlier_threshold_m=0.015,
        icp_max_correspondence_m=0.02,
        random_seed=7,
        remove_statistical_outliers=True,
    )
    if variant == "A_baseline_cad_full":
        return DetectionConfig(
            **common,
            candidate_mode="full",
            model_representation="cad",
            model_normal_source="estimate",
            observed_normal_orientation="none",
            verification_mode="none",
            remove_support_plane=False,
        )
    if variant == "B_geometry_cad":
        return DetectionConfig(
            **common,
            candidate_mode="geometry_candidates",
            model_representation="cad",
            model_normal_source="auto",
            observed_normal_orientation="towards_camera",
            verification_mode="bidirectional",
            remove_support_plane=True,
        )
    if variant == "C_geometry_reconst":
        return DetectionConfig(
            **common,
            candidate_mode="geometry_candidates",
            model_representation="reconst",
            model_normal_source="auto",
            observed_normal_orientation="towards_camera",
            verification_mode="bidirectional",
            remove_support_plane=True,
        )
    raise ValueError(f"unknown validation variant: {variant}")


def _failure_stage(reason: str | None) -> str | None:
    if not reason:
        return None
    if "overlap" in reason or reason.startswith("verification"):
        return "verification"
    if reason.startswith("global_registration") or reason.startswith("ransac"):
        return "global_registration"
    if reason.startswith("icp"):
        return "icp"
    if reason.startswith("candidate") or reason == "no_scene_candidates":
        return "candidate_generation"
    return "other"


def _metric(record: dict[str, Any], key: str) -> float | None:
    value = record.get(key)
    return None if value is None else float(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "variant", "case_id", "scene_id", "image_id", "model_id", "selection_status",
        "visibility_fraction", "symmetry", "accepted", "failure_stage", "failure_reason",
        "translation_error_m", "translation_error_mm", "rotation_error_deg",
        "ransac_inliers", "ransac_rmse_m", "icp_inliers", "icp_fitness", "icp_rmse_m",
        "feature_correspondences", "candidate_count", "candidate_points", "candidate_source",
        "model_representation", "model_normal_source", "observed_normal_orientation",
        "model_to_candidate_overlap", "candidate_to_model_overlap", "bidirectional_overlap",
        "model_to_candidate_rmse_m", "candidate_to_model_rmse_m",
        "model_to_candidate_trimmed_rmse_m", "candidate_to_model_trimmed_rmse_m",
        "extent_ratio_min", "extent_ratio_max", "runtime_s",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in fields})


def _project_overlay(
    root: Path,
    split: str,
    case: dict[str, Any],
    variant: str,
    model_path: Path,
    estimate: np.ndarray | None,
    truth: np.ndarray,
    output: Path,
) -> None:
    """Make a compact RGB overlay; GT is post-hoc diagnostic only."""

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return
    scene_root = root / split / f"{int(case['scene_id']):06d}"
    rgb_path = scene_root / "rgb" / f"{int(case['image_id']):06d}.png"
    if not rgb_path.is_file():
        return
    image = Image.open(rgb_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    camera = _read_json(scene_root / "scene_camera.json")[str(int(case["image_id"]))]
    matrix = np.asarray(camera["cam_K"], dtype=float).reshape(3, 3)
    model = load_model_catalog_entry(model_path, model_id=str(case["model_id"]), units="mm")
    points = model.points
    if len(points) > 1800:
        points = points[np.linspace(0, len(points) - 1, 1800, dtype=np.int64)]

    def draw_pose(pose: np.ndarray, color: tuple[int, int, int]) -> None:
        camera_points = points @ pose[:3, :3].T + pose[:3, 3]
        valid = camera_points[:, 2] > 1e-6
        uv = camera_points[valid] @ matrix[:3, :3].T
        uv = uv[:, :2] / uv[:, 2:3]
        width, height = image.size
        for x, y in uv:
            if 0 <= x < width and 0 <= y < height:
                draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)

    draw_pose(truth, (30, 220, 70))
    if estimate is not None:
        draw_pose(estimate, (240, 45, 35))
    draw.rectangle((5, 5, 360, 27), fill=(0, 0, 0))
    draw.text((10, 9), f"{variant}: red estimate / green GT — DIAGNOSTIC", fill=(255, 255, 255))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variant in sorted({record["variant"] for record in records}):
        subset = [record for record in records if record["variant"] == variant]

        def median(field: str, accepted_only: bool = False) -> float | None:
            values = [
                float(record[field])
                for record in subset
                if record.get(field) is not None and (not accepted_only or record["accepted"])
            ]
            return statistics.median(values) if values else None

        failures: dict[str, int] = {}
        for record in subset:
            stage = record.get("failure_stage") or "accepted"
            failures[stage] = failures.get(stage, 0) + 1
        result[variant] = {
            "cases": len(subset),
            "accepted": sum(bool(record["accepted"]) for record in subset),
            "rejected": sum(not record["accepted"] for record in subset),
            "pose_available": sum(record.get("translation_error_m") is not None for record in subset),
            "accepted_median_translation_error_m": median("translation_error_m", True),
            "all_pose_median_translation_error_m": median("translation_error_m"),
            "accepted_median_rotation_error_deg_non_symmetric": statistics.median(
                [
                    float(record["rotation_error_deg"])
                    for record in subset
                    if record.get("rotation_error_deg") is not None
                    and record["accepted"]
                    and record.get("symmetry") == "none"
                ]
            )
            if any(
                record.get("rotation_error_deg") is not None
                and record["accepted"]
                and record.get("symmetry") == "none"
                for record in subset
            )
            else None,
            "median_bidirectional_overlap": median("bidirectional_overlap"),
            "median_icp_fitness": median("icp_fitness"),
            "median_runtime_s": median("runtime_s"),
            "failure_stage_counts": failures,
            "candidate_generation_with_candidates": sum(
                int(record.get("candidate_count", 0) or 0) > 0 for record in subset
            ),
        }
    return result


def run(manifest_path: Path, dataset_root: Path, output_root: Path) -> None:
    manifest = _read_json(manifest_path)
    split = str(manifest["split"])
    cases = manifest["cases"]
    variants = ["A_baseline_cad_full", "B_geometry_cad", "C_geometry_reconst"]
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "manifest_copy.json", manifest)
    _write_json(
        output_root / "experiment_config.json",
        {
            "experiment": "frozen_model_conditioned_tless_validation_v1",
            "manifest": str(manifest_path),
            "dataset_root": str(dataset_root),
            "split": split,
            "variants": {variant: vars(_config(variant)) for variant in variants},
            "ground_truth_policy": "GT is read only after inference for evaluation/visualization; never detector input",
            "models_eval_used": False,
            "train_pbr_used": False,
            "yolo_data_used": False,
        },
    )

    records: list[dict[str, Any]] = []
    visual_budget = {"scene000003_im000233_obj000018": {"A_baseline_cad_full", "B_geometry_cad", "C_geometry_reconst"}}
    for case in cases:
        depth, intrinsics, scale_m, _, _ = _load_frame(
            dataset_root, split, int(case["scene_id"]), int(case["image_id"])
        )
        scene = depth_to_point_cloud(depth, intrinsics, depth_scale_m=scale_m)
        truth, _ = _ground_truth_entry(dataset_root, split, case)
        symmetry = _symmetry_status(dataset_root, case)
        for variant in variants:
            representation = "cad" if variant != "C_geometry_reconst" else "reconst"
            model_path = resolve_model_directory(dataset_root, representation) / f"obj_{int(case['model_id']):06d}.ply"
            model = load_model_catalog_entry(
                model_path,
                model_id=str(case["model_id"]),
                units="mm",
                representation=representation,
            )
            started = time.perf_counter()
            hypothesis = GeometryFirstDetector([model], config=_config(variant)).detect(scene)[0]
            runtime_s = time.perf_counter() - started
            diagnostics = hypothesis.diagnostics
            estimated = hypothesis.T_camera_object
            translation_error_m = None if estimated is None else float(np.linalg.norm(estimated[:3, 3] - truth[:3, 3]))
            rotation_error = None if estimated is None else float(rotation_error_degrees(estimated, truth))
            generation = diagnostics.candidate_generation or {}
            selected = generation.get("selected_candidate", {})
            record: dict[str, Any] = {
                "variant": variant,
                "case_id": case["case_id"],
                "scene_id": int(case["scene_id"]),
                "image_id": int(case["image_id"]),
                "model_id": int(case["model_id"]),
                "selection_status": case.get("selection_status"),
                "visibility_fraction": float(case["visibility_fraction"]),
                "symmetry": symmetry["status"],
                "accepted": bool(hypothesis.accepted),
                "failure_stage": _failure_stage(diagnostics.failure_reason),
                "failure_reason": diagnostics.failure_reason,
                "translation_error_m": translation_error_m,
                "translation_error_mm": None if translation_error_m is None else translation_error_m * 1000.0,
                "rotation_error_deg": rotation_error,
                "ransac_inliers": diagnostics.ransac_inliers,
                "ransac_rmse_m": diagnostics.ransac_rmse_m,
                "icp_inliers": diagnostics.icp_inliers,
                "icp_fitness": diagnostics.fitness,
                "icp_rmse_m": diagnostics.rmse_m,
                "feature_correspondences": diagnostics.feature_correspondences,
                "candidate_count": generation.get("candidate_count", 0),
                "candidate_points": diagnostics.candidate_points,
                "candidate_source": diagnostics.candidate_source,
                "selected_candidate": selected,
                "candidate_generation": generation,
                "model_representation": representation,
                "model_normal_source": diagnostics.model_normal_source,
                "observed_normal_orientation": diagnostics.observed_normal_orientation,
                "model_to_candidate_overlap": diagnostics.model_to_candidate_overlap,
                "candidate_to_model_overlap": diagnostics.candidate_to_model_overlap,
                "bidirectional_overlap": diagnostics.bidirectional_overlap,
                "model_to_candidate_rmse_m": diagnostics.model_to_candidate_rmse_m,
                "candidate_to_model_rmse_m": diagnostics.candidate_to_model_rmse_m,
                "model_to_candidate_trimmed_rmse_m": diagnostics.model_to_candidate_trimmed_rmse_m,
                "candidate_to_model_trimmed_rmse_m": diagnostics.candidate_to_model_trimmed_rmse_m,
                "extent_ratio_min": diagnostics.extent_ratio_min,
                "extent_ratio_max": diagnostics.extent_ratio_max,
                "runtime_s": runtime_s,
                "T_camera_object_estimated": None if estimated is None else estimated.tolist(),
                "T_camera_object_ground_truth": truth.tolist(),
                "symmetry_detail": symmetry,
                "model_path": str(model_path),
                "scene_points": len(scene),
                "pose_evaluation": "post_inference_gt_diagnostic_only",
            }
            records.append(record)
            if case["case_id"] in visual_budget and variant in visual_budget[case["case_id"]]:
                _project_overlay(
                    dataset_root,
                    split,
                    case,
                    variant,
                    model_path,
                    estimated,
                    truth,
                    output_root / "visuals" / f"{case['case_id']}_{variant}.png",
                )
            print(
                f"{variant} {case['case_id']}: accepted={hypothesis.accepted} "
                f"candidate={diagnostics.candidate_source}/{diagnostics.candidate_points} "
                f"trans_mm={record['translation_error_mm']} rot_deg={rotation_error} "
                f"reason={diagnostics.failure_reason}",
                flush=True,
            )

    _write_json(output_root / "per_case_results.json", records)
    _write_csv(output_root / "per_case_results.csv", records)
    _write_json(output_root / "aggregate_summary.json", _aggregate(records))
    README = """# Frozen T-LESS model-conditioned validation

This directory was produced from the persisted manifest copy before any CAD vs
reconstructed comparison. It is an engineering validation of known-model
geometric localization, not a complete detector benchmark: each model ID is
supplied by the manifest. Ground truth is read only after inference for error
reporting and the labelled diagnostic overlays.

Variants:

- `A_baseline_cad_full`: CAD, full filtered scene, historical acceptance without bidirectional verification.
- `B_geometry_cad`: CAD, support-plane removal, geometry candidates, explicit normal handling, bidirectional verification.
- `C_geometry_reconst`: reconstructed model with exactly Variant B logic and thresholds.

All geometry is metres internally. The T-LESS model files are millimetres at
the adapter boundary. `models_eval` and `train_pbr` are not used.
"""
    (output_root / "README.md").write_text(README, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.manifest, args.dataset, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
