"""Local BOP/T-LESS inspection, validation, inference, and CSV adapters.

No dataset files are downloaded. BOP/T-LESS values are converted at the
boundary: model vertices and pose translations are millimetres, internal
Open3D geometry and transforms are metres, BOP CSV translations are
millimetres, and BOP CSV time is seconds.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Iterable

import numpy as np

from .io import load_model_catalog_entry
from .registration import GeometryFirstDetector, depth_to_point_cloud, make_transform
from .types import CameraIntrinsics, DetectionConfig


BOP_MODEL_UNITS = "mm"
BOP_TRANSLATION_UNITS = "mm"
INTERNAL_GEOMETRY_UNITS = "m"
BOP_CSV_FIELDS = ("scene_id", "im_id", "obj_id", "score", "R", "t", "time")
MODEL_REPRESENTATION_DIRECTORIES = {"cad": "models_cad", "reconst": "models_reconst"}
_MODEL_RE = re.compile(r"^obj_(\d+)\.ply$", re.IGNORECASE)
_IMAGE_SUFFIXES = (".png", ".tif", ".tiff")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"BOP metadata file does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON metadata file: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"BOP metadata file must contain a JSON object: {path}")
    return value


def _read_depth_image(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("BOP depth input requires Pillow; install with pip install '.[bop]'") from exc
    if not path.is_file():
        raise FileNotFoundError(f"BOP depth image does not exist: {path}")
    try:
        with Image.open(path) as image:
            return np.asarray(image)
    except Exception as exc:
        raise ValueError(f"could not read BOP depth image {path}: {exc}") from exc


def _numeric_ids(path: Path) -> list[int]:
    if not path.is_dir():
        return []
    return sorted(int(item.name) for item in path.iterdir() if item.is_dir() and item.name.isdigit())


def _frame_ids(path: Path) -> list[int]:
    if not path.is_dir():
        return []
    return sorted(
        {
            int(item.stem)
            for item in path.iterdir()
            if item.is_file() and item.suffix.lower() in _IMAGE_SUFFIXES and item.stem.isdigit()
        }
    )


def _model_files(path: Path) -> dict[int, Path]:
    if not path.is_dir():
        return {}
    found: dict[int, Path] = {}
    for item in path.iterdir():
        match = _MODEL_RE.match(item.name)
        if item.is_file() and match:
            found[int(match.group(1))] = item
    return dict(sorted(found.items()))


def resolve_model_directory(dataset_root: str | Path, representation: str = "cad") -> Path:
    """Resolve an explicit BOP inference model directory without fallback."""

    if representation not in MODEL_REPRESENTATION_DIRECTORIES:
        raise ValueError(
            f"unsupported model representation {representation!r}; "
            f"choose one of {sorted(MODEL_REPRESENTATION_DIRECTORIES)}"
        )
    return Path(dataset_root) / MODEL_REPRESENTATION_DIRECTORIES[representation]


def _camera_ids(path: Path) -> tuple[list[int], str | None]:
    if not path.is_file():
        return [], f"missing camera metadata: {path}"
    try:
        camera = _read_json(path)
        return sorted(int(key) for key in camera), None
    except (TypeError, ValueError, FileNotFoundError) as exc:
        return [], str(exc)


def _scene_summary(scene_root: Path) -> dict[str, Any]:
    camera_path = scene_root / "scene_camera.json"
    camera_ids, camera_error = _camera_ids(camera_path)
    gt_path = scene_root / "scene_gt.json"
    gt_ids: list[int] = []
    gt_error: str | None = None
    if gt_path.is_file():
        try:
            gt_ids = sorted(int(key) for key in _read_json(gt_path))
        except (TypeError, ValueError, FileNotFoundError) as exc:
            gt_error = str(exc)
    return {
        "scene_id": int(scene_root.name),
        "path": str(scene_root),
        "camera_metadata": camera_path.is_file(),
        "camera_error": camera_error,
        "camera_frame_ids": camera_ids,
        "camera_frame_count": len(camera_ids),
        "depth_directory": (scene_root / "depth").is_dir(),
        "depth_frame_ids": _frame_ids(scene_root / "depth"),
        "depth_frame_count": len(_frame_ids(scene_root / "depth")),
        "rgb_directory": (scene_root / "rgb").is_dir(),
        "rgb_frame_ids": _frame_ids(scene_root / "rgb"),
        "rgb_frame_count": len(_frame_ids(scene_root / "rgb")),
        "ground_truth_metadata": gt_path.is_file(),
        "ground_truth_frame_ids": gt_ids,
        "ground_truth_error": gt_error,
    }


def inspect_dataset(
    dataset_root: str | Path,
    *,
    split: str | None = None,
    scene_id: int | None = None,
    image_id: int | None = None,
    model_ids: Iterable[int] | None = None,
    model_representation: str = "cad",
) -> dict[str, Any]:
    """Inspect metadata and filenames without loading image pixels."""

    root = Path(dataset_root)
    requested_models = sorted({int(value) for value in (model_ids or [])})
    report: dict[str, Any] = {
        "dataset_root": str(root),
        "dataset_exists": root.is_dir(),
        "expected_layout": {
            "model_directory": f"{MODEL_REPRESENTATION_DIRECTORIES.get(model_representation, 'invalid')}/obj_XXXXXX.ply",
            "split_directory": "test_primesense or test, as present locally",
            "scene_files": ["scene_camera.json", "depth/XXXXXX.png", "scene_gt.json optional for inference"],
        },
        "model_directory": None,
        "models_info": False,
        "available_model_ids": [],
        "model_files": [],
        "available_splits": [],
        "selected": {
            "split": split,
            "scene_id": scene_id,
            "image_id": image_id,
            "model_ids": requested_models,
            "model_representation": model_representation,
        },
        "scenes": {},
        "missing_for_inference": [],
        "missing_for_official_scoring": [],
        "warnings": [],
        "valid": False,
    }
    if not root.is_dir():
        report["missing_for_inference"].append(f"dataset root does not exist or is not a directory: {root}")
        return report

    try:
        model_dir = resolve_model_directory(root, model_representation)
    except ValueError as exc:
        report["missing_for_inference"].append(str(exc))
        return report
    models = _model_files(model_dir)
    report["model_directory"] = str(model_dir)
    report["available_model_ids"] = list(models)
    report["model_files"] = [str(path) for path in models.values()]
    report["models_info"] = (model_dir / "models_info.json").is_file()
    if not model_dir.is_dir():
        report["missing_for_inference"].append(f"missing T-LESS CAD model directory: {model_dir}")
    elif not models:
        report["missing_for_inference"].append(f"no obj_XXXXXX.ply files found in: {model_dir}")
    if not report["models_info"]:
        report["warnings"].append(f"models_info.json is absent: {model_dir / 'models_info.json'}")
        report["missing_for_official_scoring"].append(f"{model_dir.name}/models_info.json")

    split_dirs = sorted(
        item for item in root.iterdir() if item.is_dir() and item.name.startswith(("train", "val", "test"))
    )
    report["available_splits"] = [item.name for item in split_dirs]
    if not split_dirs:
        report["missing_for_inference"].append(f"no train/val/test split directories found below: {root}")
    for split_dir in split_dirs:
        scenes = {
            str(scene): _scene_summary(split_dir / f"{scene:06d}")
            for scene in _numeric_ids(split_dir)
        }
        report["scenes"][split_dir.name] = {
            "path": str(split_dir),
            "scene_ids": [int(value) for value in scenes],
            "scene_count": len(scenes),
            "scenes": scenes,
        }

    if split is not None:
        split_root = root / split
        selected = report["scenes"].get(split)
        if selected is None:
            report["missing_for_inference"].append(f"requested split does not exist: {split_root}")
        elif not selected["scene_ids"]:
            report["missing_for_inference"].append(f"requested split has no numeric scenes: {split_root}")
        if scene_id is not None:
            scene_root = split_root / f"{scene_id:06d}"
            scene = selected["scenes"].get(str(scene_id)) if selected else None
            if scene is None:
                report["missing_for_inference"].append(f"requested scene does not exist: {scene_root}")
            elif image_id is not None:
                if image_id not in scene["camera_frame_ids"]:
                    report["missing_for_inference"].append(
                        f"image {image_id} is absent from {scene_root / 'scene_camera.json'}"
                    )
                if image_id not in scene["depth_frame_ids"]:
                    report["missing_for_inference"].append(
                        f"depth frame {image_id:06d} is absent from {scene_root / 'depth'}"
                    )
                if not scene["camera_metadata"]:
                    report["missing_for_inference"].append(
                        f"camera metadata is absent: {scene_root / 'scene_camera.json'}"
                    )
                if not scene["ground_truth_metadata"]:
                    report["missing_for_official_scoring"].append(str(scene_root / "scene_gt.json"))
                elif image_id not in scene["ground_truth_frame_ids"]:
                    report["missing_for_official_scoring"].append(
                        f"{scene_root / 'scene_gt.json'} has no image {image_id}"
                    )
    for model_id in requested_models:
        if model_id not in models:
            report["missing_for_inference"].append(
                f"requested model {model_id} is absent: {model_dir / f'obj_{model_id:06d}.ply'}"
            )
    report["valid"] = not report["missing_for_inference"]
    return report


def _depth_path(scene_root: Path, image_id: int) -> Path:
    for suffix in _IMAGE_SUFFIXES:
        path = scene_root / "depth" / f"{image_id:06d}{suffix}"
        if path.is_file():
            return path
    return scene_root / "depth" / f"{image_id:06d}.png"


def _camera_entry(root: Path, split: str, scene_id: int, image_id: int) -> tuple[Path, dict[str, Any]]:
    scene_root = root / split / f"{scene_id:06d}"
    camera_path = scene_root / "scene_camera.json"
    camera = _read_json(camera_path)
    entry = camera.get(str(image_id), camera.get(image_id))
    if not isinstance(entry, dict):
        raise ValueError(f"scene {scene_id} has no camera metadata for image {image_id}: {camera_path}")
    return scene_root, entry


def _load_frame(
    root: Path,
    split: str,
    scene_id: int,
    image_id: int,
) -> tuple[np.ndarray, CameraIntrinsics, float, dict[str, Any], Path]:
    scene_root, camera_data = _camera_entry(root, split, scene_id, image_id)
    matrix = np.asarray(camera_data.get("cam_K"), dtype=float)
    if matrix.size != 9:
        raise ValueError(f"cam_K must contain 9 values for scene {scene_id}, image {image_id}")
    intrinsics = CameraIntrinsics.from_matrix(matrix.reshape(3, 3))
    if "depth_scale" not in camera_data:
        raise ValueError(f"camera metadata has no depth_scale for scene {scene_id}, image {image_id}")
    depth_scale_to_mm = float(camera_data["depth_scale"])
    if not np.isfinite(depth_scale_to_mm) or depth_scale_to_mm <= 0.0:
        raise ValueError("depth_scale must be positive finite millimetres-per-stored-unit")
    depth_path = _depth_path(scene_root, image_id)
    depth = _read_depth_image(depth_path)
    if depth.ndim != 2:
        raise ValueError(f"BOP depth image must be single-channel 2-D: {depth_path}")
    return depth, intrinsics, depth_scale_to_mm * 1e-3, camera_data, depth_path


def bop_pose_to_transform(
    cam_R_m2c: np.ndarray | list[float] | list[list[float]],
    cam_t_m2c: np.ndarray | list[float],
    *,
    translation_units: str = BOP_TRANSLATION_UNITS,
) -> np.ndarray:
    """Convert row-wise BOP model-to-camera pose to internal metres."""

    rotation = np.asarray(cam_R_m2c, dtype=float)
    if rotation.size != 9:
        raise ValueError("cam_R_m2c must contain exactly 9 values")
    rotation = rotation.reshape(3, 3)
    translation = np.asarray(cam_t_m2c, dtype=float)
    if translation.size != 3:
        raise ValueError("cam_t_m2c must contain exactly 3 values")
    translation = translation.reshape(3)
    if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
        raise ValueError("BOP pose values must be finite")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) <= 0.0:
        raise ValueError("cam_R_m2c must be a proper rotation matrix")
    if translation_units.lower() in {"mm", "millimeter", "millimeters"}:
        scale_to_m = 1e-3
    elif translation_units.lower() in {"m", "meter", "meters"}:
        scale_to_m = 1.0
    else:
        raise ValueError("translation_units must be 'mm' or 'm'")
    return make_transform(rotation, translation * scale_to_m)


def transform_to_bop_pose(
    transform: np.ndarray,
    *,
    translation_units: str = BOP_TRANSLATION_UNITS,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert internal T_camera_object metres to row-wise BOP R and t."""

    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("transform must be a finite 4x4 matrix")
    rotation = matrix[:3, :3]
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise ValueError("transform must have homogeneous last row [0, 0, 0, 1]")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) <= 0.0:
        raise ValueError("transform rotation must be a proper rotation matrix")
    if translation_units.lower() in {"mm", "millimeter", "millimeters"}:
        scale_from_m = 1e3
    elif translation_units.lower() in {"m", "meter", "meters"}:
        scale_from_m = 1.0
    else:
        raise ValueError("translation_units must be 'mm' or 'm'")
    return rotation.copy(), matrix[:3, 3].copy() * scale_from_m


def load_ground_truth_pose(
    dataset_root: str | Path,
    split: str,
    scene_id: int,
    image_id: int,
    model_id: int,
) -> np.ndarray | None:
    """Load one optional BOP GT pose as T_camera_object."""

    gt_path = Path(dataset_root) / split / f"{scene_id:06d}" / "scene_gt.json"
    if not gt_path.is_file():
        return None
    entries = _read_json(gt_path).get(str(image_id))
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if isinstance(entry, dict) and int(entry.get("obj_id", -1)) == int(model_id):
            return bop_pose_to_transform(entry["cam_R_m2c"], entry["cam_t_m2c"])
    return None


def validate_tless_experiment(
    dataset_root: str | Path,
    *,
    split: str,
    scene_id: int,
    image_id: int,
    model_ids: Iterable[int],
    output: str | Path,
    model_representation: str = "cad",
) -> dict[str, Any]:
    """Validate one case and report exactly what registration would load."""

    requested = sorted({int(value) for value in model_ids})
    report = inspect_dataset(
        dataset_root,
        split=split,
        scene_id=scene_id,
        image_id=image_id,
        model_ids=requested,
        model_representation=model_representation,
    )
    output_path = Path(output)
    report["units"] = {
        "bop_depth_scale": "stored depth unit multiplied to get millimetres",
        "bop_model_vertices": "millimetres",
        "bop_ground_truth_translation": "millimetres",
        "internal_geometry_and_pose": "metres",
        "bop_csv_translation": "millimetres",
        "bop_csv_time": "seconds",
    }
    report["model_representation"] = model_representation
    report["output"] = {
        "path": str(output_path),
        "exists": output_path.exists(),
        "parent": str(output_path.parent),
        "parent_exists": output_path.parent.is_dir(),
        "will_create_parent": not output_path.parent.exists(),
    }
    errors = list(report["missing_for_inference"])
    if output_path.exists() and output_path.is_dir():
        errors.append(f"output path is a directory, expected a CSV file: {output_path}")
    if not output_path.parent.is_dir():
        report["warnings"].append(
            f"output parent directory does not exist and will be created: {output_path.parent}"
        )
    elif not os.access(output_path.parent, os.W_OK):
        errors.append(f"output parent directory is not writable: {output_path.parent}")
    if output_path.suffix.lower() != ".csv":
        report["warnings"].append(f"official BOP output normally uses .csv: {output_path}")

    report["selected_frame"] = {}
    report["models"] = []
    if not errors:
        try:
            depth, intrinsics, depth_scale_m, camera_data, depth_path = _load_frame(
                Path(dataset_root), split, scene_id, image_id
            )
            positive = int(np.count_nonzero(np.isfinite(depth) & (depth > 0)))
            if positive == 0:
                errors.append(f"selected depth frame contains no positive finite values: {depth_path}")
            report["selected_frame"] = {
                "depth_path": str(depth_path),
                "depth_shape": list(depth.shape),
                "depth_dtype": str(depth.dtype),
                "positive_depth_pixels": positive,
                "intrinsics": {
                    "fx": intrinsics.fx,
                    "fy": intrinsics.fy,
                    "cx": intrinsics.cx,
                    "cy": intrinsics.cy,
                },
                "depth_scale_to_mm": float(camera_data["depth_scale"]),
                "depth_scale_to_m": depth_scale_m,
                "internal_scene_units": INTERNAL_GEOMETRY_UNITS,
            }
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            errors.append(str(exc))
    model_dir = resolve_model_directory(dataset_root, model_representation)
    for model_id in requested:
        path = model_dir / f"obj_{model_id:06d}.ply"
        item: dict[str, Any] = {"model_id": model_id, "path": str(path)}
        if not path.is_file():
            errors.append(f"model file does not exist: {path}")
        else:
            try:
                model = load_model_catalog_entry(path, model_id=str(model_id), units=BOP_MODEL_UNITS)
                item.update(
                    {
                        "point_count": len(model.points),
                        "bounds_m": [model.bounds_m[0].tolist(), model.bounds_m[1].tolist()],
                        "source_units": BOP_MODEL_UNITS,
                        "internal_units": INTERNAL_GEOMETRY_UNITS,
                        "representation": model_representation,
                    }
                )
            except (FileNotFoundError, ValueError) as exc:
                errors.append(str(exc))
        report["models"].append(item)
    report["missing_for_inference"] = errors
    report["valid"] = not errors
    report["would_load"] = {
        "depth": report["selected_frame"].get("depth_path"),
        "models": [item["path"] for item in report["models"]],
        "registration": "one selected scene/image against each selected model",
        "official_csv": str(output_path),
        "diagnostics_jsonl": str(output_path.with_suffix(".diagnostics.jsonl")),
    }
    return report


def _diagnostic_record(*, scene_id: int, image_id: int, model_id: int, elapsed_seconds: float, hypothesis: Any) -> dict[str, Any]:
    diagnostics = asdict(hypothesis.diagnostics)
    return {
        "scene_id": scene_id,
        "im_id": image_id,
        "obj_id": model_id,
        "accepted": bool(hypothesis.accepted),
        "candidate_type": diagnostics["candidate_source"],
        "runtime_ms": elapsed_seconds * 1000.0,
        "failure_reason": diagnostics["failure_reason"],
        "ransac_inliers": diagnostics["ransac_inliers"],
        "ransac_rmse_m": diagnostics["ransac_rmse_m"],
        "icp_inliers": diagnostics["icp_inliers"],
        "icp_fitness": diagnostics["fitness"],
        "icp_rmse_m": diagnostics["rmse_m"],
        "inlier_ratio": diagnostics["inlier_ratio"],
        "feature_correspondences": diagnostics["feature_correspondences"],
        "T_camera_object": (
            hypothesis.T_camera_object.tolist() if hypothesis.T_camera_object is not None else None
        ),
        "confidence": hypothesis.confidence,
        "model_representation": diagnostics.get("model_representation", "custom"),
        "diagnostics": diagnostics,
    }


def _write_diagnostics(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")


def _write_bop_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(BOP_CSV_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def _selected_tasks(args: argparse.Namespace, root: Path) -> list[tuple[int, int]]:
    scene_id = getattr(args, "scene_id", None)
    image_id = getattr(args, "im_id", None)
    max_frames = getattr(args, "max_frames", None)
    if (scene_id is None) != (image_id is None):
        raise ValueError("--scene-id and --image-id must be supplied together")
    if scene_id is not None:
        return [(int(scene_id), int(image_id))]
    if max_frames is None or max_frames <= 0:
        raise ValueError(
            "select exactly one --scene-id and --image-id; bounded legacy scanning requires positive --max-frames"
        )
    tasks: list[tuple[int, int]] = []
    split_root = root / args.split
    for current_scene in _numeric_ids(split_root):
        for current_image in _frame_ids(split_root / f"{current_scene:06d}" / "depth"):
            tasks.append((current_scene, current_image))
            if len(tasks) >= max_frames:
                return tasks
    raise ValueError(f"no depth frames found under {split_root}")


def run(args: argparse.Namespace) -> int:
    root = Path(args.dataset_root)
    model_ids = [int(value) for value in args.model_id]
    if getattr(args, "dry_run", False):
        if args.scene_id is None or args.im_id is None:
            raise ValueError("dry-run requires exactly one --scene-id and --image-id")
        report = validate_tless_experiment(
            root,
            split=args.split,
            scene_id=args.scene_id,
            image_id=args.im_id,
            model_ids=model_ids,
            output=args.output,
            model_representation=args.model_representation,
        )
        print(json.dumps(report, indent=2))
        return 0 if report["valid"] else 2

    tasks = _selected_tasks(args, root)
    if len(tasks) != 1:
        raise ValueError("registration requires one explicit scene/image; use tless for controlled experiments")
    scene_id, image_id = tasks[0]
    validation = validate_tless_experiment(
        root,
        split=args.split,
        scene_id=scene_id,
        image_id=image_id,
        model_ids=model_ids,
        output=args.output,
        model_representation=args.model_representation,
    )
    if not validation["valid"]:
        raise ValueError("T-LESS experiment validation failed:\n- " + "\n- ".join(validation["missing_for_inference"]))

    depth, intrinsics, depth_scale_m, _, _ = _load_frame(root, args.split, scene_id, image_id)
    scene = depth_to_point_cloud(depth, intrinsics, depth_scale_m=depth_scale_m)
    model_dir = resolve_model_directory(root, args.model_representation)
    catalogue = [
        load_model_catalog_entry(
            model_dir / f"obj_{model_id:06d}.ply",
            model_id=str(model_id),
            units="mm",
            representation=args.model_representation,
        )
        for model_id in model_ids
    ]
    config = DetectionConfig(
        voxel_size_m=args.voxel_size_m,
        ransac_inlier_threshold_m=args.ransac_threshold_m,
        icp_max_correspondence_m=args.icp_threshold_m,
        random_seed=args.seed,
        candidate_mode=args.candidate_mode,
        model_representation=args.model_representation,
        model_normal_source=args.model_normal_source,
        observed_normal_orientation=args.observed_normal_orientation,
        verification_mode=args.verification_mode,
        min_bidirectional_overlap=args.min_bidirectional_overlap,
        remove_support_plane=args.remove_support_plane,
    )
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for model in catalogue:
        detector = GeometryFirstDetector([model], config=config)
        started = time.perf_counter()
        hypothesis = detector.detect(scene, camera_frame="cam", timestamp=float(image_id))[0]
        elapsed_seconds = time.perf_counter() - started
        records.append(
            _diagnostic_record(
                scene_id=scene_id,
                image_id=image_id,
                model_id=int(model.model_id),
                elapsed_seconds=elapsed_seconds,
                hypothesis=hypothesis,
            )
        )
        if hypothesis.accepted and hypothesis.T_camera_object is not None:
            rotation, translation_mm = transform_to_bop_pose(hypothesis.T_camera_object)
            rows.append(
                {
                    "scene_id": scene_id,
                    "im_id": image_id,
                    "obj_id": int(model.model_id),
                    "score": f"{hypothesis.confidence:.8f}",
                    "R": " ".join(f"{value:.9g}" for value in rotation.reshape(-1)),
                    "t": " ".join(f"{value:.9g}" for value in translation_mm),
                    "time": f"{elapsed_seconds:.9f}",
                }
            )
    output = Path(args.output)
    diagnostics_output = Path(
        args.diagnostics_output
        if getattr(args, "diagnostics_output", None)
        else output.with_suffix(".diagnostics.jsonl")
    )
    _write_bop_csv(output, rows)
    _write_diagnostics(diagnostics_output, records)
    print(
        json.dumps(
            {
                "dataset_root": str(root),
                "split": args.split,
                "scene_id": scene_id,
                "im_id": image_id,
                "model_ids": model_ids,
                "model_representation": args.model_representation,
                "processed": 1,
                "accepted": sum(1 for item in records if item["accepted"]),
                "official_bop_csv": str(output),
                "diagnostics_jsonl": str(diagnostics_output),
                "translation_units": BOP_TRANSLATION_UNITS,
                "internal_units": INTERNAL_GEOMETRY_UNITS,
            },
            indent=2,
        )
    )
    return 0


def run_inspect(args: argparse.Namespace) -> int:
    report = inspect_dataset(
        args.dataset_root,
        split=args.split,
        scene_id=args.scene_id,
        image_id=args.im_id,
        model_ids=args.model_id,
        model_representation=args.model_representation,
    )
    if args.report_output:
        report_path = Path(args.report_output)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["valid"] else 2


def add_arguments(parser: argparse.ArgumentParser, *, require_selection: bool = False) -> None:
    parser.add_argument(
        "--dataset",
        "--dataset-root",
        dest="dataset_root",
        required=True,
        help="local BOP/T-LESS dataset root; no files are downloaded",
    )
    parser.add_argument("--split", default="test_primesense")
    parser.add_argument("--scene-id", type=int, required=require_selection)
    parser.add_argument("--image-id", "--im-id", dest="im_id", type=int, required=require_selection)
    parser.add_argument("--model-id", type=int, action="append", required=True)
    parser.add_argument("--model-representation", choices=["cad", "reconst"], default="cad")
    parser.add_argument(
        "--max-frames",
        type=int,
        help="bounded legacy scan when scene/image are omitted; never defaults to the whole dataset",
    )
    parser.add_argument("--output", required=require_selection, default=None if require_selection else "bop_predictions.csv")
    parser.add_argument("--diagnostics-output")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the selected case without registration")
    parser.add_argument("--voxel-size-m", type=float, default=0.005)
    parser.add_argument("--ransac-threshold-m", type=float, default=0.015)
    parser.add_argument("--icp-threshold-m", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--candidate-mode",
        choices=["full", "dbscan", "full_and_dbscan", "geometry_candidates"],
        default="geometry_candidates",
    )
    parser.add_argument(
        "--remove-support-plane",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="remove a dominant support plane before geometry candidate generation",
    )
    parser.add_argument("--model-normal-source", choices=["auto", "mesh", "estimate"], default="auto")
    parser.add_argument(
        "--observed-normal-orientation",
        choices=["none", "towards_camera", "away_from_camera"],
        default="towards_camera",
    )
    parser.add_argument("--verification-mode", choices=["none", "bidirectional"], default="bidirectional")
    parser.add_argument("--min-bidirectional-overlap", type=float, default=0.20)


def add_inspection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", "--dataset-root", dest="dataset_root", required=True)
    parser.add_argument("--split")
    parser.add_argument("--scene-id", type=int)
    parser.add_argument("--image-id", "--im-id", dest="im_id", type=int)
    parser.add_argument("--model-id", type=int, action="append")
    parser.add_argument("--model-representation", choices=["cad", "reconst"], default="cad")
    parser.add_argument("--report-output")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
