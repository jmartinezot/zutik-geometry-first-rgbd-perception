"""Command line entry points for the geometry-first vertical slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .io import (
    load_depth_array,
    load_model_catalog_entry,
    load_point_cloud,
    load_transform,
    write_hypotheses,
)
from .registration import GeometryFirstDetector, depth_to_point_cloud
from .synthetic import make_synthetic_scene_case, rotation_error_degrees
from .types import CameraIntrinsics, DetectionConfig


def _config_from_args(args: argparse.Namespace) -> DetectionConfig:
    return DetectionConfig(
        voxel_size_m=args.voxel_size_m,
        ransac_inlier_threshold_m=args.ransac_threshold_m,
        icp_max_correspondence_m=args.icp_threshold_m,
        random_seed=args.seed,
        remove_support_plane=args.remove_support_plane,
        candidate_mode=args.candidate_mode,
        model_representation=getattr(args, "model_representation", "custom"),
        model_normal_source=getattr(args, "model_normal_source", "auto"),
        observed_normal_orientation=getattr(args, "observed_normal_orientation", "towards_camera"),
        verification_mode=getattr(args, "verification_mode", "bidirectional"),
        min_bidirectional_overlap=getattr(args, "min_bidirectional_overlap", 0.20),
    )


def _add_registration_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--voxel-size-m", type=float, default=0.005)
    parser.add_argument("--ransac-threshold-m", type=float, default=0.015)
    parser.add_argument("--icp-threshold-m", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--remove-support-plane",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--candidate-mode",
        choices=["full", "dbscan", "full_and_dbscan", "geometry_candidates"],
        default="geometry_candidates",
    )
    parser.add_argument("--model-representation", choices=["custom", "cad", "reconst"], default="custom")
    parser.add_argument("--model-normal-source", choices=["auto", "mesh", "estimate"], default="auto")
    parser.add_argument(
        "--observed-normal-orientation",
        choices=["none", "towards_camera", "away_from_camera"],
        default="towards_camera",
    )
    parser.add_argument("--verification-mode", choices=["none", "bidirectional"], default="bidirectional")
    parser.add_argument("--min-bidirectional-overlap", type=float, default=0.20)


def _run_detect(args: argparse.Namespace) -> int:
    if bool(args.scene) == bool(args.depth):
        raise ValueError("provide exactly one of --scene or --depth")
    if args.scene:
        scene = load_point_cloud(args.scene)
    else:
        if None in (args.fx, args.fy, args.cx, args.cy):
            raise ValueError("--depth requires --fx, --fy, --cx, and --cy")
        scene = depth_to_point_cloud(
            load_depth_array(args.depth),
            CameraIntrinsics(args.fx, args.fy, args.cx, args.cy),
            depth_scale_m=args.depth_scale_m,
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
        )
    catalogue = [
        load_model_catalog_entry(
            path,
            model_id=args.model_id[i] if args.model_id and i < len(args.model_id) else None,
            semantic_label=args.semantic_label[i] if args.semantic_label and i < len(args.semantic_label) else None,
            units=args.model_units,
            representation=args.model_representation,
        )
        for i, path in enumerate(args.model)
    ]
    world_camera = load_transform(args.world_camera) if args.world_camera else None
    detector = GeometryFirstDetector(catalogue, config=_config_from_args(args))
    hypotheses = detector.detect(
        scene,
        T_world_camera=world_camera,
        camera_frame=args.camera_frame,
        world_frame=args.world_frame if world_camera is not None else None,
        timestamp=args.timestamp,
    )
    if args.output:
        write_hypotheses(args.output, hypotheses)
    print(json.dumps({"hypotheses": [hypothesis.to_dict() for hypothesis in hypotheses]}, indent=2))
    return 0 if any(hypothesis.accepted for hypothesis in hypotheses) else 2


def _run_synthetic(args: argparse.Namespace) -> int:
    model, scene, expected = make_synthetic_scene_case()
    config = DetectionConfig(
        voxel_size_m=0.006,
        ransac_inlier_threshold_m=0.012,
        icp_max_correspondence_m=0.018,
        max_rmse_m=0.012,
        random_seed=19,
        candidate_mode="dbscan",
        remove_support_plane=True,
        cluster_eps_m=0.045,
        cluster_min_points=20,
    )
    hypothesis = GeometryFirstDetector([model], config=config).detect(scene)[0]
    translation_error = None
    rotation_error = None
    if hypothesis.T_camera_object is not None:
        translation_error = float(np.linalg.norm(hypothesis.T_camera_object[:3, 3] - expected[:3, 3]))
        rotation_error = rotation_error_degrees(hypothesis.T_camera_object, expected)
    payload = {
        "expected_T_camera_object": expected.tolist(),
        "hypothesis": hypothesis.to_dict(),
        "translation_error_m": translation_error,
        "rotation_error_deg": rotation_error,
        "notes": "T_A_B maps coordinates from B into A; translations are metres.",
    }
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if hypothesis.accepted else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect = subparsers.add_parser("detect", help="register known models against a point cloud or depth image")
    source = detect.add_mutually_exclusive_group(required=False)
    source.add_argument("--scene", help="scene point cloud (.npy/.npz/.ply/.xyz/.obj)")
    source.add_argument("--depth", help="depth image stored as .npy/.npz")
    detect.add_argument("--model", action="append", required=True, help="catalogue model; repeat for multiple models")
    detect.add_argument("--model-id", action="append", help="stable model ID, in --model order")
    detect.add_argument("--semantic-label", action="append", help="optional label, in --model order")
    detect.add_argument("--model-units", default="m", choices=["m", "cm", "mm"])
    detect.add_argument("--fx", type=float)
    detect.add_argument("--fy", type=float)
    detect.add_argument("--cx", type=float)
    detect.add_argument("--cy", type=float)
    detect.add_argument("--depth-scale-m", type=float, default=1.0)
    detect.add_argument("--min-depth-m", type=float, default=0.05)
    detect.add_argument("--max-depth-m", type=float, default=10.0)
    detect.add_argument("--world-camera", help="4x4 T_world_camera (.npy/.json/.txt)")
    detect.add_argument("--camera-frame", default="camera")
    detect.add_argument("--world-frame", default="world")
    detect.add_argument("--timestamp", type=float)
    detect.add_argument("--output", help="write structured JSON result")
    _add_registration_options(detect)
    detect.set_defaults(handler=_run_detect)

    synthetic = subparsers.add_parser("synthetic", help="run the deterministic registration example")
    synthetic.add_argument("--output", help="write the example result JSON")
    synthetic.set_defaults(handler=_run_synthetic)

    bop = subparsers.add_parser("bop", help="run the bounded local BOP/T-LESS adapter")
    from .bop import add_arguments, add_inspection_arguments

    add_arguments(bop, require_selection=False)
    bop.set_defaults(handler=_run_bop_namespace)

    tless = subparsers.add_parser(
        "tless",
        help="run exactly one selected local T-LESS scene/image against selected models",
    )
    add_arguments(tless, require_selection=True)
    tless.set_defaults(handler=_run_bop_namespace)

    tless_dry_run = subparsers.add_parser(
        "tless-dry-run",
        help="validate one selected T-LESS experiment without registration",
    )
    add_arguments(tless_dry_run, require_selection=True)
    tless_dry_run.set_defaults(handler=_run_tless_dry_run)

    tless_inspect = subparsers.add_parser(
        "tless-inspect",
        help="inspect local T-LESS metadata and filenames without loading images",
    )
    add_inspection_arguments(tless_inspect)
    tless_inspect.set_defaults(handler=_run_tless_inspect)
    return parser


def _run_bop_namespace(args: argparse.Namespace) -> int:
    from .bop import run as bop_run

    return bop_run(args)


def _run_tless_dry_run(args: argparse.Namespace) -> int:
    args.dry_run = True
    return _run_bop_namespace(args)


def _run_tless_inspect(args: argparse.Namespace) -> int:
    from .bop import run_inspect

    return run_inspect(args)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
