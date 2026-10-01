import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import zutik_geometry.bop as bop_module
from zutik_geometry.bop import (
    BOP_CSV_FIELDS,
    bop_pose_to_transform,
    inspect_dataset,
    load_ground_truth_pose,
    resolve_model_directory,
    transform_to_bop_pose,
    validate_tless_experiment,
)
from zutik_geometry.cli import build_parser, main
from zutik_geometry.registration import depth_to_point_cloud, make_transform
from zutik_geometry.types import CameraIntrinsics, PoseHypothesis, RegistrationDiagnostics


def _write_ply(path: Path, points_mm: np.ndarray, normals: np.ndarray | None = None) -> None:
    if normals is None:
        rows = "\n".join(" ".join(f"{value:.6f}" for value in point) for point in points_mm)
        properties = ["property float x", "property float y", "property float z"]
    else:
        rows = "\n".join(
            " ".join(f"{value:.6f}" for value in np.concatenate((point, normal)))
            for point, normal in zip(points_mm, normals)
        )
        properties = [
            "property float x", "property float y", "property float z",
            "property float nx", "property float ny", "property float nz",
        ]
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                f"element vertex {len(points_mm)}",
                *properties,
                "end_header",
                rows,
                "",
            ]
        ),
        encoding="ascii",
    )


def _write_fixture(
    root: Path,
    *,
    include_model: bool = True,
    include_camera: bool = True,
    include_depth: bool = True,
    include_ground_truth: bool = True,
) -> Path:
    model_dir = root / "models_cad"
    split_dir = root / "test_primesense"
    scene_dir = split_dir / "000001"
    model_dir.mkdir(parents=True)
    scene_dir.mkdir(parents=True)
    (model_dir / "models_info.json").write_text(
        json.dumps({"1": {"diameter": 10.0}}), encoding="utf-8"
    )
    if include_model:
        _write_ply(
            model_dir / "obj_000001.ply",
            np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]], dtype=float),
        )
    camera = {
        "0": {
            "cam_K": [100.0, 0.0, 0.5, 0.0, 100.0, 0.5, 0.0, 0.0, 1.0],
            "depth_scale": 1.0,
        }
    }
    if include_camera:
        (scene_dir / "scene_camera.json").write_text(json.dumps(camera), encoding="utf-8")
    if include_depth:
        (scene_dir / "depth").mkdir()
        Image.fromarray(np.array([[1000, 1000], [1000, 0]], dtype=np.uint16)).save(
            scene_dir / "depth" / "000000.png"
        )
    if include_ground_truth:
        (scene_dir / "scene_gt.json").write_text(
            json.dumps(
                {
                    "0": [
                        {
                            "obj_id": 1,
                            "cam_R_m2c": np.eye(3).reshape(-1).tolist(),
                            "cam_t_m2c": [1000.0, -20.0, 500.0],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
    return root


def test_pose_conversion_identity_translation_rotation_and_round_trip():
    identity = bop_pose_to_transform(np.eye(3), [1000.0, -20.0, 500.0])
    np.testing.assert_allclose(identity[:3, 3], [1.0, -0.02, 0.5])
    np.testing.assert_allclose(identity[:3, :3], np.eye(3))

    angle = 0.37
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transform = make_transform(rotation, [0.12, -0.34, 0.56])
    bop_rotation, bop_translation = transform_to_bop_pose(transform)
    np.testing.assert_allclose(bop_rotation, rotation)
    np.testing.assert_allclose(bop_translation, [120.0, -340.0, 560.0])
    np.testing.assert_allclose(bop_pose_to_transform(bop_rotation, bop_translation), transform)


def test_generated_fixture_inspection_is_metadata_only_and_reports_counts(tmp_path, monkeypatch):
    root = _write_fixture(tmp_path / "tless")

    def fail_if_image_loaded(_path):
        raise AssertionError("metadata inspection must not load image pixels")

    monkeypatch.setattr(bop_module, "_read_depth_image", fail_if_image_loaded)
    report = inspect_dataset(root)

    assert report["valid"]
    assert report["available_splits"] == ["test_primesense"]
    assert report["available_model_ids"] == [1]
    scene = report["scenes"]["test_primesense"]["scenes"]["1"]
    assert scene["camera_frame_count"] == 1
    assert scene["depth_frame_count"] == 1
    assert scene["ground_truth_metadata"]


def test_model_representation_resolution_is_explicit_without_fallback(tmp_path):
    root = tmp_path / "tless"
    (root / "models_cad").mkdir(parents=True)
    (root / "models_reconst").mkdir()
    assert resolve_model_directory(root, "cad") == root / "models_cad"
    assert resolve_model_directory(root, "reconst") == root / "models_reconst"
    with pytest.raises(ValueError, match="unsupported model representation"):
        resolve_model_directory(root, "eval")


def test_dry_run_validates_frame_intrinsics_depth_scale_and_model_geometry(tmp_path):
    root = _write_fixture(tmp_path / "tless")
    output = tmp_path / "output" / "one-case.csv"
    output.parent.mkdir()
    report = validate_tless_experiment(
        root,
        split="test_primesense",
        scene_id=1,
        image_id=0,
        model_ids=[1],
        output=output,
    )

    assert report["valid"]
    assert report["selected_frame"]["depth_scale_to_mm"] == 1.0
    assert report["selected_frame"]["depth_scale_to_m"] == 0.001
    assert report["models"][0]["source_units"] == "mm"
    assert report["models"][0]["bounds_m"][1] == [0.01, 0.01, 0.01]
    assert report["would_load"]["registration"].startswith("one selected")

    depth, intrinsics, scale_m, _, _ = bop_module._load_frame(root, "test_primesense", 1, 0)
    cloud = depth_to_point_cloud(depth, intrinsics, depth_scale_m=scale_m)
    assert np.max(cloud[:, 2]) == 1.0


def test_inspect_and_dry_run_cli_commands_work_on_generated_fixture(tmp_path, capsys):
    root = _write_fixture(tmp_path / "tless")
    assert main(["tless-inspect", "--dataset", str(root)]) == 0
    assert main(
        [
            "tless-dry-run",
            "--dataset",
            str(root),
            "--split",
            "test_primesense",
            "--scene-id",
            "1",
            "--image-id",
            "0",
            "--model-id",
            "1",
            "--output",
            str(tmp_path / "out" / "case.csv"),
        ]
    ) == 0
    assert "depth_scale_to_m" in capsys.readouterr().out


def test_incomplete_fixture_fails_clearly_without_writing(tmp_path):
    root = _write_fixture(
        tmp_path / "incomplete",
        include_camera=False,
        include_depth=False,
        include_model=False,
    )
    output = tmp_path / "result.csv"
    report = validate_tless_experiment(
        root,
        split="test_primesense",
        scene_id=1,
        image_id=0,
        model_ids=[1],
        output=output,
    )

    assert not report["valid"]
    assert any("camera metadata" in message for message in report["missing_for_inference"])
    assert any("depth frame" in message for message in report["missing_for_inference"])
    assert any("model" in message for message in report["missing_for_inference"])
    assert not output.exists()
    assert main(["tless-inspect", "--dataset", str(root), "--split", "test_primesense"]) == 2


def test_ground_truth_pose_matches_internal_camera_object_convention(tmp_path):
    root = _write_fixture(tmp_path / "tless")
    expected = make_transform(np.eye(3), [1.0, -0.02, 0.5])
    loaded = load_ground_truth_pose(root, "test_primesense", 1, 0, 1)
    assert loaded is not None
    np.testing.assert_allclose(loaded, expected)


def test_tless_run_keeps_official_csv_separate_from_diagnostics(tmp_path, monkeypatch):
    root = _write_fixture(tmp_path / "tless")
    transform = make_transform(np.eye(3), [1.0, -0.02, 0.5])
    diagnostics = RegistrationDiagnostics(
        method="test",
        accepted=True,
        failure_reason=None,
        model_points=4,
        scene_points=3,
        candidate_points=3,
        candidate_source="full",
        candidate_index=0,
        feature_correspondences=8,
        ransac_inliers=7,
        icp_inliers=6,
        inlier_ratio=0.875,
        global_fitness=0.8,
        fitness=0.75,
        ransac_rmse_m=0.001,
        rmse_m=0.002,
        icp_iterations=3,
        downsample_voxel_m=0.005,
    )
    hypothesis = PoseHypothesis(
        model_id="1",
        semantic_label=None,
        T_camera_object=transform,
        T_world_object=None,
        camera_frame="cam",
        world_frame=None,
        timestamp=0.0,
        confidence=0.75,
        accepted=True,
        diagnostics=diagnostics,
    )

    class FakeDetector:
        def __init__(self, _catalogue, config):
            self.config = config

        def detect(self, _scene, **_kwargs):
            return [hypothesis]

    monkeypatch.setattr(bop_module, "GeometryFirstDetector", FakeDetector)
    parser = build_parser()
    args = parser.parse_args(
        [
            "tless",
            "--dataset",
            str(root),
            "--split",
            "test_primesense",
            "--scene-id",
            "1",
            "--image-id",
            "0",
            "--model-id",
            "1",
            "--output",
            str(tmp_path / "results" / "geometry_tless-test.csv"),
        ]
    )
    assert args.handler(args) == 0

    csv_path = tmp_path / "results" / "geometry_tless-test.csv"
    diagnostics_path = csv_path.with_suffix(".diagnostics.jsonl")
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
        assert tuple(rows[0].keys()) == BOP_CSV_FIELDS
    assert rows[0]["scene_id"] == "1"
    assert rows[0]["im_id"] == "0"
    assert rows[0]["obj_id"] == "1"
    assert rows[0]["t"] == "1000 -20 500"
    assert float(rows[0]["time"]) >= 0.0

    record = json.loads(diagnostics_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["scene_id"] == 1
    assert record["im_id"] == 0
    assert record["obj_id"] == 1
    assert record["accepted"]
    assert record["candidate_type"] == "full"
    assert record["icp_fitness"] == 0.75
    np.testing.assert_allclose(record["T_camera_object"], transform)
