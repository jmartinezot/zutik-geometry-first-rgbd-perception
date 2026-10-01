import math

import numpy as np
import pytest

from zutik_geometry.io import load_depth_array
from zutik_geometry.registration import (
    _estimate_normals,
    _to_point_cloud,
    compute_bidirectional_verification,
    depth_to_point_cloud,
    make_transform,
    register_model_to_scene,
    transform_points,
)
from zutik_geometry.synthetic import (
    make_asymmetric_model,
    make_synthetic_case,
    make_synthetic_scene_case,
    rotation_error_degrees,
)
from zutik_geometry.types import CameraIntrinsics, DetectionConfig, ModelCatalogEntry


def rotation_z(angle: float) -> np.ndarray:
    return np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def config_for_object_crop(**overrides) -> DetectionConfig:
    values = dict(
        voxel_size_m=0.006,
        ransac_inlier_threshold_m=0.012,
        icp_max_correspondence_m=0.018,
        max_rmse_m=0.012,
        random_seed=19,
        candidate_mode="full",
        remove_statistical_outliers=False,
        # The synthetic fixture has no camera-facing surface convention.  The
        # production RGB-D adapter explicitly uses towards_camera; keep this
        # fixture's normal assumption explicit rather than relying on a
        # production default.
        observed_normal_orientation="none",
    )
    values.update(overrides)
    return DetectionConfig(**values)


def assert_pose_error(result, expected, *, translation_m=0.006, rotation_deg=3.0):
    assert result.accepted, result.diagnostics
    assert result.T_camera_object is not None
    assert np.linalg.norm(result.T_camera_object[:3, 3] - expected[:3, 3]) <= translation_m
    assert rotation_error_degrees(result.T_camera_object, expected) <= rotation_deg


def test_open3d_registration_recovers_nontrivial_se3_with_noise_and_outliers():
    model, scene, expected_camera_object = make_synthetic_case()
    result = register_model_to_scene(model, scene, config=config_for_object_crop())

    assert_pose_error(result, expected_camera_object, translation_m=0.004, rotation_deg=2.0)
    assert result.diagnostics.method == "open3d-fpfh-ransac-point-to-plane-icp"
    assert result.diagnostics.ransac_inliers >= 6
    assert result.diagnostics.icp_inliers >= 12
    assert result.diagnostics.global_fitness >= 0.5
    assert result.diagnostics.fitness >= 0.5
    assert result.diagnostics.rmse_m is not None and result.diagnostics.rmse_m <= 0.012


def test_transform_direction_and_camera_to_world_composition():
    model, scene, expected_camera_object = make_synthetic_case()
    world_camera = np.eye(4)
    world_camera[:3, :3] = rotation_z(0.41)
    world_camera[:3, 3] = [1.2, -0.4, 0.25]
    result = register_model_to_scene(
        model,
        scene,
        config=config_for_object_crop(),
        T_world_camera=world_camera,
        world_frame="world",
    )

    assert_pose_error(result, expected_camera_object, translation_m=0.004, rotation_deg=2.0)
    assert result.T_world_object is not None
    np.testing.assert_allclose(result.T_world_object, world_camera @ result.T_camera_object, atol=1e-9)
    transformed_expected = transform_points(expected_camera_object, model.points[:20])
    transformed_estimated = transform_points(result.T_camera_object, model.points[:20])
    np.testing.assert_allclose(transformed_estimated, transformed_expected, atol=0.01)


def test_scene_level_candidate_generation_finds_target_in_background_scene():
    model, scene, expected_camera_object = make_synthetic_scene_case()
    config = config_for_object_crop(
        candidate_mode="dbscan",
        remove_support_plane=True,
        cluster_eps_m=0.045,
        cluster_min_points=20,
    )
    result = register_model_to_scene(model, scene, config=config)

    assert_pose_error(result, expected_camera_object, translation_m=0.006, rotation_deg=3.0)
    assert result.diagnostics.candidate_source == "dbscan"
    assert result.diagnostics.scene_points == len(scene)
    assert result.diagnostics.candidate_points < result.diagnostics.scene_points
    assert result.diagnostics.fitness >= 0.20
    assert result.diagnostics.candidate_generation["support_plane"]["removed"]
    assert result.diagnostics.candidate_generation["clusters_before_filtering"] >= 1


def test_geometry_candidate_mode_exposes_plane_and_cluster_diagnostics():
    model, scene, expected_camera_object = make_synthetic_scene_case()
    result = register_model_to_scene(
        model,
        scene,
        config=config_for_object_crop(
            candidate_mode="geometry_candidates",
            remove_support_plane=True,
            cluster_eps_m=0.045,
            cluster_min_points=20,
        ),
    )
    assert_pose_error(result, expected_camera_object, translation_m=0.006, rotation_deg=3.0)
    generation = result.diagnostics.candidate_generation
    assert generation["mode"] == "geometry_candidates"
    assert generation["candidate_count"] >= 1
    assert generation["support_plane"]["input_points"] == len(scene)
    assert generation["after_support_plane_points"] < len(scene)
    assert all("rejection_reason" in item for item in generation["clusters"])


def test_bidirectional_verification_requires_both_geometries_to_agree():
    rng = np.random.default_rng(123)
    model = rng.uniform([-0.1, -0.06, -0.04], [0.1, 0.06, 0.08], size=(500, 3))
    expected = make_transform(rotation_z(0.53), np.array([0.25, -0.08, 0.72]))
    aligned = transform_points(expected, model)
    correct = compute_bidirectional_verification(model, aligned, expected, distance_threshold_m=0.005)
    assert correct.bidirectional_overlap == pytest.approx(1.0)
    assert np.max(correct.extent_ratio) == pytest.approx(1.0, abs=1e-6)

    unrelated = rng.uniform([-0.9, -0.8, 0.3], [0.9, 0.8, 1.5], size=(2000, 3))
    large_scene = np.vstack((aligned, unrelated))
    large = compute_bidirectional_verification(model, large_scene, expected, distance_threshold_m=0.005)
    assert large.model_to_candidate_overlap >= 0.99
    assert large.candidate_to_model_overlap < 0.35
    assert large.bidirectional_overlap < 0.35

    partial = compute_bidirectional_verification(model[::3], aligned, expected, distance_threshold_m=0.005)
    assert partial.bidirectional_overlap >= 0.30
    wrong = compute_bidirectional_verification(model, aligned, make_transform(np.eye(3), [0.7, 0.4, 0.2]), distance_threshold_m=0.005)
    assert wrong.bidirectional_overlap < 0.10


def test_observed_normal_orientation_is_unit_length_and_viewpoint_consistent():
    grid = np.array([[x, y, 1.0] for x in np.linspace(-0.1, 0.1, 8) for y in np.linspace(-0.1, 0.1, 8)])
    cloud = _to_point_cloud(grid)
    config = DetectionConfig(voxel_size_m=0.01, normal_neighbors=30)
    _estimate_normals(cloud, config, 2.0, orientation="towards_camera")
    normals = np.asarray(cloud.normals)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-6)
    # On a plane in front of the camera, a normal pointing towards the camera
    # has a negative camera-frame z component.
    assert float(np.mean(normals[:, 2])) < -0.8
    away_cloud = _to_point_cloud(grid)
    _estimate_normals(away_cloud, config, 2.0, orientation="away_from_camera")
    np.testing.assert_allclose(np.asarray(away_cloud.normals), -normals, atol=1e-6)


def test_partial_visibility_is_registered_without_model_crop_input():
    model, _, expected_camera_object = make_synthetic_case()
    rng = np.random.default_rng(77)
    visible_mask = (np.arange(len(model.points)) % 3) != 0
    partial_target = transform_points(expected_camera_object, model.points[visible_mask])
    partial_target += rng.normal(0.0, 0.0008, partial_target.shape)
    outliers = rng.uniform([-0.4, -0.35, 0.3], [0.55, 0.35, 1.2], size=(100, 3))

    result = register_model_to_scene(
        model,
        np.vstack((partial_target, outliers)),
        config=config_for_object_crop(min_fitness=0.15, min_icp_inliers=10, max_rmse_m=0.015),
    )

    assert_pose_error(result, expected_camera_object, translation_m=0.008, rotation_deg=4.0)
    assert result.diagnostics.fitness >= 0.15


def test_unrelated_model_is_rejected_with_diagnostics():
    _, scene, _ = make_synthetic_case()
    wrong_model = ModelCatalogEntry("unrelated", make_asymmetric_model(seed=99))
    result = register_model_to_scene(
        wrong_model,
        scene,
        config=config_for_object_crop(min_fitness=0.35, max_rmse_m=0.008, min_icp_inliers=20),
    )

    assert not result.accepted
    assert result.diagnostics.failure_reason is not None
    assert result.diagnostics.method == "open3d-fpfh-ransac-point-to-plane-icp"


def test_fixed_seed_is_reproducible_for_high_overlap_scene():
    model, scene, expected_camera_object = make_synthetic_case()
    config = config_for_object_crop(random_seed=1234)
    first = register_model_to_scene(model, scene, config=config)
    second = register_model_to_scene(model, scene, config=config)

    assert_pose_error(first, expected_camera_object, translation_m=0.004, rotation_deg=2.0)
    assert_pose_error(second, expected_camera_object, translation_m=0.004, rotation_deg=2.0)
    assert first.accepted == second.accepted
    assert first.diagnostics.candidate_source == second.diagnostics.candidate_source == "full"
    np.testing.assert_allclose(first.T_camera_object, second.T_camera_object, atol=1e-8)


def test_invalid_and_missing_depth_are_rejected_before_registration(tmp_path):
    model, _, _ = make_synthetic_case()
    intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, cx=0.0, cy=0.0)
    empty_scene = depth_to_point_cloud(np.zeros((3, 3), dtype=np.uint16), intrinsics, depth_scale_m=0.001)
    with pytest.raises(ValueError, match="scene_points"):
        register_model_to_scene(model, empty_scene)
    with pytest.raises(FileNotFoundError):
        load_depth_array(tmp_path / "missing-depth.npy")
    with pytest.raises(ValueError, match="2-D"):
        depth_to_point_cloud(np.ones((2, 2, 1)), intrinsics)


def test_invalid_or_empty_model_is_rejected_at_contract_boundary():
    with pytest.raises(ValueError, match="N >= 3"):
        ModelCatalogEntry("empty", np.empty((0, 3)))
    with pytest.raises(ValueError, match="finite"):
        ModelCatalogEntry("nan", np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [np.nan, 1.0, 0.0]]))
