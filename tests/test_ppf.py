from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from zutik_geometry.ppf import (
    PPFBridgeConfig,
    parse_ppf_result,
    run_ppf_bridge,
    serialize_point_normals,
)


def _cloud() -> tuple[np.ndarray, np.ndarray]:
    points = np.array([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]], dtype=np.float64)
    normals = np.array([[0.0, 0.0, 2.0], [0.0, 3.0, 0.0]], dtype=np.float64)
    return points, normals


def test_ppf_serialization_is_explicitly_metric(tmp_path):
    points, normals = _cloud()
    path = tmp_path / "cloud.xyzn"
    serialize_point_normals(path, points, normals, name="test")
    lines = path.read_text().splitlines()
    assert lines[0] == "ZUTIK_PPF_XYZ_NORMALS_V1 m 2"
    assert [float(value) for value in lines[1].split()] == pytest.approx(
        [0.01, 0.02, 0.03, 0.0, 0.0, 1.0]
    )


def test_ppf_rejects_empty_or_invalid_cloud():
    with pytest.raises(ValueError, match="empty"):
        serialize_point_normals("/tmp/unused.xyzn", np.empty((0, 3)), np.empty((0, 3)), name="model")
    with pytest.raises(ValueError, match="zero-length"):
        serialize_point_normals(
            "/tmp/unused.xyzn", np.zeros((1, 3)), np.zeros((1, 3)), name="model"
        )


def test_ppf_result_parsing_preserves_transform_direction(tmp_path):
    path = tmp_path / "result.json"
    transform = np.eye(4).tolist()
    transform[0][3] = 0.123
    path.write_text(
        json.dumps(
            {
                "status": "ok",
                "translation_units": "m",
                "transform_convention": "T_target_source",
                "runtime_s": 0.5,
                "model_points": 10,
                "scene_points": 20,
                "diagnostics": {"model_ppf_features": 90},
                "hypotheses": [{"rank": 1, "votes": 7, "transform": transform}],
            }
        )
    )
    result = parse_ppf_result(path)
    assert result.hypotheses[0].T_camera_object[0, 3] == pytest.approx(0.123)
    assert result.hypotheses[0].votes == 7


def test_ppf_result_rejects_wrong_contract(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"status": "ok", "translation_units": "mm"}))
    with pytest.raises(ValueError, match="translation_units"):
        parse_ppf_result(path)


def test_ppf_bridge_surfaces_subprocess_failure():
    points, normals = _cloud()
    config = PPFBridgeConfig(executable=sys.executable, timeout_s=5.0)
    # Python sees the first bridge option as its script and exits nonzero.
    with pytest.raises(RuntimeError, match="bridge failed"):
        run_ppf_bridge(points, normals, points, normals, config)

