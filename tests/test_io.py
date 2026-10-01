import numpy as np

from zutik_geometry.io import load_model_catalog_entry, load_point_cloud


def _write_ply_with_normals(path, points_mm, normals):
    rows = "\n".join(
        " ".join(f"{value:.6f}" for value in np.concatenate((point, normal)))
        for point, normal in zip(points_mm, normals)
    )
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                f"element vertex {len(points_mm)}",
                "property float x",
                "property float y",
                "property float z",
                "property float nx",
                "property float ny",
                "property float nz",
                "end_header",
                rows,
                "",
            ]
        ),
        encoding="ascii",
    )


def test_point_cloud_and_explicit_model_units(tmp_path):
    source = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 20.0, 0.0]])
    path = tmp_path / "part.npy"
    np.save(path, source)

    loaded = load_point_cloud(path)
    model = load_model_catalog_entry(path, model_id="part", units="mm")

    np.testing.assert_allclose(loaded, source)
    np.testing.assert_allclose(model.points, source * 1e-3)


def test_ply_model_normals_are_preserved_and_normalized(tmp_path):
    source = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 20.0, 0.0]])
    normals = np.array([[0.0, 0.0, 2.0], [0.0, -3.0, 0.0], [4.0, 0.0, 0.0]])
    path = tmp_path / "with-normals.ply"
    _write_ply_with_normals(path, source, normals)
    model = load_model_catalog_entry(path, model_id="part", units="mm", representation="cad")
    np.testing.assert_allclose(np.linalg.norm(model.normals, axis=1), 1.0)
    np.testing.assert_allclose(model.normals[0], [0.0, 0.0, 1.0])
    assert model.representation == "cad"
