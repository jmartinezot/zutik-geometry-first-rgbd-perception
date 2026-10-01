import numpy as np

from zutik_geometry.evaluation import (
    classify_acceptance,
    diagnostic_accuracy,
    error_bands,
    rotation_error_degrees,
    translation_error_m,
)


def test_acceptance_classification_keeps_internal_decision_separate():
    assert classify_acceptance(True, True) == "accepted_accurate"
    assert classify_acceptance(True, False) == "accepted_inaccurate_false_positive"
    assert classify_acceptance(False, True) == "rejected_would_be_accurate"
    assert classify_acceptance(False, False) == "rejected_inaccurate"


def test_diagnostic_accuracy_and_bands_are_explicit():
    assert diagnostic_accuracy(0.019, 4.9)
    assert not diagnostic_accuracy(0.021, 0.0)
    assert diagnostic_accuracy(0.019, None, rotation_applicable=False)
    assert error_bands(0.021, 7.0) == {
        "translation_band": "20-50mm",
        "rotation_band_non_symmetric": "5-15deg",
    }
    assert error_bands(0.001, 100.0, rotation_applicable=False)[
        "rotation_band_non_symmetric"
    ] == "not_applicable"


def test_pose_error_helpers_use_camera_object_transform_convention():
    identity = np.eye(4)
    translated = identity.copy()
    translated[:3, 3] = [0.003, 0.004, 0.0]
    assert np.isclose(translation_error_m(translated, identity), 0.005)
    angle = np.deg2rad(90.0)
    rotated = identity.copy()
    rotated[:3, :3] = [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0, 0, 1]]
    assert np.isclose(rotation_error_degrees(rotated, identity), 90.0)
