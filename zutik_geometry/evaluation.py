"""Pure post-hoc evaluation helpers.

These helpers deliberately do not participate in inference.  They make the
distinction between an internal geometric acceptance decision and a later
comparison with ground truth explicit and testable.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np


EvaluationClass = Literal[
    "accepted_accurate",
    "accepted_inaccurate_false_positive",
    "rejected_would_be_accurate",
    "rejected_inaccurate",
]


def rotation_error_degrees(estimated: np.ndarray, ground_truth: np.ndarray) -> float:
    """Return the relative geodesic rotation error in degrees."""

    est = np.asarray(estimated, dtype=float)
    gt = np.asarray(ground_truth, dtype=float)
    if est.shape != (4, 4) or gt.shape != (4, 4):
        raise ValueError("estimated and ground_truth must be 4x4 transforms")
    relative = est[:3, :3].T @ gt[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def translation_error_m(estimated: np.ndarray, ground_truth: np.ndarray) -> float:
    """Return Euclidean translation error in metres."""

    est = np.asarray(estimated, dtype=float)
    gt = np.asarray(ground_truth, dtype=float)
    if est.shape != (4, 4) or gt.shape != (4, 4):
        raise ValueError("estimated and ground_truth must be 4x4 transforms")
    return float(np.linalg.norm(est[:3, 3] - gt[:3, 3]))


def diagnostic_accuracy(
    translation_m: float,
    rotation_deg: float | None,
    *,
    translation_tolerance_m: float = 0.020,
    rotation_tolerance_deg: float = 5.0,
    rotation_applicable: bool = True,
) -> bool:
    """Apply transparent diagnostic tolerances, not a BOP score.

    For declared symmetric objects, callers may set ``rotation_applicable`` to
    false.  This avoids treating a naive rotation error as symmetry-aware
    correctness; official BOP pose errors remain the authoritative option.
    """

    if translation_tolerance_m <= 0.0 or rotation_tolerance_deg <= 0.0:
        raise ValueError("diagnostic tolerances must be positive")
    if not np.isfinite(translation_m) or translation_m < 0.0:
        raise ValueError("translation_m must be finite and non-negative")
    if rotation_applicable and (rotation_deg is None or not np.isfinite(rotation_deg)):
        return False
    return bool(
        translation_m <= translation_tolerance_m
        and (not rotation_applicable or float(rotation_deg) <= rotation_tolerance_deg)
    )


def classify_acceptance(accepted: bool, accurate: bool) -> EvaluationClass:
    """Classify internal acceptance against post-hoc diagnostic correctness."""

    if accepted and accurate:
        return "accepted_accurate"
    if accepted and not accurate:
        return "accepted_inaccurate_false_positive"
    if not accepted and accurate:
        return "rejected_would_be_accurate"
    return "rejected_inaccurate"


def error_bands(
    translation_m: float,
    rotation_deg: float | None,
    *,
    rotation_applicable: bool = True,
) -> dict[str, str | None]:
    """Return coarse labelled bands used in reports, without hiding values."""

    translation_mm = float(translation_m) * 1000.0
    if translation_mm <= 20.0:
        translation_band = "<=20mm"
    elif translation_mm <= 50.0:
        translation_band = "20-50mm"
    else:
        translation_band = ">50mm"
    if not rotation_applicable or rotation_deg is None:
        rotation_band: str | None = "not_applicable"
    elif rotation_deg <= 5.0:
        rotation_band = "<=5deg"
    elif rotation_deg <= 15.0:
        rotation_band = "5-15deg"
    else:
        rotation_band = ">15deg"
    return {
        "translation_band": translation_band,
        "rotation_band_non_symmetric": rotation_band,
    }
