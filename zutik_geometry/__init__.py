"""Geometry-first RGB-D object detection and 6-DoF pose estimation.

The package uses one Open3D/Python point-cloud backend. It is deliberately
independent of the learned Grounding DINO/YOLO-World, SAM 2, and FoundationPose
branch described by the project guide.
"""

from .registration import (
    GeometryFirstDetector,
    depth_to_point_cloud,
    compute_bidirectional_verification,
    register_model_to_scene,
    transform_points,
)
from .bop import (
    bop_pose_to_transform,
    inspect_dataset,
    load_ground_truth_pose,
    transform_to_bop_pose,
    validate_tless_experiment,
)
from .types import (
    CameraIntrinsics,
    DetectionConfig,
    GraphReadyObjectRecord,
    GeometricVerification,
    ModelCatalogEntry,
    PoseHypothesis,
    RegistrationDiagnostics,
)

__all__ = [
    "CameraIntrinsics",
    "bop_pose_to_transform",
    "DetectionConfig",
    "depth_to_point_cloud",
    "compute_bidirectional_verification",
    "GeometryFirstDetector",
    "GraphReadyObjectRecord",
    "GeometricVerification",
    "inspect_dataset",
    "load_ground_truth_pose",
    "ModelCatalogEntry",
    "PoseHypothesis",
    "RegistrationDiagnostics",
    "register_model_to_scene",
    "transform_points",
    "transform_to_bop_pose",
    "validate_tless_experiment",
]
