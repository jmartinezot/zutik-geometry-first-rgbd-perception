# Geometry-First 3D Object Detection and Pose Estimation

> This is the original design and evaluation brief, not a description of the
> current API or a claim that every recommended component is implemented. See
> the [project README](README.md) and
> [implementation guide](docs/GEOMETRY_FIRST_PERCEPTION.md) for current behavior,
> and the [experiment records](experiments/geometry_baseline_summary_v1.md) for
> measured results.

## Purpose

This guide describes a geometry-first alternative branch for industrial RGB-D perception. It is intended for implementation and evaluation in an existing repository, not for proposing a new recognition algorithm.

The branch should use established point-cloud methods to detect known objects from a CAD/model catalogue and estimate their 6-DoF poses. It should be compared with the current vision-foundation pipeline, if available:

**Grounding DINO or YOLO-World → SAM 2 → FoundationPose**

The geometric branch is a specialized alternative for cases where the object catalogue is known, depth is reliable, and visual appearance or prompt wording is not a dependable identifier. It is not expected to outperform the vision pipeline in every setting. In particular, it requires usable 3D models and enough distinctive visible geometry; open-vocabulary recognition of arbitrary unseen objects remains a strength of the vision-language front end.

## Relationship to the EHU tasks

| Task | Defensible contribution from this branch | Boundary of the claim |
|---|---|---|
| **T2.1: geometric perception for object detection and localization from 3D information** | A reproducible pipeline that takes RGB-D/depth data and a catalogue of object models, finds object instances in a point cloud, and outputs their metric locations. | It is a CAD/model-based detector; it does not by itself provide open-vocabulary recognition of arbitrary objects. |
| **T2.2: 3D-geometry-based detection and pose estimation, integrated for tracking/manipulation** | An existing registration method estimates an object's 6-DoF pose; an adapter publishes the pose in the repository's coordinate convention and downstream interface. | Do not claim delivery to tracking/manipulation unless the relevant interface is present and exercised. |
| **T2.3: incremental metric-semantic maps, 3D scene graphs, and vision-language interpretation** | The branch can supply graph-ready object records: model/class ID, world pose, size, confidence, and geometric-fit evidence. If a World Semantic Model already exists, add an adapter to it. | Object detection and pose estimation alone do not implement an incremental scene graph or vision-language interpretation. Keep this as an integration point unless those capabilities are explicitly implemented and tested. |

The engineering contribution is the **selection, implementation, integration, and evaluation of published geometry methods**, including the data and coordinate-frame adapters. Do not claim a novel registration method.

## Recommended technical approach

### Inputs

Use the repository's existing message and configuration types where possible. The minimum inputs are:

- A depth image or scene point cloud in metric units.
- Camera intrinsics if point clouds are generated from depth images.
- An optional camera pose in the world/map frame for world-coordinate outputs.
- One or more object models, preferably CAD meshes or sampled model point clouds in metric units.
- A stable model identifier and optional class/name for each catalogue entry.

RGB can be retained for visualization or an optional semantic label, but the geometry branch must not silently depend on Grounding DINO, YOLO-World, or SAM 2 to produce its geometric result. A learned mask may be used as an optional region proposal in a separately reported hybrid experiment.

### Geometry pipeline

Implement one main backend, selected after inspecting the repository:

1. **Prefer Open3D/Python** when the project is Python-first or has no point-cloud backend. Use voxel downsampling, normal estimation, FPFH descriptors, RANSAC-based global registration, and point-to-plane ICP refinement. Open3D documents this global-registration-plus-local-refinement workflow.
2. **Prefer PCL/C++** when the project already uses PCL or its C++/ROS interfaces. Its correspondence-grouping tutorial covers 3D model-instance recognition and returns a rigid transform for each detected instance.

Do not implement both backends unless the repository already needs both. Keep algorithm parameters in configuration rather than scattering constants through the code.

Suggested processing stages:

1. Convert depth to a camera-frame cloud if needed; discard invalid or out-of-range points.
2. Downsample and remove isolated outliers. Estimate normals and local descriptors on the model and scene.
3. If appropriate for the dataset, remove a dominant support plane with RANSAC and form candidate clusters with DBSCAN/Euclidean clustering. Keep this configurable: removing every large plane can erase valid industrial surfaces.
4. Match each catalogue model to the scene or candidate clusters using FPFH/RANSAC or the selected PCL correspondence-grouping method.
5. Refine each accepted hypothesis with ICP on the original or less heavily downsampled points.
6. Expose raw verification measures such as correspondence count, overlap/fitness, inlier ratio, and registration RMSE. Reject weak hypotheses using configurable thresholds. Deduplicate repeated hypotheses in 3D, while documenting how symmetric models are handled.
7. Return the object pose and diagnostics. If a world camera pose is available, compose it with the camera-frame object pose.

Document the transform convention explicitly. For example, if `T_A_B` maps coordinates from frame B into frame A, then `T_world_object = T_world_camera @ T_camera_object`. Follow the repository's actual convention rather than assuming this example is already used there.

### Output contract

Reuse the repository's existing types if available. Otherwise define a small, versioned record containing:

- `model_id` and optional semantic label;
- `T_camera_object` and, when available, `T_world_object`;
- translation and rotation in a documented format;
- a confidence or acceptance decision, without hiding the raw geometric diagnostics;
- inlier/correspondence count, overlap or fitness, and RMSE;
- frame/timestamp and units.

If a World Semantic Model or scene-graph interface exists, add a small adapter that supplies stable object records to it. Do not create a second semantic-map or VLM stack if the repository already has one.

## Evaluation plan

### Datasets

- **Primary: BOP T-LESS.** It contains RGB-D data, 3D object models, and annotated 6-DoF poses for textureless, industry-relevant objects. Use the BOP toolkit for standard pose evaluation. Start with a small, explicitly selected subset already present locally; document how to obtain the full dataset rather than downloading large archives automatically.
- **Optional second dataset: BOP ITODD.** Use it if available and if a second industrial object set is needed.
- **Optional mapping/scene-graph evaluation: 3RScan/3DSSG.** Use this only if there is a graph or incremental-map adapter to test. These are changing indoor environments rather than industrial scenes, so they validate the mapping mechanics, not industrial generalization.

Respect each dataset's terms and license. Keep raw datasets outside the source tree unless the repository convention says otherwise.

### Baselines and fairness

If the current learned pipeline is available, run both approaches on the same RGB-D frames and the same evaluation split:

- **Geometry branch:** depth/point cloud plus CAD catalogue; no task-specific detector training.
- **Vision branch:** Grounding DINO or YOLO-World prompts, SAM 2 masks, and FoundationPose inputs as configured by that pipeline.

Record the information available to each branch. This is not an identical-input comparison: the geometry branch assumes a model catalogue, while the vision branch uses image/text evidence. Report that distinction rather than presenting the results as a universal ranking.

### Metrics

Report, at minimum:

- Object detection precision/recall or the dataset's supported detection metric.
- 6-DoF pose accuracy using the BOP evaluation toolkit where applicable; account for object symmetries according to the benchmark.
- Geometric diagnostics: accepted/rejected hypotheses, inlier ratio or overlap, and registration RMSE.
- Runtime and failure rate. State whether timing is CPU or GPU and whether preprocessing is included.
- Results split, where possible, by occlusion, visible surface amount, and object-shape ambiguity.

For a scene-graph adapter, separately report object association/duplicate rate, node and relation agreement against annotations, and detection of moved/new/missing objects. Do not infer these T2.3 results from pose accuracy alone.

### Hypothesis to test, not assume

The geometry branch may be useful when:

- the object belongs to a known CAD catalogue;
- RGB appearance, color, texture, or prompt phrasing is a weak cue;
- the depth sensor sees enough distinctive surface geometry;
- an operator needs geometric evidence for why a pose hypothesis was accepted.

It may be weaker when objects are heavily occluded, geometrically repetitive or symmetric, have poor depth returns, or are absent from the model catalogue. FoundationPose itself supports model-based and model-free pose/tracking, so the geometry branch must be compared empirically rather than described as inherently more accurate.

## Implementation workflow for Codex

1. Inspect the repository instructions, architecture, dependencies, tests, coordinate-frame conventions, and existing data loaders before editing.
2. Identify whether the codebase is Python/Open3D-oriented, C++/PCL-oriented, or already has an object/pose interface. Reuse the existing stack where practical.
3. Implement a small vertical slice: load one RGB-D frame or point cloud and one CAD model, produce a pose hypothesis plus diagnostics, and serialize or publish the result using a repository-consistent interface.
4. Add a deterministic synthetic registration test with a known rigid transform, noise, and outliers. Test transform direction and camera-to-world composition. Add input-validation tests for missing depth/model data.
5. Add a dataset adapter/evaluation script for a small local BOP subset if the data are present. If not, document exact acquisition and invocation steps; do not silently download the full dataset.
6. If the repository includes a World Semantic Model, implement and test only the adapter needed to pass object records into it. Do not claim full T2.3 coverage unless incremental graph updates and the required vision-language interpretation are also exercised.
7. Run focused tests and the relevant existing test suite. Summarize changed files, commands run, results, assumptions, and known failure cases.

## Acceptance criteria

- The geometry path runs without requiring Grounding DINO, YOLO-World, SAM 2, FoundationPose, or model training.
- The selected implementation follows the repository's language, style, dependency, and coordinate-frame conventions.
- A known rigid-transform test passes and checks the returned pose convention.
- The program returns structured object poses and geometric diagnostics, including clear failure behavior when no valid match is found.
- There is a documented command to run a small local dataset example or, when data are unavailable, a deterministic synthetic example.
- The evaluation report distinguishes demonstrated results from the proposed use-case hypothesis.
- Documentation claims T2.1/T2.2 support accurately and does not state that the geometry branch alone completes T2.3.

## References

- [Open3D: Global registration (FPFH/RANSAC and ICP refinement)](https://www.open3d.org/docs/latest/tutorial/pipelines/global_registration.html)
- [Point Cloud Library: 3D object recognition based on correspondence grouping](https://pointclouds.org/documentation/tutorials/correspondence_grouping.html)
- [BOP datasets and toolkit](https://bop.felk.cvut.cz/datasets/)
- [T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects](https://arxiv.org/abs/1701.05498)
- [FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects](https://openaccess.thecvf.com/content/CVPR2024/html/Wen_FoundationPose_Unified_6D_Pose_Estimation_and_Tracking_of_Novel_Objects_CVPR_2024_paper.html)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [YOLO-World](https://github.com/AILab-CVC/YOLO-World)
- [SAM 2](https://github.com/facebookresearch/sam2)
- [3RScan dataset](https://github.com/WaldJohannaU/3RScan) and [3DSSG scene graphs](https://3dssg.github.io/)
