# Zutik geometry-first RGB-D perception

This repository contains a runnable geometry-first branch for detecting known CAD/model instances in RGB-D-derived 3-D data and estimating metric 6-DoF object poses. It is deliberately a model-based registration component: it does not train a detector, replace a learned detector, or claim to be a complete semantic scene-graph system.

The branch was added to a repository that did not contain an existing learned perception implementation, so there is no Grounding DINO/YOLO-World, SAM 2, FoundationPose, ROS, or World Semantic Model interface here to modify or compare against. If those components are added elsewhere, this package can provide a geometry-first pose hypothesis and diagnostics at their integration boundary.

## Geometry-first branch versus the learned pipeline

The learned perception path in the wider project is conceptually:

~~~text
Grounding DINO / YOLO-World -> SAM 2 -> FoundationPose
~~~

That path can discover objects from RGB/text cues and estimate poses using learned visual features. This branch instead starts with a known CAD/model catalogue and metric RGB-D geometry, then verifies model hypotheses by 3-D registration. It is useful for textureless industrial objects, calibrated depth cameras, measurable metric localization, and deployments where a known catalogue is acceptable. It is not open-world detection: without a corresponding model, it cannot identify an object. It is also sensitive to missing depth, occlusion, repetitive or symmetric geometry, poor CAD sampling, and clutter that produces a misleading candidate. The learned stack audit found no such implementation or local assets here, so no learned-vs-geometry result or overall winner is claimed. See [the learned-stack audit](experiments/learned_stack_audit_v1.md) and [the frozen comparison protocol](experiments/learned_comparison_protocol_v1.md).

## Scope and EHU relationship

| Area | What this repository demonstrates | What it does not claim |
| --- | --- | --- |
| EHU T2.1 | Known model/CAD registration against an RGB-D point cloud, metric localization, and structured acceptance/rejection diagnostics | A learned object detector, open-world discovery, or a multi-instance tracker |
| EHU T2.2 | A vertical pose-estimation/integration adapter with explicit camera/world frame composition and a local BOP result writer | Manipulation execution, tracking, or integration with an absent robotics stack |
| EHU T2.3 | GraphReadyObjectRecord, a narrow object-observation record suitable for a future graph boundary | An incremental metric-semantic map, scene graph, VLM, moved/new/missing-object reasoning, or full T2.3 |

The method receives known CAD/model samples and a scene point cloud (or a depth image plus intrinsics). It does not receive an exact object-only crop by default. Support-plane removal, statistical outlier removal, and configurable DBSCAN candidate generation operate on the complete scene. One catalogue entry produces at most one best accepted pose; same-model multi-instance detection is a known limitation.

## Architecture

The package keeps repository-specific concerns around the established Open3D algorithms:

- zutik_geometry/types.py defines units, frames, configuration, diagnostics, pose hypotheses, and the graph-ready observation adapter.
- zutik_geometry/io.py loads metric point clouds, depth arrays, CAD samples, transforms, and JSON results.
- zutik_geometry/registration.py owns scene candidate generation, Open3D registration orchestration, bidirectional verification, thresholds, ranking, duplicate suppression, and failure behavior.
- zutik_geometry/synthetic.py builds deterministic asymmetric-model and full-scene fixtures containing support/background geometry, noise, and outliers.
- zutik_geometry/bop.py is a local-only BOP adapter. It never downloads data and writes the official pose-result CSV shape.
- zutik_geometry/cli.py exposes detect, synthetic, and bop commands.

The registration backend is Open3D 0.18--0.20 through the Python API. There is one complete backend, not parallel NumPy/PCL implementations. The backend follows the established global-to-local workflow:

~~~mermaid
flowchart LR
    A[Depth image or scene point cloud] --> B[Validate finite metric points]
    B --> C[Optional support-plane and statistical-outlier filtering]
    C --> D[Support-plane removal and geometry-candidate generation]
    M[Known CAD/model point samples] --> E[Voxel downsample]
    D --> E
    E --> F[Hybrid-search normals]
    F --> G[Open3D FPFH descriptors]
    G --> H[Feature-matching RANSAC]
    H --> I[Point-to-plane ICP]
    I --> J[Thresholds and diagnostics]
    J --> K[T_camera_object]
    K --> L[T_world_object = T_world_camera @ T_camera_object]
    J --> N[Accepted or structured rejection]
~~~

## Frames, units, and rotation convention

The transform name is explicit: T_A_B maps coordinates from frame B into frame A. Therefore the registration result is:

~~~text
T_camera_object: object/model frame -> camera frame
T_world_object = T_world_camera @ T_camera_object
~~~

The camera frame used by depth projection is right-handed with x right, y down, and z forward. All package translations, distances, thresholds, point coordinates, and RMSE values are metres. A rigid transform is a 4x4 homogeneous matrix with a 3x3 rotation block and a 3-vector translation block. Rotation error in tests is the geodesic angle of R_est @ R_expected.T; it is reported in degrees. BOP CAD vertices are converted from millimetres to metres on input and pose translations are converted back to millimetres only when writing the BOP CSV.

## Open3D pipeline and diagnostics

registration.py calls the established Open3D operations directly:

1. voxel_down_sample reduces the registration clouds, with a deterministic point cap for bounded CPU work.
2. estimate_normals uses hybrid KD-tree search with configurable radius and neighbour count.
3. compute_fpfh_feature computes the standard Open3D FPFH descriptor, not the former repository-local approximation.
4. registration_ransac_based_on_feature_matching performs feature-based global registration with correspondence, edge-length, and distance checkers.
5. registration_icp refines the result with point-to-plane ICP using target normals.

Diagnostics are evidence from the relevant Open3D stage, not calibrated probabilities:

- feature_correspondences: the pre-RANSAC feature-correspondence count used as the denominator for the reported ratio. Open3D can warn and fall back from mutual filtering to its original correspondence set when mutual matches are too sparse.
- ransac_inliers: correspondences retained by the global registration result.
- inlier_ratio: ransac_inliers / feature_correspondences.
- global_fitness: Open3D RANSAC source coverage before ICP.
- fitness: Open3D ICP source coverage after refinement.
- ransac_rmse_m and rmse_m: inlier residual errors in metres for the global and refined results.
- confidence: a bounded ranking score derived from registration evidence; it is not a probability or a learned confidence calibration.

Rejected results retain accepted=false, a machine-readable failure_reason, and the available diagnostics. Invalid depth, non-finite points, empty models, malformed transforms, and unusable candidates fail explicitly instead of returning an untrusted pose.

## Native installation

The declared runtime is Python 3.10 or newer. The checked-in workflow was validated with Python 3.12 on Linux and Open3D 0.20.0. A virtual environment is recommended, especially on distributions whose system Python is externally managed:

~~~bash
cd /home/scpmaotj/Github/zutik-geometry-first-rgbd-perception
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test,bop]'
python -m pytest -q
python -m zutik_geometry --help
python -m zutik_geometry synthetic
~~~

The bop extra installs Pillow for PNG depth input. No T-LESS files, model checkpoints, or large datasets are downloaded by installation or by the synthetic command. make install, make test, and make synthetic provide equivalent local shortcuts after make is available.

## CLI examples

Register a known model against a point cloud. Supported point-cloud inputs include .npy, .npz, .ply, .xyz, and .obj; the model unit declaration is explicit:

~~~bash
python -m zutik_geometry detect \
  --scene /data/frame.npy \
  --model /data/models/part.ply \
  --model-id part-01 \
  --model-units mm \
  --candidate-mode full_and_dbscan \
  --output /tmp/part-pose.json
~~~

For a depth array, pass camera intrinsics and the multiplier that converts stored depth units to metres:

~~~bash
python -m zutik_geometry detect \
  --depth /data/frame_depth.npy \
  --fx 525 --fy 525 --cx 319.5 --cy 239.5 \
  --depth-scale-m 0.001 \
  --model /data/models/part.npy \
  --model-id part-01
~~~

Use --remove-support-plane when a dominant tabletop or similar support plane should be removed. --candidate-mode accepts full, dbscan, full_and_dbscan, or geometry_candidates; geometry_candidates is the default for the core CLI and BOP adapter. full remains an explicit diagnostic/backward-compatible mode. Add --world-camera /data/T_world_camera.npy to emit T_world_object using the composition above. The CLI exits with status 0 when at least one model is accepted and status 2 when all requested models are rejected.

For BOP T-LESS, model geometry is an explicit inference choice: `--model-representation cad` resolves only `models_cad`, while `--model-representation reconst` resolves only `models_reconst`. There is no fallback and `models_eval` is not an inference source. Both representations use the same BOP model-to-camera pose convention; representation changes the sampled geometry used by registration, not the frame or units. Observed normals are estimated at the configured physical scale and oriented towards the camera by default in the BOP path. Reliable PLY mesh normals are preserved for model representations when present.

Bidirectional mode requires both model-to-candidate and candidate-to-model overlap to meet the conservative default floor of 0.20. Open3D RANSAC fitness and ICP fitness/RMSE remain raw diagnostics; they are not probabilities, and one-sided ICP fitness is not sufficient evidence for model-conditioned detection.

## Synthetic scene demonstration

The synthetic command creates a complete scene, not an object-only input: a transformed asymmetric target, a support plane, a background wall, Gaussian point noise, and unrelated outliers. Support-plane removal plus DBSCAN supplies the candidate; the exact target point set is not passed to the detector.

~~~bash
python -m zutik_geometry synthetic
~~~

The command prints the expected transform, the recovered T_camera_object, candidate provenance, fitness/inlier diagnostics, translation error in metres, and rotation error in degrees. The checked-in test and CLI fixture use a fixed random seed; Open3D 0.20's Python RANSAC binding does not expose a per-call seed parameter, so the project guarantees bounded deterministic test outcomes rather than claiming bitwise identity for every low-overlap in-process run.

## Small local BOP T-LESS evaluation

No BOP data is included and no benchmark score is claimed. The adapter requires an already-downloaded local BOP layout and explicitly selected model IDs. It does not use scene_gt.json to turn ground-truth labels into a detector catalogue.

Install the optional reader:

~~~bash
python -m pip install -e '.[bop]'
~~~

Obtain T-LESS using the official BOP distribution instructions or an approved local mirror; do not place the archive in the image or source tree. A small evaluation needs at least models_cad/obj_XXXXXX.ply, one scene's scene_camera.json, and its PNG depth frame. The adapter accepts either a current test split or a legacy test_primesense split. For example, with a local root at /data/bop/tless:

~~~text
/data/bop/tless/
  models_cad/obj_000001.ply
  test/000001/scene_camera.json
  test/000001/depth/000000.png
~~~

For a local installation that supports filtered Hugging Face downloads, the required-file smoke setup can be obtained with:

~~~bash
mkdir -p /data/bop
python -m pip install 'huggingface_hub[cli]'
huggingface-cli download bop-benchmark/tless \
  --repo-type dataset \
  --local-dir /data/bop/tless \
  --include 'models_cad/obj_000001.ply' \
  --include 'test/000001/scene_camera.json' \
  --include 'test/000001/depth/000000.png'
~~~

If the chosen BOP distribution does not expose those paths as individually downloadable files, obtain the official archive and keep only the same model, scene metadata, and depth files in the local root. Verify that the split directory and frame naming match the archive before invoking the adapter.

Inspect the completed local directory before registration. This reads metadata and filenames only; it does not load every image:

~~~bash
python -m zutik_geometry tless-inspect --dataset /data/tless
~~~

The canonical first experiment is one explicit scene, image, and CAD model. Use the split reported by inspection, normally test_primesense for the classic T-LESS distribution or test for a BOP scenewise archive:

~~~bash
python -m zutik_geometry tless-dry-run \
  --dataset /data/tless \
  --split test_primesense \
  --scene-id 1 --image-id 0 \
  --model-id 1 \
  --model-representation cad \
  --output /output/geometry_tless-test.csv

python -m zutik_geometry tless \
  --dataset /data/tless \
  --split test_primesense \
  --scene-id 1 --image-id 0 \
  --model-id 1 \
  --model-representation cad \
  --output /output/geometry_tless-test.csv
~~~

The dry run validates the selected depth, per-image intrinsics, depth scale, model geometry, selected representation, units, and output path. The registration command processes only that selected frame and model. It writes strict official BOP columns to the CSV and writes geometric evidence, including model representation and candidate diagnostics, to the separate /output/geometry_tless-test.diagnostics.jsonl file. BOP depth scale converts stored depth units to millimetres; model vertices and BOP poses are millimetres; internal Open3D points and T_camera_object are metres; CSV translations are converted back to millimetres; CSV time is seconds. Repeat --model-id for an explicitly selected model subset, but do not omit scene/image selection for the canonical command.

The legacy bop command remains available for compatibility. It requires the same explicit selection, or a positive --max-frames when deliberately using its bounded scan mode; it no longer silently processes an entire dataset.

### Experiment progression

1. **Smoke test:** inspect, dry-run, and execute one frame plus one known model.
2. **Small subset:** repeat the explicit command for a recorded list of scenes, frames, and models, keeping one CSV and diagnostics file per controlled run.
3. **BOP evaluation:** combine official-format predictions for the selected split and run the official BOP toolkit against the dataset annotations.
4. **Analysis:** report BOP pose metrics together with acceptance/rejection reasons, candidate type, RANSAC/ICP diagnostics, units, and runtime.

The checked-in `experiments/tless_validation_manifest_scene_conditioned_v1.json` is a frozen 10-case engineering set. `scripts/run_tless_validation.py` compares historical full-scene CAD behavior with geometry-candidate CAD and reconstructed-model variants in Docker. The resulting `experiments/tless_validation_v1/` record is model-conditioned localization, not full catalogue detection or a BOP benchmark: the manifest supplies model identity, and GT is used only after inference for error reporting and diagnostic overlays.

The follow-up architecture diagnosis is recorded separately in
`experiments/tless_registration_diagnosis_v2/`; the v1 directory and manifest
are immutable. It adds post-hoc acceptance/correctness labels, an official BOP
CSV/parser audit, GT-visible candidate-ceiling measurements, CAD versus
reconstructed-model oracle registration, a deterministic GT-initialized ICP
study, and a three-variant Open3D bakeoff. The bakeoff compares the current
feature-RANSAC plus point-to-plane ICP path with FGR plus point-to-plane ICP and
feature-RANSAC plus Generalized ICP. These are diagnostic oracle-candidate
experiments, not detector benchmarks: GT is used only after candidate creation
or to construct explicitly labelled oracle clouds, and model identity is still
supplied by the frozen manifest.

The v1 result demonstrates mechanical data/units/frame flow but has 14
accepted-but-inaccurate false positives and 16 rejected inaccurate estimates
under the transparent diagnostic bands used by v2. Across the ten oracle
clouds, strict FPFH mutual matches are only about 1--2% of raw nearest-feature
matches and the raw matches are mostly not within 10 mm at the GT pose. The
current global-to-ICP stack therefore remains an experimental real-data
baseline; no production defaults or model representation were switched as a
result of this study. The next production direction should address the global
correspondence/registration problem first, while preserving a separate
candidate-isolation effort; the evidence does not justify a claim that either
stage is solved.

The frozen evidence trail and current comparison configuration are summarized
in [the geometry baseline record](experiments/geometry_baseline_summary_v1.md)
and its JSON companion. It records synthetic success, the real T-LESS false
positive, candidate and registration diagnoses, and the separate PCL/PPF spike.
The production baseline remains the CAD/Open3D path; PPF was not integrated
and no production defaults were tuned from these results.

### Learned comparison status

The repository contains references to Grounding DINO, YOLO-World, SAM 2, and
FoundationPose in the task documentation, but no wrappers, configs,
checkpoints, learned environment, or evaluation command. The native geometry
environment also lacks PyTorch and those component packages. Therefore the
one-case learned smoke test and the ten-case model-conditioned comparison were
stopped before inference under the documented stop condition. Experiment A
(model-conditioned pose) is the only comparison that could be valid with the
current manifest; Experiment B (end-to-end catalogue detection plus pose) is
not valid until both pipelines support instance selection. No learned result
has been fabricated.

Inference and scoring are separate steps. This repository produces the pose CSV; the official BOP toolkit reads that CSV together with the downloaded dataset annotations and computes the selected metrics. A typical local evaluator setup is:

~~~bash
git clone https://github.com/thodan/bop_toolkit.git /tmp/bop_toolkit
python -m pip install -e /tmp/bop_toolkit
# Configure BOP_PATH/results_path/eval_path in the toolkit for /data/bop.
python /tmp/bop_toolkit/scripts/eval_bop24_pose.py \
  --results_path=/tmp \
  --result_filenames=geometry_tless-test.csv \
  --num_workers=1
~~~

Use the evaluator script and arguments supplied by the exact BOP toolkit revision in use; some revisions require explicit dataset/result paths or a target list. Do not interpret a one-frame, one-model CSV smoke run as a benchmark score.

## Tests

The focused suite checks explicit pose errors and failure behavior, including:

- known nontrivial SE(3) recovery with translation and rotation tolerances;
- noise, outliers, partial visibility, and missing model points;
- unrelated-model rejection;
- T_camera_object direction and T_world_object composition;
- invalid or missing depth and invalid/empty model input;
- fixed-seed repeatability on the checked-in fixture;
- full-scene candidate generation with background/support geometry rather than an exact object crop;
- point-cloud/model unit conversion.

Run it with:

~~~bash
python -m pytest -q
~~~

## Docker

The image is CPU-oriented and based on python:3.12-slim-bookworm. It installs the package, Open3D, Pillow, and test dependencies, but contains no dataset, CAD archive, checkpoint, or generated result. The runtime user is non-root (zutik, UID 10001). Open3D's registration path does not require a GUI or CUDA runtime here.

Build and smoke-test it:

~~~bash
docker build -t zutik-geometry:local .
docker run --rm zutik-geometry:local --help
docker run --rm zutik-geometry:local synthetic
docker run --rm --entrypoint python zutik-geometry:local -m pytest -q
~~~

Mount external read-only data at runtime. For example:

~~~bash
docker run --rm \
  -v /data/bop:/data/bop:ro \
  zutik-geometry:local bop \
  --dataset-root /data/bop/tless \
  --split test --scene-id 1 --im-id 0 --model-id 1 --max-frames 1 \
  --output /tmp/tless.csv
~~~

Because /tmp/tless.csv is inside the ephemeral container, bind-mount a writable results directory when the output must persist:

~~~bash
docker run --rm \
  -v /data/bop:/data/bop:ro \
  -v "$PWD/results:/results" \
  zutik-geometry:local bop \
  --dataset-root /data/bop/tless \
  --split test --scene-id 1 --im-id 0 --model-id 1 --max-frames 1 \
  --output /results/tless.csv
~~~

The image's default entrypoint is the package CLI. Use --entrypoint python for arbitrary Python commands such as pytest.

For a real local T-LESS directory, mount the dataset read-only and the result directory read-write:

~~~bash
docker run --rm \
  -v /local/path/to/tless:/data/tless:ro \
  -v "$PWD/output:/output" \
  zutik-geometry:local tless-inspect \
  --dataset /data/tless

docker run --rm \
  -v /local/path/to/tless:/data/tless:ro \
  -v "$PWD/output:/output" \
  zutik-geometry:local tless \
  --dataset /data/tless \
  --split test_primesense \
  --scene-id 1 --image-id 0 --model-id 1 \
  --output /output/geometry_tless-test.csv
~~~

## Limitations and future integration boundary

The learned comparison is blocked before inference because no learned pipeline,
compatible environment, or local checkpoint is present; the exact protocol and
blockers are recorded in `experiments/learned_comparison_protocol_v1.md`.

This is a single-object-per-catalogue-entry geometry branch. FPFH/RANSAC/ICP can degrade with severe occlusion, sparse or invalid depth, repeated/symmetric geometry, bad scale/intrinsics, incomplete CAD sampling, and clutter that creates misleading clusters. The current CAD loader consumes point samples/mesh vertices; it does not perform triangle-area surface resampling. Open3D registration is CPU-oriented. Open3D 0.20 does not provide a Python per-call RANSAC seed, so exact bitwise reproducibility is not promised. There is no learned-pipeline comparison because no learned pipeline or compatible local data is present.

The narrow integration boundary is an accepted/rejected PoseHypothesis and its optional GraphReadyObjectRecord. A future system may attach this observation to an existing World Semantic Model, but this repository does not invent that interface or claim the incremental scene-graph behavior required for a full T2.3 implementation.
