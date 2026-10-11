# Geometry-first RGB-D branch

This page describes the implementation in detail. For the project entry point
and experiment index, see the [root README](../README.md).

This repository had no source tree, dependency manifest, tests, coordinate-frame contract, learned pipeline, or World Semantic Model interface when this branch was added. The implementation therefore adds one Python/Open3D backend and does not replace or invoke Grounding DINO/YOLO-World, SAM 2, or FoundationPose. The later learned-stack audit confirms that those components remain documentation references only in this repository; no learned comparison has been run.

## What is implemented

`zutik_geometry` takes a metric camera-frame point cloud (or a depth image plus intrinsics) and a catalogue of CAD/model point samples. The scene is not assumed to be an exact object crop: configurable support-plane removal, statistical outlier filtering, and Open3D DBSCAN generate geometry candidates. `geometry_candidates` is the normal CLI/BOP mode; `full` remains an explicit diagnostic mode. One best pose is returned per catalogue model.

For each candidate it runs:

1. invalid-depth filtering and optional support-plane removal;
2. Open3D statistical outlier filtering, voxel downsampling, and bounded point-count capping;
3. Open3D hybrid-search normal estimation and `compute_fpfh_feature`;
4. Open3D feature-based RANSAC global registration;
5. Open3D point-to-plane ICP refinement;
6. bidirectional geometric verification, conservative acceptance thresholds, duplicate suppression, and raw registration diagnostics.

The registration method is an implementation of the established global-registration-plus-local-refinement workflow, not a novel estimator. Open3D was selected because the repository is Python-only, contains no PCL/ROS/C++ interfaces to reuse, and the guide explicitly prefers Open3D for a Python-first/no-backend repository. The initial environment did not have Open3D and its system Python is externally managed, but neither is a deployment prohibition: the dependency is declared in `pyproject.toml` and installs cleanly in a virtual environment. Adding the established implementation is preferable to maintaining custom feature and optimizer math.

The concrete calls mirror the [Open3D global-registration tutorial](https://www.open3d.org/docs/latest/tutorial/pipelines/global_registration.html): `voxel_down_sample`, hybrid-search normals, `compute_fpfh_feature`, feature-matching RANSAC, and point-to-plane `registration_icp`.

The BOP adapter resolves model representation explicitly: `cad` means only
`models_cad`, and `reconst` means only `models_reconst`; there is no fallback
and `models_eval` is not used for inference. PLY mesh normals are retained
when present. Observed normals are estimated at the configured metric scale
and can be oriented towards or away from the camera. Verification reports
model-to-candidate and candidate-to-model overlap, raw and trimmed nearest
neighbour RMSE, and extent ratios. Both directional overlap values are needed
for bidirectional acceptance; these are geometric coverage measures, not
probabilities.

## Frames, units, and outputs

The convention is `T_A_B` maps coordinates from frame B into frame A. A model point is in the object frame, so the primary result is:

```text
T_camera_object: object/model frame -> camera frame
T_world_object = T_world_camera @ T_camera_object
```

All package input/output translations and distances are metres. The camera frame is right-handed with x right, y down, z forward for depth projection. Rotations are 3x3 matrices inside 4x4 homogeneous transforms. BOP model vertices are explicitly converted from millimetres to metres and BOP result translations are converted back to millimetres only in the CSV adapter.

`PoseHypothesis` always contains `accepted` and `RegistrationDiagnostics`. A rejected hypothesis includes a machine-readable `failure_reason` and measures such as feature correspondence count, RANSAC inliers/RMSE, ICP inliers, fitness, and final RMSE. Malformed arrays raise `ValueError` with the input name rather than producing an untrusted pose.

`PoseHypothesis.to_graph_record()` is a narrow, one-observation graph-ready adapter carrying model ID, label, frame, pose, size, confidence, and evidence. There was no World Semantic Model in this repository, so this branch does not claim incremental graph association, moved/new/missing-object tracking, or vision-language interpretation.

## Run the vertical slice

From the repository root:

```bash
python3 -m pip install -e '.[test]'
pytest -q
python3 -m zutik_geometry synthetic
```

The deterministic synthetic example creates one asymmetric model, applies a known rigid transform, adds Gaussian depth-like noise and outliers, and prints the recovered `T_camera_object`. It does not download a model, dataset, checkpoint, or learned model.

Open3D's documented global-registration call is stochastic RANSAC. The configuration records a fixed seed and makes filtering, voxelization, candidate ordering, and test inputs deterministic; Open3D 0.20's Python binding does not expose a per-call RANSAC seed parameter, so exact bitwise equality is not promised across repeated calls or platform builds. Tests therefore require stable acceptance and explicit pose-error bounds rather than identical floating-point output.

For a local point cloud and model sample:

```bash
python3 -m zutik_geometry detect \
  --scene /data/frame.npy \
  --model /data/models/part.ply \
  --model-id part-01 \
  --model-units mm \
  --output /tmp/part-pose.json
```

For a depth array in `.npy` format, provide intrinsics and the explicit stored-unit conversion:

```bash
python3 -m zutik_geometry detect \
  --depth /data/frame_depth.npy \
  --fx 525 --fy 525 --cx 319.5 --cy 239.5 \
  --depth-scale-m 0.001 \
  --model /data/models/part.npy --model-id part-01
```

Add `--world-camera /data/T_world_camera.npy` to receive `T_world_object`; the CLI defaults to frame names `camera` and `world`.

## Small local BOP T-LESS evaluation

No benchmark result is claimed. The canonical adapter commands are `python3 -m zutik_geometry tless-inspect`, `tless-dry-run`, and `tless`; they read an already-downloaded BOP scene, use the explicitly selected model IDs, and write a BOP pose-result CSV. Pillow is optional and is needed only for PNG depth input:

```bash
python3 -m pip install -e '.[bop]'
```

The official BOP dataset page provides the current Hugging Face download form.
For a local copy, use the current
[`hf` CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) outside the
source tree (the archive is large and is intentionally not downloaded by this
project):

```bash
mkdir -p /data/bop
cd /data/bop
python3 -m pip install --upgrade huggingface_hub
hf download bop-benchmark/tless \
  --local-dir /data/bop/tless --repo-type dataset
```

The BOP toolkit describes T-LESS as 30 textureless industrial objects and the BOP directory is normally `/data/bop/tless`; use the split directory actually present in the downloaded archive (`test` for a BOP scenewise archive or `test_primesense` for the classic Primesense layout). Inspect a local directory before loading image data:

```bash
cd /home/scpmaotj/Github/zutik-geometry-first-rgbd-perception
python3 -m zutik_geometry tless-inspect --dataset /data/bop/tless
```

Then validate and run one explicit scene/frame/model. The adapter deliberately does not infer object labels from `scene_gt.json` and does not silently process the full dataset:

```bash
python3 -m zutik_geometry tless-dry-run \
  --dataset /data/bop/tless \
  --split test_primesense \
  --scene-id 1 --image-id 0 --model-id 1 \
  --model-representation cad \
  --output /tmp/tless-obj01.csv
python3 -m zutik_geometry tless \
  --dataset /data/bop/tless \
  --split test_primesense \
  --scene-id 1 --image-id 0 --model-id 1 \
  --model-representation cad \
  --output /tmp/tless-obj01.csv
```

The BOP depth scale is the multiplier from stored depth units to millimetres, then is converted to metres before projection. BOP model vertices and pose translations are millimetres; internal Open3D geometry and T_camera_object are metres; BOP CSV translations are millimetres; CSV runtime is seconds. Registration evidence is written separately to `/tmp/tless-obj01.diagnostics.jsonl`. The BOP convention is model-to-camera, matching T_camera_object.

For standard scoring, install the official BOP toolkit separately, configure `BOP_PATH`/result paths as required by that toolkit, and run its evaluation against the generated CSV, for example:

```bash
git clone https://github.com/thodan/bop_toolkit.git /tmp/bop_toolkit
python3 -m pip install -e /tmp/bop_toolkit
cd /tmp/bop_toolkit
python3 scripts/eval_bop24_pose.py \
  --result_filenames=/tmp/tless-obj01.csv --num_workers 1
```

The exact metric result depends on the selected split, target list, symmetry handling, sensor, and whether the requested object list is a complete detector catalogue. Record those inputs with the result. For a fair learned-pipeline comparison, run both branches on the same RGB-D frames and split, but report that geometry receives a known CAD catalogue while the learned branch receives RGB/text/masks/FoundationPose inputs; these are unequal information assumptions.

## Frozen real-data diagnosis

The immutable ten-case baseline is in
`experiments/tless_validation_v1/`, with its manifest at
`experiments/tless_validation_manifest_scene_conditioned_v1.json`. The
follow-up record is in `experiments/tless_registration_diagnosis_v2/`.
`scripts/run_tless_registration_diagnosis_v2.py` reproduces the diagnostic
record against the same `test_primesense` cases without changing production
configuration. It labels internal `accepted` separately from post-inference
correctness, audits BOP CSV parsing and official pose-error functions, measures
whether geometry-only candidates contain the GT-visible object, and compares
three established Open3D variants on GT-derived oracle clouds.

The diagnostic evidence is negative but actionable. The ten geometry candidate
sets contain a candidate meeting the explicit 5 mm recall/purity diagnostic
rule in only 1/10 cases; most high-recall clusters also contain substantial
support/background geometry. On oracle clouds, strict mutual FPFH matches are
typically only 1--2% of raw matches, and only a small fraction of raw matches
are within 10 mm of their GT-corresponding observed point. Generalized ICP
improves the descriptive median translation result relative to point-to-plane
ICP, especially for reconstructed models, but neither global initializer nor
refinement reaches reliable pose accuracy across the frozen set. Exact-GT ICP
also drifts on both representations, so a high one-sided ICP fitness is not a
correctness guarantee.

The official toolkit parser accepted all three ten-row diagnostic CSVs and its
official MSSD error calculation completed using a temporary symlink-only
standard-layout adapter. No aggregate BOP score is reported: the files are
model-conditioned selected cases, not complete detector outputs. The v2 record
also uses the toolkit's official `mssd`, `mspd`, `re`, and `te` functions for
post-hoc rows where available; continuous symmetries are not discretized by
the local report and remain an official-evaluator concern.

Conclusion: synthetic registration and the data/units/frame adapter work, but
the real-data branch is not yet a robust T-LESS detector. The evidence supports
changing the global correspondence/registration strategy first in the next
production iteration, not silently changing thresholds or switching CAD versus
reconstructed models from ten cases. Candidate generation and hypothesis
verification remain separate unresolved ceilings. This is an incremental
geometry-first contribution toward EHU T2.1/T2.2; it is not a full T2.1
detector benchmark or a T2.3 scene-graph/VLM implementation.

## Tests and scope

The focused tests cover:

- deterministic noisy/outlier registration and the returned transform direction;
- camera-to-world composition;
- metric depth projection;
- rejection diagnostics and malformed-input failures;
- explicit point-cloud/model unit conversion.

This demonstrates a T2.1 sub-capability—geometry-only candidate generation and
known-model localization—not complete catalogue detection, because model ID is
currently supplied by the caller. It demonstrates a T2.2 pose-estimation/
integration adapter at the object-pose record level, not tracking or
manipulation delivery. It provides only the T2.3 graph-ready observation
adapter, not an incremental metric-semantic map, a scene graph, or a VLM.

The frozen evidence trail is summarized in
`experiments/geometry_baseline_summary_v1.md` (and its JSON companion). The
current comparison baseline is the production CAD/Open3D configuration:
geometry candidates, support-plane removal, standard Open3D FPFH/RANSAC,
point-to-plane ICP, observed normals oriented toward the camera, and
bidirectional verification. The experimental PCL/Drost PPF result remains in
`experiments/tless_ppf_spike_v1/`; it is not imported by the production CLI and
does not replace Open3D.

## Learned-stack comparison status

The repository was audited for Grounding DINO, YOLO-World, SAM 2, FoundationPose,
wrappers, checkpoints, preprocessing, pose serialization, and evaluation
commands. Only documentation references were found. The declared environment
contains NumPy/Open3D (plus test/BOP helpers), but no PyTorch or learned
component packages, and no checkpoint/configuration is present. The one-case
learned smoke test and frozen ten-case comparison therefore stop before
inference; no learned numbers or overall winner are claimed.

The pre-registered fairness rules, exact manifest hash, Experiment A/B scope,
metrics, runtime accounting, and failure taxonomy are in
`experiments/learned_comparison_protocol_v1.md`. The concrete audit is in
`experiments/learned_stack_audit_v1.md`. Experiment A is a future
model-conditioned pose comparison; Experiment B remains invalid until both
methods support catalogue-wide detection without GT instance hints.

Known limitations are deliberate: Open3D registration remains CPU-oriented in this path; FPFH/RANSAC/ICP degrade with severe occlusion, repetitive/symmetric geometry, sparse or invalid depth, and catalogue omissions; one catalogue entry produces at most one accepted instance; CAD meshes are consumed as their sampled vertices rather than surface-resampled triangles; Open3D 0.20 does not guarantee bitwise per-call RANSAC reproducibility from Python; and the learned baseline is blocked before inference by missing implementation, environment, and checkpoints. Symmetry equivalence is left to the BOP evaluator and is not hidden in the acceptance score.
