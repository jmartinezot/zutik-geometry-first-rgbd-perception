# T-LESS diagnostic experiment: scene 3, image 233, model 18

This is a diagnostic-only follow-up to the preserved baseline in `experiments/tless_smoke`. Production thresholds and parameters were not changed. No additional T-LESS frames were run, and no YOLO, SAM, masks, or GT pose was passed to production registration.

## Table of Contents

- [Baseline](#baseline)
- [Why baseline reported zero DBSCAN candidates](#why-baseline-reported-zero-dbscan-candidates)
- [GT-only oracle cloud and registration](#gt-only-oracle-cloud-and-registration)
- [Support plane](#support-plane)
- [Verification metrics](#verification-metrics)
- [Visual artifacts](#visual-artifacts)
- [Failure decomposition](#failure-decomposition)

Project context and the experiment index are in the [root README](../../README.md#small-local-bop-t-less-evaluation).
The preserved input run is the [one-case smoke test](../tless_smoke/README.md).

## Baseline

The original BOP adapter used `candidate_mode=full`, voxel size `0.005 m`, statistical outlier filtering (`nb_neighbors=20`, `std_ratio=2.0`), RANSAC threshold `0.015 m`, and ICP threshold `0.02 m`. It processed `384,259` scene points, retained `367,497`, and used one full-scene candidate. The accepted false positive had:

- RANSAC: `351/947` inliers, ratio `0.370644`, RMSE `0.009785 m`;
- ICP: `2,148` inliers, fitness `0.8592`, RMSE `0.008796 m`;
- error: `177.289 mm`, `128.932°` naive rotation error.

The frozen baseline is [baseline_snapshot.json](baseline_snapshot.json); the original CSV and diagnostics remain in the smoke-test directory.

## Why baseline reported zero DBSCAN candidates

It did not run DBSCAN. The preserved BOP run explicitly used `--candidate-mode=full`; the current adapter default is `geometry_candidates`. `_scene_candidates` only calls `cluster_dbscan` for `dbscan`, `full_and_dbscan`, or `geometry_candidates`. Therefore “DBSCAN candidates: 0” means “DBSCAN was not executed,” not “all points were noise” and not “clusters were rejected.”

The configured diagnostic values were `eps=0.04 m`, `min_points=20`, statistical outlier filtering before clustering, no pre-clustering voxel downsampling, no support-plane removal, and only a minimum cluster check of 3 points in the candidate code. There is no maximum cluster-size filter; the 2,500-point cap applies later during registration.

With DBSCAN explicitly enabled for diagnosis, the default settings found 7 clusters and 2 noise points. The target was merged into a 347,439-point cluster: target recall `0.9987`, target purity `0.0473`. The full result is [dbscan_default_analysis.json](dbscan_default_analysis.json).

The small fixed sweep was:

| eps (m) | min points | clusters | noise fraction | target cluster size | target recall | target purity |
|---:|---:|---:|---:|---:|---:|---:|
| 0.02 | 10 | 7 | 0.000005 | 347,439 | 0.9987 | 0.0473 |
| 0.02 | 20 | 7 | 0.000005 | 347,439 | 0.9987 | 0.0473 |
| 0.04 | 10 | 7 | 0.000005 | 347,439 | 0.9987 | 0.0473 |
| 0.04 | 20 | 7 | 0.000005 | 347,439 | 0.9987 | 0.0473 |
| 0.06 | 10 | 6 | 0.000000 | 348,651 | 0.9987 | 0.0472 |
| 0.06 | 20 | 6 | 0.000000 | 348,651 | 0.9987 | 0.0472 |

Results: [dbscan_sensitivity.csv](dbscan_sensitivity.csv) and [dbscan_sensitivity.json](dbscan_sensitivity.json). The target does not naturally form its own current-DBSCAN cluster; it is merged with support/background geometry.

## GT-only oracle cloud and registration

The visible target cloud was created from `mask_visib/000233_000004.png` and depth only. It contains `16,456` points, with dimensions `0.1147 × 0.1182 × 0.1630 m`, centroid `[-0.11841, -0.03364, 0.77353] m`, and depth range `0.7401–0.9031 m`. It is strictly diagnostic: [oracle_target_cloud.ply](oracle_target_cloud.ply).

The unchanged registration pipeline was run on this cloud without GT initialization. It rejected the result because `icp_inliers_below_threshold`:

- RANSAC: `729/947` inliers, ratio `0.769799`, RMSE `0.007708 m`;
- ICP: `0` inliers, fitness `0.0`, no RMSE;
- error of the rejected pose: `266.302 mm`, `155.152°`.

This means ideal scene isolation did not make global registration succeed. However, the GT-transformed CAD has strong direct geometric agreement with the oracle cloud at 20 mm: model-to-cloud coverage `0.8874`, cloud-to-model coverage `0.9937`. Thus the direct CAD/depth scale and frame path are plausible; the unchanged global FPFH/RANSAC stage selected a wrong transform. See [oracle_registration.json](oracle_registration.json) and [gt_cad_vs_oracle_cloud.json](gt_cad_vs_oracle_cloud.json).

## Support plane

Support-plane removal was disabled in production. Using the existing diagnostic parameters (`0.008 m` threshold, 1,000 iterations, minimum ratio `0.45`), the dominant plane contained `265,138/384,259` points (`0.6900`) with normalized equation:

```text
-0.006503 x + 0.734065 y + 0.679048 z - 0.536982 = 0
```

Only `170/16,456` target points (`1.03%`) were plane inliers; the target’s median distance to the plane was `33.6 mm`. If enabled, the current plane test would detect and remove this plane without removing most of the target, but this experiment does not change that production default. Details: [support_plane_analysis.json](support_plane_analysis.json).

## Verification metrics

At the existing ICP correspondence threshold of `0.02 m`, using the same 2,500-point cap as current ICP:

| Transform / candidate | Model → candidate | Candidate → model | Minimum bidirectional overlap | Mean symmetric NN distance |
|---|---:|---:|---:|---:|
| Baseline false pose / full scene | 0.8592 | 0.0652 | 0.0652 | 0.1289 m |
| Baseline false pose / oracle cloud | 0.0000 | 0.0000 | 0.0000 | 0.1305 m |
| Oracle RANSAC pose / oracle cloud | 0.0000 | 0.0000 | 0.0000 | 0.2165 m |
| GT pose reference / oracle cloud | 0.8704 | 0.9928 | 0.8704 | 0.0088 m |

Open3D fitness in this usage is source/model coverage: many model points found a nearby scene point. It does not require the much larger scene candidate to be explained by the model. The false full-scene candidate was 8–17× larger than the transformed model along its axes, which explains how it could achieve `0.8592` one-sided fitness while having only `0.0652` reverse coverage. Bidirectional verification clearly separates the false positive from the GT reference offline, but it has not been made a production acceptance rule.

## Visual artifacts

- [full RGB](full_scene_rgb.png)
- [full depth](full_scene_depth.png)
- [DBSCAN clusters, GT target annotated](dbscan_clusters_gt_annotated.png)
- [DBSCAN point cloud, GT target annotated](dbscan_clusters_gt_annotated.ply)
- [dominant plane, GT target annotated](dominant_plane_gt_overlay.png)
- [dominant plane point cloud](support_plane_diagnostic.ply)
- [failed full-scene estimate](failed_full_scene_estimate.png)
- [oracle-crop estimate vs GT, diagnostic / GT only](oracle_crop_registration_vs_gt.png)

## Failure decomposition

- **DATA/UNITS/FRAMES — working.** Dry-run passed; GT-transformed CAD agrees directly with visible depth at the 20 mm diagnostic threshold.
- **CANDIDATE GENERATION — likely problematic.** Production used the entire filtered scene; DBSCAN was not executed. When explicitly diagnosed, it merged the target with the dominant scene cluster. Plane removal was disabled although a dominant plane was measurable.
- **GLOBAL REGISTRATION — likely problematic.** Oracle isolation still produced a wrong FPFH/RANSAC transform despite strong GT geometric overlap.
- **ICP REFINEMENT — inconclusive/downstream.** ICP refined the wrong global hypothesis in the baseline and rejected the wrong oracle hypothesis. A correct-init ICP test was not run because GT pose initialization is prohibited in this iteration.
- **HYPOTHESIS VERIFICATION — likely problematic.** One-sided model coverage/ICP fitness accepted the false positive; offline bidirectional overlap rejects it clearly.
