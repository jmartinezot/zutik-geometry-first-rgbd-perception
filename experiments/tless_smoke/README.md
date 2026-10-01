# Real T-LESS one-case geometry-first experiment

This is one smoke experiment only. The dataset was not copied into the repository.

## Selection

- Dataset: `/home/datasets/bop/tless`
- Split: `test_primesense`
- Scene: `3`
- Image: `233`
- CAD model: `18`, `/home/datasets/bop/tless/models_cad/obj_000018.ply`
- Selection basis: `scene_gt_info.json` reported 99.44% visible fraction, 16,460 visible pixels, and a 139x145 pixel box for this instance. Model 18 has no `symmetries_discrete` or `symmetries_continuous` entry in `models_info.json`.
- Ground truth was used only for selection and post-inference evaluation. No GT crop, mask, pose, YOLO output, `train_pbr` data, or YOLO data was passed to registration.

## Preflight

- RGB: 720x540 PNG
- Depth: 720x540 uint16 PNG, 384,259 positive pixels
- BOP depth scale: 0.1 mm/stored unit = 0.0001 m/stored unit
- Positive metric depth range: 0.6311–2.6776 m; median 0.7627 m
- Intrinsics: fx=1075.65091572, fy=1073.90347929, cx=375.06888344, cy=300.72159802
- CAD: 18,095 points, source units mm, internal units m; bounds approximately 98.7 x 98.7 x 63.8 mm
- Dry-run: passed

## Native inference

The existing defaults were used unchanged: voxel size 0.005 m, RANSAC threshold 0.015 m, ICP threshold 0.02 m, seed 7, candidate mode `full`, support-plane removal disabled, statistical outlier filtering enabled.

- Scene points: 384,259 before filtering; 367,497 after statistical filtering
- Support plane removed: no
- DBSCAN candidates: 0 (the configured candidate mode was `full`)
- Selected candidate: full filtered scene, 367,497 points
- RANSAC: 351 inliers / 947 feature correspondences; inlier ratio 0.370644; RMSE 0.009785 m
- ICP: 2,148 inliers; fitness 0.8592; RMSE 0.008796 m
- Decision: accepted, confidence 0.596342, no rejection reason
- Detector runtime: 358.8 ms; end-to-end native wall time: 10.54 s
- Separate preprocessing timing is not exposed by the current CLI.

The accepted result is nevertheless inaccurate: the estimator registered model 18 onto a different large object in the scene. This is visible in the overlay and is classified as a full-scene candidate-generation/global-registration false positive, not as a threshold-tuning opportunity.

### Estimated pose

```text
T_camera_object_estimated =
[[ 0.25825065, -0.67542076, -0.69073396,  0.04560476],
 [-0.96604819, -0.18615936, -0.17915244, -0.02686901],
 [-0.00758331,  0.71354852, -0.70056477,  0.70375672],
 [ 0.0       ,  0.0       ,  0.0       ,  1.0       ]]
```

### Ground-truth pose, read after inference

```text
T_camera_object_ground_truth =
[[-0.86910545,  0.49441825, -0.01437454, -0.11749747],
 [-0.34426527, -0.58378321,  0.73530805, -0.02977498],
 [ 0.35515853,  0.64400958,  0.67757994,  0.77318804],
 [ 0.0       ,  0.0       ,  0.0       ,  1.0       ]]
```

- Translation error: 0.177289 m / 177.289 mm
- Naive geodesic rotation error: 128.932 degrees
- No official symmetry entry is present for model 18, so the naive error is not being symmetry-adjusted here. Correct benchmark handling remains deferred to the official BOP evaluator.

## Artifacts

- [dataset inspection](dataset_inspection.json)
- [dry-run report](dry_run.json)
- [configuration](config.json)
- [post-hoc metrics](posthoc_metrics.json)
- [official BOP CSV](tless_scene000003_im000233_obj000018.csv)
- [diagnostics JSONL](tless_scene000003_im000233_obj000018.diagnostics.jsonl)
- [estimated vs GT overlay](pose_overlay_estimated_vs_gt.png)
- [estimated-only overlay](pose_overlay_estimated.png)
- [ground-truth-only overlay](pose_overlay_ground_truth.png)
- [native/Docker comparison](docker_comparison.json)
- [Docker output directory](docker/)

## Docker

The same dataset, split, scene, image, model, and configuration were run with the dataset mounted read-only and `experiments/tless_smoke/docker` mounted read-write. Docker also accepted the result and produced the same transform, fitness, inliers, and RMSE to numerical precision; detector runtime was 360.8 ms.

## Conclusion

A: the real T-LESS RGB-D/CAD path is mechanically working: layout discovery, `test_primesense` frame loading, 0.1-mm depth scaling, intrinsics, mm-to-m conversion, official CSV writing, diagnostics, and Docker mounting all work.

B: this one example did not produce a geometrically accurate pose. The most likely cause is the current full-scene candidate/global-registration false positive: the target was not isolated, no support plane was removed, and no DBSCAN candidate was selected. No parameter tuning was performed.
