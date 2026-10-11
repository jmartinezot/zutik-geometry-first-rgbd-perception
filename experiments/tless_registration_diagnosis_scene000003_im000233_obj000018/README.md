# T-LESS registration diagnosis: scene 3 / image 233 / model 18

This directory is a registration-only diagnosis of the preserved T-LESS case:

```text
dataset: /home/datasets/bop/tless
split: test_primesense
scene: 3
image: 233
model: 18
```

The earlier smoke-test and candidate-generation diagnosis directories are
unchanged. No production registration defaults, candidate-generation defaults,
learned methods, or T-LESS dataset files were changed.

## Table of Contents

- [Exact production configuration reproduced](#exact-production-configuration-reproduced)
- [Main evidence](#main-evidence)
- [Verification evidence](#verification-evidence)
- [Visual artifacts](#visual-artifacts)
- [Updated failure decomposition](#updated-failure-decomposition)
- [Reproduction and test status](#reproduction-and-test-status)

Project context and the experiment index are in the [root README](../../README.md#small-local-bop-t-less-evaluation).
Earlier records: the [smoke test](../tless_smoke/README.md) and
[candidate-generation diagnosis](../tless_diagnosis_scene000003_im000233_obj000018/README.md).

All registration experiments here use Open3D 0.20.0 in the existing
`zutik-geometry:local` image. The standard production model is
`models_cad/obj_000018.ply`; the `models_reconst` and `models_eval` runs are
diagnostic comparisons only. The oracle cloud is made from the visible GT mask
for diagnosis and is never used as a production detector input or as a pose
initialization.

## Exact production configuration reproduced

See [registration_config.json](registration_config.json) and
[preprocessing_and_normals_audit.json](preprocessing_and_normals_audit.json).
The important values are:

- voxel `0.005 m`; normal radius `0.010 m`, max neighbors `30`;
- FPFH radius `0.025 m`, max neighbors `100`;
- feature-RANSAC distance `0.015 m`, 20,000 iterations, confidence `0.999`;
- point-to-plane ICP distance `0.020 m`, 35 iterations;
- statistical outlier removal enabled with 20 neighbors / std ratio 2.0;
- support-plane removal disabled in the preserved production `full` candidate run;
- Open3D seed `7`, registration point cap `2500`.

The CAD model has 18,095 points and dimensions approximately
`98.721 x 98.739 x 63.774 mm`; the GT-visible oracle has 16,456 raw points,
16,366 after the production statistical filter, and 810 points after the
production voxel step.

## Main evidence

At the GT pose, CAD-to-observed geometry is consistent enough for a metric
registration, but it is not identical. For `models_cad`, CAD-to-observed
nearest-neighbor distances have median `8.41 mm`, p90 `20.50 mm`, and 88.74%
coverage at 20 mm. The reverse observed-to-CAD median is `4.21 mm` with 99.37%
coverage at 20 mm. `models_reconst` is materially closer: CAD-to-observed
median `3.84 mm` and 94.64% coverage at 20 mm.

The current FPFH descriptors are the primary global-registration bottleneck on
this case:

- 947 raw nearest-feature correspondences are produced;
- only 53 are reciprocal (5.60%);
- raw correspondence error at the GT pose has median `55.53 mm`; only 9/947
  (0.95%) are within 10 mm;
- reciprocal correspondence error has median `51.97 mm`; only 1/53 is within
  10 mm and none is within 5 mm.

Open3D therefore emits its mutual-filter fallback warning and uses the raw set.
Strict reciprocal-only RANSAC also remains wrong, so the fallback is not the
sole cause: the descriptor correspondences themselves are overwhelmingly
geometrically incorrect.

The exact current CAD oracle run reproduces the preserved behavior: RANSAC
returns 729 inliers with 7.708 mm RANSAC RMSE but a pose error of 12.0 mm / 171.7°;
ICP then has zero inliers and is rejected. The high RANSAC consensus is not a
GT-correctness measure.

The small scale study (`registration_scale_study.csv`) used 3, 5, and 8 mm
voxels, with radii and thresholds tied to voxel size, and two seeds per scale.
None produced stable correct rotations; the result varies between large
wrong-orientation hypotheses. This does not support changing the production
scale from this single frame.

The GT-initialized ICP study (`icp_capture_range_gt_initialized_diagnostic.json`)
is explicitly diagnostic only. Even exact-GT initialization drifted to
30.9 mm / 25.0° with the current CAD model and point-to-plane settings. Small
and medium perturbations did not yield a stable final pose, so ICP is not an
independently reliable recovery mechanism on this CAD/oracle pair.

The model representation comparison is the strongest controlled distinction:

| representation | RANSAC error | post-ICP error | post-ICP fitness / RMSE |
|---|---:|---:|---:|
| `models_cad` | 14.9 mm / 168.0° | 20.8 mm / 158.9° | 0.8996 / 8.90 mm |
| `models_reconst` | 8.95 mm / 61.8° | **7.47 mm / 2.65°** | **0.9852 / 7.33 mm** |
| `models_eval` | 9.72 mm / 178.3° | 4.32 mm / 176.8° | 0.9676 / 8.63 mm |

These are diagnostic runs with the same Open3D pipeline and no GT initialization.
They do not authorize a production model switch. The local T-LESS dataset
documentation describes CAD and reconstructed models; the original T-LESS paper
describes them as manually created CAD and semi-automatically reconstructed
models. BOP's toolkit uses CAD as the default T-LESS model type, so the result
should be treated as an observed sensor/model mismatch on this case rather than
an undocumented convention change.

## Verification evidence

At a 20 mm nearest-neighbor threshold, using the same 2,500-point refinement cap:

- preserved full-scene false positive: model-to-scene `0.8592`, scene-to-model
  `0.0652`, minimum bidirectional overlap `0.0652`, symmetric mean distance
  `128.9 mm`;
- CAD at GT on the oracle: `0.8780` / `0.9960`, minimum `0.8780`, symmetric
  mean `8.71 mm`;
- diagnostic `models_reconst` post-ICP: `0.9852` / `0.9932`, minimum `0.9852`,
  symmetric mean `4.97 mm`.

The one-sided ICP fitness that accepted the full-scene false positive is model
coverage only: it does not require the entire scene candidate to be explained by
the model. Bidirectional overlap and extent compatibility clearly separate that
false positive from the GT-aligned and reconstructed-model hypotheses. These
metrics remain offline diagnostics, not production thresholds.

## Visual artifacts

- [GT-aligned CAD vs oracle](gt_aligned_cad_vs_oracle.png)
- [normal orientation diagnostic](cad_and_oracle_normals_diagnostic.png)
- [FPFH correspondences scored with GT after matching](fpfh_correspondences_gt_scored.png)
- [RANSAC/model-representation hypotheses](ransac_hypotheses_comparison.png)
- [real RGB pose overlay](real_rgb_pose_hypotheses_overlay.png)
- [GT-initialized ICP perturbations](icp_capture_range_visuals.png)

Files whose names or captions contain `GT ONLY` are diagnostic artifacts and
must not be used as detector inputs.

## Updated failure decomposition

- **DATA / UNITS / FRAMES — working.** The real BOP path, depth conversion,
  camera frame, CAD scale, and GT-aligned geometry are mutually consistent.
- **MODEL REPRESENTATION — likely problematic.** `models_cad` is geometrically
  less close to the Primesense visible surface than `models_reconst` on this
  object; the diagnostic reconst run succeeds where CAD does not.
- **NORMALS — likely problematic for the current descriptor/refinement path.**
  Local normal signs are inconsistent; camera-facing/away variants materially
  change results but do not make CAD registration reliable.
- **FPFH FEATURES — likely problematic.** Correct feature matches are nearly
  absent under the known GT evaluator.
- **GLOBAL RANSAC — inconclusive downstream of poor features.** It finds large
  internally consistent but geometrically wrong consensus sets; no evidence yet
  isolates RANSAC from its bad input correspondences.
- **ICP — likely problematic on this CAD/oracle pair.** It drifts even from
  exact GT initialization and does not provide a stable capture basin here.
- **CANDIDATE GENERATION — already known likely problematic.** The prior
  diagnosis showed DBSCAN merging the target into a giant scene cluster; this
  iteration did not alter or retune it.
- **HYPOTHESIS VERIFICATION — likely problematic in production.** One-sided
  source fitness accepts a small model inside a much larger scene; offline
  bidirectional metrics distinguish the cases but remain unpromoted.

## Reproduction and test status

The diagnostics were executed with the existing container environment, mounting
the repository read-write for experiment artifacts and T-LESS read-only:

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD":/opt/zutik-geometry \
  -v /home/datasets/bop/tless:/datasets/tless:ro \
  -w /opt/zutik-geometry zutik-geometry:local ...
```

The exact inputs, configuration, JSON/CSV outputs, and matrices are retained in
this directory. No production source files were modified in this iteration.
