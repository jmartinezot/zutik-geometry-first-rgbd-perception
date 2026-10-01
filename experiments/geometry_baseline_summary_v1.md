# Geometry-first baseline summary v1

This record freezes the evidence accumulated before any learned comparison.
The existing experiment directories and manifests are preserved unchanged.
All real-data results below are model-conditioned: the manifest supplies the
object ID for each case, while GT masks and poses are used only for selection
or post-hoc evaluation.

## Current comparison baseline

The comparison baseline is the current production CLI/BOP Open3D path, not the
PPF spike and not the individually best-performing diagnostic variant:

| Item | Frozen value |
| --- | --- |
| Model representation | T-LESS `models_cad`, explicit `cad` selection; metres internally |
| Candidate input | Full filtered scene followed by `geometry_candidates` support-plane/DBSCAN candidate generation |
| Scene preprocessing | Invalid-depth filtering, statistical outlier removal, support-plane removal enabled in CLI/BOP defaults |
| Voxel size | `0.005 m` |
| Normals | CAD `auto` (mesh normals when available, otherwise estimate); observed normals estimated and oriented `towards_camera` |
| Normal search | radius `2 * voxel = 0.010 m`, max neighbors `30` |
| FPFH | standard Open3D `compute_fpfh_feature`; radius `5 * voxel = 0.025 m`, max neighbors `100` |
| Global registration | Open3D feature-matching RANSAC, threshold `0.015 m`, `20,000` iterations, confidence `.999` |
| Refinement | Open3D point-to-plane ICP, threshold `0.020 m`, `35` iterations |
| Verification | bidirectional geometric overlap at `0.020 m`, minimum overlap `.20`; evidence is not a probability |
| Reproducibility | seed `7` for repository-side ordering; Open3D 0.20 Python RANSAC has no per-call seed argument |
| Pose convention | `T_camera_object`; `T_world_object = T_world_camera @ T_camera_object`; translations/distances in metres internally |
| Runtime environment | Python 3.12, Open3D 0.20.0, CPU, Docker image `zutik-geometry:local` |
| Reference hardware | Linux x86_64, dual Intel Xeon E5-2640 v4 (40 logical CPUs), NVIDIA TITAN X/TITAN Xp present but unused |

The frozen ten-case geometry run corresponding to this configuration is the
v1 `B_geometry_cad` record. It is retained as evidence, not re-run or tuned
for this comparison.

## Evidence timeline

### 1. Deterministic synthetic registration

- **Purpose:** verify the vertical slice, transform direction, units, noise,
  outliers, partial visibility, and scene-level negative geometry.
- **Method:** asymmetric synthetic CAD/model, known SE(3), depth-like noise and
  outliers, support/background geometry, Open3D voxel/normal/FPFH/RANSAC/ICP.
- **Cases:** deterministic checked-in synthetic fixture and CLI.
- **Key result:** accepted pose within the explicit translation/rotation test
  tolerances; the test suite covers transform direction and world composition.
- **Conclusion:** the implementation is mechanically valid on a controlled
  scene. This does not establish real T-LESS accuracy.
- **Artifacts:** `tests/test_registration.py`, `zutik_geometry/synthetic.py`,
  `scripts`/CLI synthetic output.

### 2. First real T-LESS smoke test

- **Purpose:** validate the real BOP data, units, camera model, CAD loading,
  output contract, and one complete inference path.
- **Method:** `test_primesense`, scene 3, image 233, model 18, CAD model,
  full filtered scene; GT was used only for case selection and post-hoc scoring.
- **Key result:** the adapter path was mechanically correct, but the accepted
  pose was a false positive: translation error `177.289 mm`, naive rotation
  error `128.932 deg`, RANSAC `351/947`, ICP fitness `.8592`, ICP RMSE
  `8.796 mm`.
- **Conclusion:** production acceptance was not equivalent to pose correctness;
  one-sided ICP fitness can accept a small model inside a much larger scene.
- **Artifacts:** `experiments/tless_smoke/`.

### 3. Single-case candidate-generation diagnosis

- **Purpose:** explain the zero DBSCAN candidates and the full-scene false
  positive without changing production defaults.
- **Method:** same case; GT-visible oracle cloud only for diagnosis, current
  DBSCAN/support-plane instrumentation, bidirectional overlap metrics.
- **Key result:** the target was merged with support/background geometry rather
  than forming a clean object cluster; directional model-to-scene coverage
  alone explained the high one-sided ICP fitness.
- **Conclusion:** candidate isolation and one-sided verification are separate
  ceilings; an exact object crop is not the production input.
- **Artifacts:** `experiments/tless_diagnosis_scene000003_im000233_obj000018/`.

### 4. Registration diagnosis v1/v2

- **Purpose:** determine whether oracle candidate isolation rescues the
  Open3D FPFH front end and whether scale, normals, model representation, or
  ICP capture range is the primary issue.
- **Method:** frozen ten-case manifest; CAD and reconstructed models; GT-aligned
  geometry, standard Open3D FPFH correspondence audits, RANSAC/FGR/GICP and
  point-to-plane ICP diagnostics, with GT used only after correspondence or
  oracle-cloud creation.
- **Key result:** median strict mutual FPFH ratios were about `1.65%` CAD and
  `1.21%` reconstructed; median raw correspondences within 10 mm were about
  `1.34%` and `1.99%`. No tested global/refinement variant reached the
  diagnostic `20 mm / 5 deg` band across the frozen set. Exact-GT ICP also
  drifted (CAD median `21.9 mm / 18.7 deg`; reconstructed `25.4 mm / 39.0 deg`).
  Only `1/10` geometry candidate sets met the diagnostic target recall/purity
  rule.
- **Conclusion:** real-data failure is not explained by candidate generation
  alone. The FPFH correspondence front end is inadequate on this frozen set;
  refinement and candidate isolation remain independent weaknesses.
- **Artifacts:** `experiments/tless_registration_diagnosis_scene000003_im000233_obj000018/`,
  `experiments/tless_registration_diagnosis_v2/`.

### 5. Experimental Drost-style PPF spike

- **Purpose:** test one published model-based global-registration alternative
  without changing production Open3D.
- **Method:** official PCL `PPFEstimation`/`PPFRegistration` bridge in a
  separate experimental Docker image; deterministic synthetic check first;
  same ten oracle clouds; CAD and reconstructed representations; GT only for
  post-hoc BOP errors and analysis.
- **Key result:** synthetic bridge passed (`0.338 mm`, `1.048 deg`). On real
  oracle clouds, CAD top-1 median was `25.77 mm / 151.78 deg`; reconstructed
  median was `8.56 mm / 61.25 deg`, with only `1/10` reconstructed cases in the
  diagnostic `20 mm / 5 deg` band. PCL 1.13 exposed only its final pose, so
  the requested top-K ranking was not available. The PPF image added about
  `76.2 MB` and no production path was changed.
- **Conclusion:** PPF is a useful negative/limited architecture result, not a
  justified production replacement. The baseline remains Open3D; no further
  geometry algorithm search is part of this comparison.
- **Artifacts:** `experimental/ppf_pcl/`, `experiments/tless_ppf_spike_v1/`.

## Scope conclusion

Synthetic success proves the transform and registration plumbing under
controlled geometry, not robust T-LESS detection. The real frozen evidence
shows poor model-conditioned pose accuracy, weak geometry-only candidate
isolation, and false positives under the existing acceptance evidence. PPF
remains experimental and is not production-integrated. A learned comparison
is not reported in this repository because the learned stack is absent and
the required assets are not locally available; see
`experiments/learned_stack_audit_v1.md` and
`experiments/learned_comparison_protocol_v1.md`.

