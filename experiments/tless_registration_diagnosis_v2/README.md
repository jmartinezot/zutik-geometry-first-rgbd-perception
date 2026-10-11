# T-LESS registration diagnosis v2

This directory is a new diagnostic record. `experiments/tless_validation_v1/`
and the frozen manifest are preserved unchanged. The ten cases are the frozen
`test_primesense` model-conditioned cases.

## Table of Contents

- [Scope and data use](#scope-and-data-use)
- [Method and units](#method-and-units)
- [Reproduction](#reproduction)
- [Outputs](#outputs)

Project context and the experiment index are in the [root README](../../README.md#small-local-bop-t-less-evaluation).
The preceding frozen run is [T-LESS validation v1](../tless_validation_v1/README.md).

## Scope and data use

Ground truth is used only after candidate generation for labeling, to construct
explicitly marked oracle clouds, or to calculate post-hoc errors. It is never
passed as an inference initialization, crop, mask, candidate choice, or
threshold.

## Method and units

The bakeoff uses the unchanged frozen geometry settings and compares Open3D
feature-RANSAC + point-to-plane ICP, FGR + point-to-plane ICP, and
feature-RANSAC + Generalized ICP. The registration results are therefore an
oracle-candidate study, not a detector benchmark. `models_eval` is used only by
the official BOP pose-error audit; inference uses CAD and reconstructed models
separately.

Units: internal geometry and `T_camera_object` translations are metres; BOP
CSV and official pose-error translations are millimetres.

## Reproduction

Run the experiment with:

```bash
python scripts/run_tless_registration_diagnosis_v2.py
```

The official toolkit was installed in `/tmp/zutik-geometry-native-venv` from
the checkout at `/tmp/bop_toolkit`. The toolkit parser accepted all three
ten-row CSVs. Official per-scene MSSD error files were also generated for all
three files using the temporary symlink-only standard-layout adapter at
`/tmp/zutik-tless-bop-eval-root`; the dataset itself was not copied or changed.
No aggregate BOP score is reported because these files are selected,
model-conditioned cases rather than complete detector outputs.

## Outputs

Important outputs:

- `posthoc_acceptance_correctness.json`: internal acceptance versus transparent
  post-inference diagnostic correctness and official pose-error values.
- `bop_compatibility.json`: CSV fields, target-key checks, parser result, and
  official evaluator command record.
- `candidate_oracle_analysis.json`: GT-visible candidate recall/purity ceiling.
- `model_geometry_and_scale_audit.json`: CAD/reconstructed geometry, spacing,
  normals, and GT-aligned nearest-neighbor discrepancy.
- `feature_correspondence_diagnostics.json`: raw, strict mutual, and effective
  Open3D fallback correspondences evaluated against GT after matching.
- `registration_oracle_bakeoff.json`: global-initializer and post-refinement
  pose errors for all three variants and both representations.
- `icp_oracle_study.json`: exact-GT and deterministic-small-perturbation ICP,
  explicitly labeled diagnostic-only.
- `aggregate_diagnostics.json` and `hypothesis_ranking.json`: descriptive
  aggregate evidence; neither fits a production score or threshold.

No production defaults, candidate-generation parameters, model representation,
or acceptance rule were changed by this experiment.
