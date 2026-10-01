# Frozen T-LESS model-conditioned validation

This directory was produced from the persisted manifest copy before any CAD vs
reconstructed comparison. It is an engineering validation of known-model
geometric localization, not a complete detector benchmark: each model ID is
supplied by the manifest. Ground truth is read only after inference for error
reporting and the labelled diagnostic overlays.

Variants:

- `A_baseline_cad_full`: CAD, full filtered scene, historical acceptance without bidirectional verification.
- `B_geometry_cad`: CAD, support-plane removal, geometry candidates, explicit normal handling, bidirectional verification.
- `C_geometry_reconst`: reconstructed model with exactly Variant B logic and thresholds.

All geometry is metres internally. The T-LESS model files are millimetres at
the adapter boundary. `models_eval` and `train_pbr` are not used.

Observed descriptive result (not a benchmark): A accepted 4/10, B accepted
6/10, and C accepted 4/10. Median per-case runtime was 0.343 s, 3.267 s, and
3.286 s respectively. Among accepted cases, median translation errors were
161.4 mm, 484.2 mm, and 281.8 mm; these are not accurate-pose
rates and include false positives. The full per-case evidence is in the CSV
and JSON files.

Reproduction command (dataset is mounted read-only and the repository/output
mount is writable):

```bash
docker run --rm --user "$(id -u):$(id -g)" --entrypoint python \
  -e PYTHONPATH=/opt/zutik-geometry \
  -v "$PWD":/opt/zutik-geometry \
  -v /home/datasets/bop/tless:/home/datasets/bop/tless:ro \
  -w /opt/zutik-geometry zutik-geometry:local \
  scripts/run_tless_validation.py \
  --manifest experiments/tless_validation_manifest_scene_conditioned_v1.json \
  --dataset /home/datasets/bop/tless \
  --output experiments/tless_validation_v1
```

`metadata/inspect.json` and `metadata/dry_run.json` record the real-data
inspection and adapter validation for the selected split/frame/model.
