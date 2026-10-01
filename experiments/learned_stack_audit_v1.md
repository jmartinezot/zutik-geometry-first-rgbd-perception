# Learned perception stack audit v1

Audit performed against the repository source tree, `pyproject.toml`, Docker
files, scripts, tests, and the checked native environment. No large assets were
downloaded.

## Findings

| Component | Repository status | Native environment status | Runnable here? |
| --- | --- | --- | --- |
| Grounding DINO | Referenced in the guide and README only; no wrapper, config, or inference script | Python module absent | No |
| YOLO-World | Referenced in the guide and README only; no wrapper, config, or inference script | `ultralytics` absent | No |
| SAM 2 | Referenced in the guide and README only; no wrapper, config, or inference script | `sam2` absent | No |
| FoundationPose | Referenced in the guide and README only; no wrapper, config, or inference script | module absent; PyTorch absent | No |
| Learned end-to-end pipeline | No implementation, service, adapter, or evaluation command | No learned runtime | No |
| Checkpoints | None in the repository or experiment records | No compatible checkpoint found locally | No |

The only declared runtime dependencies are `numpy` and `open3d`; test/BOP
extras add `pytest`/`Pillow`. The native validation environment contains
Open3D, BOP toolkit helpers, and the geometry package, but not PyTorch,
`groundingdino`, `ultralytics`, `sam2`, `foundationpose`, or `transformers`.
There is no learned Dockerfile, Conda/environment file, model configuration,
RGB-D preprocessing wrapper, learned pose serializer, or learned evaluation
script to exercise.

The repository has no World Semantic Model interface. Its only related
boundary is the narrow `GraphReadyObjectRecord` adapter in the geometry path.
The T-LESS adapter does provide RGB/depth/intrinsics/CAD loading and the
documented `T_camera_object`/metre contract, but there is no learned component
to consume those inputs.

## Stop-condition decision

The learned smoke test and frozen learned comparison were **not run**. Running
them would require a major new integration project plus external dependencies
and checkpoints, contrary to the requested no-download/no-surgery boundary.
No learned accuracy, runtime, or winner claim is fabricated.

## Reproducible audit commands

```bash
rg -n -i 'grounding.?dino|yolo.?world|sam.?2|sam2|foundation.?pose|torch|checkpoint|weights|learned' .
/tmp/zutik-geometry-native-venv/bin/python - <<'PY'
import importlib.util
for name in ['torch', 'groundingdino', 'ultralytics', 'sam2', 'foundationpose', 'transformers']:
    print(name, bool(importlib.util.find_spec(name)))
PY
```

