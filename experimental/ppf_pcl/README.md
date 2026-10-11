# Experimental PCL Point Pair Feature spike

This directory is an architecture spike only. It does not replace the
production Open3D path in `zutik_geometry.registration` and is not imported by
the normal `zutik` CLI.

## Table of Contents

- [Method](#method)
  - [PCL 1.13 hypothesis access](#pcl-113-hypothesis-access)
- [Bridge input and pose convention](#bridge-input-and-pose-convention)
- [Fixed configuration](#fixed-configuration)
- [Build and run](#build-and-run)

Related documentation: the [project overview](../../README.md) and the
[T-LESS PPF experiment results](../../experiments/tless_ppf_spike_v1/README.md).

## Method

The executable is PCL's published Drost-style implementation:

1. `PPFEstimation` computes oriented point-pair signatures for every sampled
   model pair;
2. `PPFHashMapSearch` quantizes angle and distance signatures;
3. `PPFRegistration` samples scene reference points, votes in pose space, and
clusters pose hypotheses.

### PCL 1.13 hypothesis access

The image uses Debian bookworm's PCL 1.13.0. That release does not expose the
newer `getBestPoseCandidates()` accessor in its public header, so this spike
returns the official highest-vote `getFinalTransformation()` only and records
`ranked_pose_accessor_available=false`. It does not reimplement PCL's private
clustering merely to manufacture a top-K interface. If a later experiment
needs top-K evidence, it must use a PCL release that publicly exposes the
candidate list or a separately reviewed upstream-compatible build.

## Bridge input and pose convention

The bridge input is an explicit text format whose first line is
`ZUTIK_PPF_XYZ_NORMALS_V1 m N`, followed by `N` rows of six floating-point
values: `x y z nx ny nz`. There are no implicit millimetre conversions. The
JSON output declares `transform_convention=T_target_source`; when the source
is a CAD object and the target is the camera scene, this is the repository's
`T_camera_object`.

## Fixed configuration

The initial global spike configuration is intentionally fixed before the real
oracle experiment:

| parameter | value |
| --- | ---: |
| angle hash bin | 10 degrees |
| distance hash bin | 5 mm |
| scene reference sampling rate | every 5th point |
| position vote cluster | 30 mm |
| rotation vote cluster | 25 degrees |
| retained hypotheses | 20 |

The Python runner deterministically voxel-samples inputs to a bounded point
count before invoking the bridge. That is an experiment throughput guard, not
a new production default. The point-count and sampling settings are recorded
in each experiment record.

## Build and run

The separate `Dockerfile.ppf-experimental` starts from the existing local
Open3D image and adds PCL only to this experimental image. It does not include
T-LESS. From the repository root:

```bash
docker build -f experimental/ppf_pcl/Dockerfile.ppf-experimental \
  -t zutik-geometry-ppf:local .
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace:ro" \
  -v "$PWD/experiments/tless_ppf_spike_v1:/output" \
  -v /home/datasets/bop/tless:/datasets/tless:ro \
  zutik-geometry-ppf:local \
  python /workspace/scripts/run_tless_ppf_spike.py --all \
    --dataset-root /datasets/tless --output-dir /output
```

The synthetic bridge check is run first by `--all`; a failed synthetic check
prevents the T-LESS run. `--synthetic` runs only that deterministic check.
