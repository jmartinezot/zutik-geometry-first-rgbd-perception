# Experimental Drost/PCL PPF spike

This record is separate from and does not modify the frozen
`tless_validation_v1` or `tless_registration_diagnosis_v2` artifacts. It uses
the exact ten-case manifest at
`experiments/tless_validation_manifest_scene_conditioned_v1.json`, with GT
masks only to construct the already-established oracle target clouds. GT poses
are not supplied to PPF, used as initializations, or used to choose a
hypothesis.

## Method and backend decision

Drost et al. define an oriented point-pair feature for points `p1`, `p2` with
normals `n1`, `n2` as
`(||p2-p1||, angle(n1,p2-p1), angle(n2,p2-p1), angle(n1,n2))`. The model
pairs are quantized into a hash table. Scene reference points query that table,
vote for a model reference point and pose, and clustered pose votes become
recognition hypotheses. This is a model-global/local-voting method, rather
than local FPFH descriptors followed by correspondence RANSAC. It explicitly
uses oriented normals, permits a sparse scene reference sample, and is
designed for clutter and partial visibility. Model pair preprocessing can be
cached; the pair enumeration is the principal cost and motivates the bounded
experiment sampling.

The spike selected official PCL `PPFEstimation` and `PPFRegistration` rather
than a custom approximation. PCL is the established C++ implementation of the
Drost method and provides the required hash search, scene-reference sampling,
voting, and clustering. No credible maintained Python PPF binding was found;
the Python-first repository therefore calls a narrow C++ subprocess bridge.
A new Python implementation was rejected because it would recreate another
approximation of the global front end and would have a higher validation and
maintenance risk. ROS was not introduced.

The implementation choices were:

| option | published-method fidelity | dependency/maintenance cost | decision |
|---|---|---|---|
| PCL 1.13 `PPFEstimation` + `PPFRegistration` | highest practical fidelity; official Drost-style API | C++/CMake subprocess and a sizeable native runtime; PCL is BSD-licensed | selected for the spike |
| maintained Python PPF package | no credible maintained package with this exact model-based PPF API was found in the inspected environment | lower integration cost only if one existed; otherwise dependency and reproducibility risk | not selected |
| new NumPy/Python implementation | could be made faithful only with another independently audited voting/hash implementation | high correctness risk, especially normals, voting, clustering, and top-K behavior | explicitly rejected |

PCL is distributed under the BSD license. The spike does not vendor PCL source;
the experimental Docker build installs the distribution package and links the
official libraries. PCL's model-pair preprocessing can be cached in a future
adapter, but this first bridge rebuilds the hash for each invocation so its
runtime is explicit and reproducible.

The image uses Debian bookworm PCL **1.13.0**. Current PCL documentation
describes a newer public `getBestPoseCandidates()` accessor, but PCL 1.13's
public header exposes only `getFinalTransformation()`. The bridge reports this
as `ranked_pose_accessor_available=false`, returns the official top-1 result,
and does not reimplement PCL's private clustering. Consequently this record
does not contain a genuine top-K evaluation: `best_of_k` fields equal the only
exposed result and are explicitly not inference or top-K evidence.

Primary references:

- [Drost et al., CVPR 2010](https://iacl.ece.jhu.edu/proceedings/cvpr2010/papers/0993.pdf)
- [PCL PPFEstimation documentation](https://pointclouds.org/documentation/classpcl_1_1_p_p_f_estimation.html)
- [PCL PPFRegistration documentation](https://pointclouds.org/documentation/classpcl_1_1_p_p_f_registration.html)
- [PCL 1.13 PPFRegistration header](https://raw.githubusercontent.com/PointCloudLibrary/pcl/pcl-1.13.1/registration/include/pcl/registration/ppf_registration.h)
- [An improved PPF method evaluated on T-LESS](https://arxiv.org/abs/1802.08516)

## Fixed configuration

All 20 real runs use one configuration, without per-object tuning:

```text
PPF angle hash bin:                  10 deg
PPF distance hash bin:               0.005 m
scene reference sampling:            every 5th point
position vote clustering:            0.030 m
rotation vote clustering:            25 deg
requested top-K:                     20 (not exposed by PCL 1.13)
preprocessing voxel size:             0.008 m
model/scene point caps:               500 / 800
model normals:                        CAD mesh normals when present
observed normals:                     Open3D, oriented towards camera
internal units:                       metres
pose convention:                      T_camera_object = T_target_source
```

The caps are an experiment throughput guard for PPF's pair processing, not a
production default. The existing Open3D FPFH/RANSAC/ICP path and all
candidate-generation defaults remain unchanged.

## Synthetic validation

The bridge first runs a deterministic asymmetric box with an offset raised
feature, 0.7 mm Gaussian noise, 14.4% deterministic model points missing
(85.6% retained), and 120 clutter points. The asymmetric feature is important:
a plain rectangular
box has exact 180-degree symmetries and would make an otherwise correct
registration look like a rotation failure.

Result: **0.338 mm translation error and 1.048 degrees rotation error** for
the only PCL 1.13 exposed hypothesis. The synthetic result is in
`synthetic_result.json`.

## Frozen oracle result

The experiment ran all ten oracle clouds for both `models_cad` and
`models_reconst` (20 registrations total). There were no bridge failures or
empty PPF results. The primary official BOP errors below use `models_eval` and
the existing symmetry metadata after inference:

| representation | top-1 median MSSD | top-1 median TE | top-1 median naive RE | diagnostic 20 mm / 5 deg |
|---|---:|---:|---:|---:|
| CAD | 102.76 mm | 25.77 mm | 151.78 deg | 0/10 |
| reconstructed | 47.00 mm | 8.56 mm | 61.25 deg | 1/10 |

The reconstructed representation is materially better on this frozen oracle
set, especially in translation, but still has large rotation failures and only
one case in the diagnostic band. CAD has no case in that band. Since PCL 1.13
exposes no top-K candidate list, the post-hoc best-of-K count is not
available; the stored `best_of_k` values are the single returned hypothesis,
not an oracle selection over hidden PCL candidates.

The official symmetry statuses in the rows include identity-only,
identity-plus-discrete, and continuous-symmetry-not-discretized models. The
official MSSD values are retained as the symmetry-aware primary metric; naive
rotation is reported for diagnosis and can be misleading for symmetric
objects.

## Refinement comparison

Refinement was diagnostic only and started from raw PPF top-1, never from GT:

- CAD + GICP improved translation in 7/10 and rotation in 3/10; median final
  errors were 21.89 mm / 165.25 degrees, so it did not fix orientation.
- reconstructed + GICP improved translation in 7/10 and rotation in 4/10;
  median final errors were 4.40 mm / 63.73 degrees. Translation improved, but
  median rotation was slightly worse than raw PPF.
- point-to-plane ICP improved translation in only 1/10 CAD and 2/10
  reconstructed cases, and produced median final errors of 186.45 / 153.89
  degrees for CAD and 43.86 / 138.45 degrees for reconstructed geometry.

These results support keeping refinement optional and evidence-driven; they do
not justify adding ICP automatically to a future PPF backend.

Bidirectional overlap was computed with the existing 20 mm, two-sided metric
for every returned hypothesis. It cannot be evaluated as a ranker here because
only one PCL hypothesis is public per run. It is also not sufficient by itself:
for example, a CAD hypothesis has 0.941 bidirectional overlap while its naive
rotation error is 35.76 degrees, and a reconstructed hypothesis has 0.961
overlap with 24.99 degrees rotation error. The full per-case values and
post-hoc official labels are in `ppf_hypotheses.json`; the explicit ranking
limitation is in `ranking_analysis.json`.

## Reproduction

Build the isolated image; the production image is not modified:

```bash
docker build -f experimental/ppf_pcl/Dockerfile.ppf-experimental \
  -t zutik-geometry-ppf:local .
```

The final image is 1,725,622,522 bytes versus 1,649,430,268 bytes for the
Open3D image, an increase of about 76.2 MB. The compiler and `libpcl-dev` are
only in the build stage. T-LESS is mounted read-only; it is not copied into
the image. The exact real run used:

```bash
docker run --rm --user "$(id -u):$(id -g)" --entrypoint python \
  -e PYTHONPATH=/workspace:/bop_toolkit:/hostvenv \
  -v "$PWD:/workspace:ro" \
  -v "$PWD/experiments/tless_ppf_spike_v1:/output" \
  -v /home/datasets/bop/tless:/datasets/tless:ro \
  -v /tmp/bop_toolkit:/bop_toolkit:ro \
  -v /tmp/zutik-geometry-native-venv/lib/python3.12/site-packages:/hostvenv:ro \
  zutik-geometry-ppf:local \
  /workspace/scripts/run_tless_ppf_spike.py --all \
  --dataset-root /datasets/tless \
  --manifest /workspace/experiments/tless_validation_manifest_scene_conditioned_v1.json \
  --output-dir /output
```

The BOP toolkit and SciPy mounts are evaluation adapters only; neither is a
new project dependency. Native regression tests passed with
`/tmp/zutik-geometry-native-venv/bin/pytest -q`: **29 passed**. The same
complete suite passed in the production image and the experimental image:
**29 passed** in each (the read-only mount only produced a pytest cache warning).

## Decision

On this oracle-only ten-case set, reconstructed-model PPF is substantially
better than the frozen FPFH oracle baseline in translation and descriptive
rotation median (v2 FPFH reconstructed median approximately 35.10 mm / 113.59
degrees after point-to-plane ICP), while CAD PPF is not. The improvement is
not consistent enough for production integration: top-1 reconstructed PPF has
only 1/10 in the 20 mm / 5 degree diagnostic band, genuine top-K evidence is
unavailable with the installed PCL API, and the additional dependency is
nontrivial. Candidate generation was not tested here; every input was an
oracle object cloud. PPF remains a promising experimental direction, not a
replacement decision.

This comparison is directional, not a matched benchmark: both methods use the
same ten GT-derived oracle clouds and post-hoc evaluation, but frozen FPFH used
5 mm / 2500-point Open3D preprocessing and point-to-plane ICP in its headline
v2 result, while PPF uses 8 mm / 500-or-800-point preprocessing and the PPF
pose before refinement in this headline table. Those unequal assumptions are
retained explicitly rather than hidden behind a claim of a clean algorithmic
win.
