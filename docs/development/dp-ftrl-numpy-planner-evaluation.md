# DP-FTRL NumPy planner conversion evaluation

This one-time evaluation measures the host-side DP-FTRL planner conversion
from Torch/autograd to NumPy/SciPy. It compares only converted numerical
planning paths with their matching Torch implementations; it is not an
end-to-end training or provider-runtime benchmark.

## Snapshot and method

Measurements were taken on 2026-08-18 from the pre-conversion baseline
[`f77a1b0438746288ca4fd89adbde16ae0c4e200d`](https://github.com/JetBrains-Research/opaque/commit/f77a1b0438746288ca4fd89adbde16ae0c4e200d)
(`refactor(dpsgd): make mechanisms provider-neutral`) and the local working
tree containing the NumPy/SciPy conversion. The BLT entries were remeasured
after its feasible-descent recovery was added, using the same isolated and
interleaved procedure.

| Property | Value |
| --- | --- |
| Host | Apple M5 Max, arm64; macOS 26.6 |
| Python | 3.12.13 |
| Numerical stack | NumPy 2.1.3; SciPy 1.16.3; Torch 2.9.1 |
| Precision | `float64` inputs, coefficients, and comparisons |
| Repetitions | Three fresh child processes per implementation and operation, interleaved `baseline/current` order |
| Timing | `time.perf_counter()` around the requested planner/query after imports; median and min--max reported |
| Memory | Per-child `resource.getrusage(RUSAGE_SELF).ru_maxrss`; macOS reports this high-water mark in bytes |

The baseline source tree was loaded ahead of the editable current package in
each baseline child. This retains the original Torch/autograd planner while
using matching public strategy arguments. Cold calls clear the relevant
planner cache. Derived calls first populate the same cache in that child, then
time an equivalent coefficient or plan query. Each process is otherwise
isolated, so cache and allocator state do not cross a comparison boundary.

The peak-RSS measurement includes the complete Python child process and its
imports, rather than attempting to infer an allocator-specific heap delta.
This is the relevant host high-water mark for a process that invokes a
planner, but is not a portable accounting of individual array allocations.

## Workloads

| Path | Matched workload |
| --- | --- |
| Band-MF | Prefix workload (`momentum=1.0`): `n=1,000`, `bands=50`; and `n=5,000`, `bands=100` |
| BLT | `max_buffers=3`, `momentum=0.9`, `min_sep=10`, `max_participations=1`; `n=100` and `n=500` |
| BiSR | `bandwidth=4`, `momentum=0.3`, unnormalized; `n=100`, `min_sep=25`, `max_participations=4` |
| BSR | `bandwidth=4`, `alpha=1.0`, `beta=0.5`; `n=100`, `min_sep=25`, `max_participations=4` |
| Toeplitz | Materialize a `512 × 512` lower-triangular matrix from 16 coefficients `0.72^i` |
| Sensitivity | Single-participation sensitivity of a `384 × 384` lower-triangular geometric matrix with decay `0.97` |

## Performance and memory

The table reports cold planner/query medians, with min--max in parentheses.
`Δ time` and `Δ RSS` are `(NumPy − Torch) / Torch`; negative values are
reductions. RSS is a fresh-process high-water mark.

| Path | Torch time | NumPy/SciPy time | Δ time | Torch peak RSS | NumPy/SciPy peak RSS | Δ RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Band-MF `n=1,000` | 463.20 ms (446.58–480.51) | 69.74 ms (68.99–72.51) | -84.9% | 248.7 MiB | 71.8 MiB | -71.1% |
| Band-MF `n=5,000` | 3,645.15 ms (3,625.16–3,667.60) | 727.11 ms (722.58–727.79) | -80.1% | 311.2 MiB | 83.1 MiB | -73.3% |
| BLT `n=100` | 84.75 ms (84.50–84.98) | 27.81 ms (27.76–31.53) | -67.2% | 237.3 MiB | 70.6 MiB | -70.2% |
| BLT `n=500` | 79.33 ms (79.18–80.69) | 36.94 ms (36.59–37.55) | -53.4% | 237.4 MiB | 70.8 MiB | -70.2% |
| BiSR | 98.33 µs (96.00–105.21) | 31.50 µs (27.50–31.96) | -68.0% | 210.1 MiB | 70.5 MiB | -66.5% |
| BSR | 203.13 µs (198.42–207.29) | 15.00 µs (14.63–16.04) | -92.6% | 209.9 MiB | 70.4 MiB | -66.5% |
| Toeplitz materialization | 425.75 µs (413.46–427.71) | 172.50 µs (170.58–176.92) | -59.5% | 211.0 MiB | 72.2 MiB | -65.8% |
| Sensitivity helper | 1.168 ms (1.026–1.266) | 0.913 ms (0.752–5.842) | -21.0% | 217.0 MiB | 75.3 MiB | -65.3% |

Warm derived-plan queries are deliberately separated from cold optimization.
Their absolute times are small, but the medians establish that cache behavior
did not introduce a material regression except for Band-MF's copy/query
overhead, which is only a fraction of a microsecond.

| Path | Torch derived query | NumPy/SciPy derived query | Δ time |
| --- | ---: | ---: | ---: |
| Band-MF `n=1,000` | 2.50 µs | 3.13 µs | +25.0% |
| Band-MF `n=5,000` | 2.87 µs | 3.17 µs | +10.2% |
| BLT `n=100` | 9.21 µs | 5.96 µs | -35.3% |
| BLT `n=500` | 16.92 µs | 9.21 µs | -45.6% |
| BiSR | 21.54 µs | 15.83 µs | -26.5% |
| BSR | 9.08 µs | 2.50 µs | -72.5% |
| Toeplitz materialization | 44.92 µs | 31.25 µs | -30.4% |
| Sensitivity helper | 566.67 µs | 334.79 µs | -40.9% |

### Conclusion by converted path

- **Band-MF:** cold planning is 5.0–6.6× faster and uses 71–73% less
  peak RSS. Cache-hit queries are microsecond-scale and 10–25% slower; this
  does not offset the cold-planning gain.
- **BLT:** the `n=100` and `n=500` cold planners are 3.0× and 2.1× faster,
  respectively. Both cases reduce peak RSS by about 70%, and cached queries
  are faster. There is no measured CPU-time regression in this path.
- **BiSR and BSR:** closed-form coefficient planning is 3.1× and 13.5×
  faster, respectively, with approximately 66% lower peak RSS.
- **Toeplitz and sensitivity helpers:** host calculations are 2.5× and 1.3×
  faster, respectively, with 65–66% lower peak RSS.

## Numerical and utility comparison

All deterministic comparisons use an absolute and relative elementwise error
when the optimizer identifies a unique coefficient vector. For BLT, whose
optimizer can select non-unique coefficients, the common unpenalized loss and
privacy-relevant sensitivity are the comparison values instead.

| Path | Compared value | Torch | NumPy/SciPy | Difference | Result |
| --- | --- | ---: | ---: | ---: | --- |
| Band-MF `n=1,000` | Toeplitz objective | 18.498054541653847 | 18.498054541653854 | 7.11e-15 | Coefficient max abs error `1.22e-15`; equivalent |
| Band-MF `n=5,000` | Toeplitz objective | 39.25645142835226 | 39.25645142835230 | 4.26e-14 | Coefficient max abs error `2.28e-15`; equivalent |
| BLT `n=100` | Common BLT objective | 2.126025545492097 | 2.126025545492094 | 2.22e-15 | Coefficient max abs error `3.65e-12`; equivalent |
| BLT `n=500` | Common BLT objective | 3.720475229403458 | 2.126025545733023 | -1.594449683670435 | NumPy/SciPy is 42.8% lower; coefficients are non-unique (max abs delta `0.1139`) |
| BLT `n=500` | Sensitivity | 1.646880908886618 | 1.202345922468010 | -0.444534986418608 | NumPy/SciPy is lower by 27.0% |
| BiSR | Coefficients, sensitivity, max inverse row L2 | 1.438986231925707; 2.883495941167799; 1.194921167967264 | same | 0 | Exact for this workload |
| BSR | Coefficients, sensitivity, max inverse row L2 | 1.468770777778565; 2.937541555557130; 1.359453747053997 | same | 0 | Exact for this workload |
| Toeplitz | Materialized matrix; sensitivity squared | L2 32.57084419931439; 2.076355472495902 | same | 0 | Exact |
| Sensitivity helper | Single-participation sensitivity | 4.113450348806113 | 4.113450348806113 | 0 | Exact |

The BiSR and BSR inverse-row-L2 values are the derived inputs used to obtain
realized noise standard deviations. Their equality therefore covers the
realized-noise scale for the measured plan contexts.

The current regression and plan contracts also passed after the measurements:

```text
uv run pytest packages/opaque-dpftrl/tests/matrix_factorization/test_buffered_toeplitz.py \
  packages/opaque-dpftrl/tests/noise/test_blt_mf_noise.py \
  packages/opaque-dpftrl/tests/noise/test_execution_plan.py \
  packages/opaque-dpftrl/tests/accounting/test_namespace.py \
  packages/opaque-dpftrl/tests/accounting/test_per_step.py \
  packages/opaque-dpftrl/tests/accounting/test_per_step_invariants.py -q

190 passed, 5 skipped in 43.57s

uv run pytest packages/opaque-torch/tests/dpftrl/test_planning_regression.py -q

5 passed in 25.45s
```

## Scope boundaries and limitations

- This report does not attribute any effect to `_engine.py`,
  `_mf_gaussian_noise.py`, `_identity.py`, `_lambda_cgd.py`, or
  `_second_moment.py`. Their Torch removal is provider-runtime refactoring,
  not a substitution of a Torch host planner with NumPy.
- Provider-native streaming noise execution and training-loop throughput are
  excluded. They do not have the direct pre-conversion Torch-versus-NumPy host
  planner relationship measured here.
- RSS semantics are platform-specific. The process high-water mark is useful
  for comparing these isolated macOS children, but should not be extrapolated
  as a per-array allocation number or to a different operating system.
- Three repetitions characterize these host workloads, not a performance
  guarantee. Sub-microsecond cache-hit deltas are especially susceptible to
  timer and scheduler noise.
- No reusable benchmark or performance gate was added. The measurement
  harness and baseline checkout were temporary and removed after this report.