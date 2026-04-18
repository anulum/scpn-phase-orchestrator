# Fractal Dimension — Grassberger-Procaccia + Kaplan-Yorke

The `monitor.dimension` module estimates fractal dimensions of
phase-space trajectories. Two independent kernels:

* **Correlation dimension ``D₂``** via the Grassberger-Procaccia
  1983 algorithm — count pairs within ``ε`` and fit a log-log slope.
* **Kaplan-Yorke / information dimension ``D_KY``** from a Lyapunov
  spectrum — Kaplan & Yorke 1979.

This is the tenth module migrated to the AttnRes-level standard:
five-language backend chain (Rust → Mojo → Julia → Go → Python),
bit-exact parity across all four non-Python backends on the
deterministic full-pairs path, multi-backend benchmark, and
``pytest.mark.slow`` stability tests.

---

## 1. Mathematical formalism

### 1.1 Correlation integral

For an embedded trajectory ``x_t ∈ ℝ^d`` of length ``T``, the
Grassberger-Procaccia correlation integral is

$$
C(\varepsilon) \;=\; \frac{2}{T(T-1)} \sum_{t < s} \Theta\bigl(\varepsilon - \lVert x_t - x_s \rVert\bigr),
$$

where ``Θ`` is the Heaviside step. In the scaling region this
behaves as ``C(ε) ∝ ε^{D₂}``. Subsampling ``N_{pairs} ≤
T(T-1)/2`` is routinely used to keep the ``O(T²)`` cost tractable.

### 1.2 Correlation dimension

``D₂`` is the log-log slope of ``C(ε)``:

$$
D_2 \;=\; \lim_{\varepsilon \to 0} \frac{d \log C(\varepsilon)}{d \log \varepsilon}.
$$

In practice SPO identifies the "plateau" — the window of ``ε``-pairs
where the local slope has the lowest variance — and reports the mean
slope as ``D₂``.

### 1.3 Kaplan-Yorke dimension

Given a descending Lyapunov spectrum ``λ_1 ≥ λ_2 ≥ … ≥ λ_N``,

$$
D_{\mathrm{KY}} \;=\; j + \frac{\sum_{i=1}^{j} \lambda_i}{\lvert \lambda_{j+1} \rvert}
$$

with ``j`` the largest index such that ``Σ_{i=1}^{j} λ_i ≥ 0``. It
returns ``0`` when ``λ_1 < 0`` (stable fixed point) and ``N`` when
all exponents are non-negative.

### 1.4 Full-pairs vs subsampled modes

* **Full pairs** (``T(T-1)/2 ≤ max_pairs``): every pair is
  evaluated; the output is deterministic and bit-exact across
  every backend.
* **Subsampled** (``T(T-1)/2 > max_pairs``): a seeded RNG draws
  ``max_pairs`` ``(i, j)`` index pairs. The Python dispatcher owns
  the RNG and passes deterministic indices to Julia / Go / Mojo, so
  those four backends agree bit-exact on the same subsample. The
  **Rust path retains its in-kernel RNG for API stability**, so
  Rust's sample differs from the others by seed — the statistical
  answer is the same to sampling-noise precision but not bit-exact.

### 1.5 Invariants

* ``0 ≤ C(ε) ≤ 1``.
* ``C(0) = 0`` and ``C(∞) = 1``.
* ``C(ε)`` is non-decreasing in ``ε``.
* ``0 ≤ D_KY ≤ N``.
* ``D_KY`` is permutation-invariant in the exponents (internal
  sort).

---

## 2. API

```python
from scpn_phase_orchestrator.monitor.dimension import (
    ACTIVE_BACKEND, AVAILABLE_BACKENDS,
    CorrelationDimensionResult,
    correlation_integral,
    correlation_dimension,
    kaplan_yorke_dimension,
)
```

### 2.1 `correlation_integral`

```python
def correlation_integral(
    trajectory: NDArray,   # (T, d)
    epsilons: NDArray,     # (K,) — sorted internally
    max_pairs: int = 50000,
    seed: int = 42,
) -> NDArray: ...
```

Returns a ``(K,)`` array of fractions in ``[0, 1]``.

### 2.2 `correlation_dimension`

```python
def correlation_dimension(
    trajectory: NDArray,
    n_epsilons: int = 30,
    max_pairs: int = 50000,
    seed: int = 42,
) -> CorrelationDimensionResult: ...
```

Returns a dataclass with ``D2``, the sampled ``ε`` / ``C(ε)`` curves,
local slopes, and the detected scaling-range endpoints.

### 2.3 `kaplan_yorke_dimension`

```python
def kaplan_yorke_dimension(lyapunov_exponents: NDArray) -> float: ...
```

Input order is irrelevant — the kernel sorts internally.

---

## 3. Backend fallback chain

Resolved at import time **Rust → Mojo → Julia → Go → Python**.

### 3.1 Loader probes

| Backend | Probe                                                           | Artefact                              |
| ------- | --------------------------------------------------------------- | ------------------------------------- |
| Rust    | `from spo_kernel import correlation_integral_rust`              | `spo_kernel` wheel via maturin.       |
| Mojo    | `mojo/dimension_mojo` executable                                | `mojo build mojo/dimension.mojo …`.   |
| Julia   | `juliacall` + `julia/dimension.jl`                              | Julia 1.11.                           |
| Go      | `ctypes.CDLL("go/libdimension.so")`                             | `go build -buildmode=c-shared`.       |
| Python  | Pure NumPy                                                      | Always available.                     |

### 3.2 Parity tolerances (full-pairs mode)

| Backend | Tolerance | Reason                                  |
| ------- | --------- | --------------------------------------- |
| Rust    | `1e-12`   | Shared f64; deterministic on full pairs. |
| Julia   | `1e-12`   | `juliacall` direct Float64.             |
| Go      | `1e-12`   | `ctypes` `double*`.                     |
| Mojo    | `1e-9`    | Subprocess text round-trip.             |
| Python  | exact     | Reference.                              |

Measured parity on a 30×3 trajectory with 8 epsilons: all four
non-Python backends bit-equivalent at ``0.0e+00``. Kaplan-Yorke is
deterministic on every backend (``0.0e+00`` diff).

---

## 4. Per-backend build notes

### 4.1 Rust

Pre-existing `spo-engine/src/dimension.rs`: `correlation_integral_rust`
(RNG-owning) + `kaplan_yorke_dimension_rust` (pure f64). Kept as-is
for API backward compatibility — the dispatcher wraps it directly.

### 4.2 Julia (`julia/dimension.jl`)

Pure Julia, no external packages. `correlation_integral` takes
caller-supplied pair index arrays (``idx_i``, ``idx_j``) and an
`@inbounds` loop over pairs then epsilons.
`kaplan_yorke_dimension` mirrors Python's semantics exactly,
including the ``j = -1`` initial sentinel for the all-negative case.

### 4.3 Go (`go/dimension.go`)

c-shared `.so` with two exports. Pair indices arrive as `*longlong`
(matches numpy `int64`). `kaplanYorkeDimension` sorts a scratch
copy descending before the cumulative sum.

### 4.4 Mojo (`mojo/dimension.mojo`)

Stdin executable with two verbs (`CI` / `KY`). Mojo 0.26 lacks a
built-in descending sort, so the KY branch uses a hand-rolled
insertion sort — fine for the usual Lyapunov-spectrum size
(``N ≤ 32``).

### 4.5 Python (`src/.../monitor/dimension.py`)

NumPy vectorised pair subtraction + `sqrt(sum)`. The dispatcher
also owns the RNG that decides whether to subsample; its seed is
threaded through to every non-Rust backend.

---

## 5. Benchmarks

Measured on the local Ubuntu 24.04 host, ``d = 3``, ``K = 20``, one
warm-up + five measured calls. Reproduce with
`python benchmarks/dimension_benchmark.py --T-list 50 150 400`.

| T   | rust (ms) | mojo (ms) | julia (ms) | go (ms) | python (ms) |
| --- | --------: | --------: | ---------: | ------: | ----------: |
| 50  |     1.041 |     44.65 |      0.245 |   0.995 |       0.169 |
| 150 |     1.224 |     50.91 |      0.475 |   1.982 |       1.110 |
| 400 |     2.997 |    104.61 |      2.056 |   3.704 |       8.350 |

Observations:

* **Python wins at T=50** — the NumPy vector ops fit the whole
  pair set in registers; ctypes / juliacall call overheads
  dominate.
* **Julia wins at T=150** — after `juliacall` warm-up the
  `@inbounds` loop approaches the Rust kernel.
* **Rust wins at T=400** — allocation-free iteration over the
  full pair set pays off.
* **Go c-shared** is competitive with Rust up to T=150 then falls
  behind the compiled-release Rust path.
* **Mojo** is subprocess-bound at every size.

Raw JSON: `python benchmarks/dimension_benchmark.py --output /tmp/dim_bench.json`.

---

## 6. Usage examples

### 6.1 Estimating ``D₂`` on a Kuramoto trajectory

```python
import numpy as np
from scpn_phase_orchestrator.upde.engine import upde_run
from scpn_phase_orchestrator.monitor.dimension import correlation_dimension

rng = np.random.default_rng(0)
n = 32
phases = rng.uniform(0, 2 * np.pi, n)
omegas = rng.normal(0.0, 0.3, n)
knm = rng.uniform(0.1, 0.3, (n, n))
np.fill_diagonal(knm, 0.0)
alpha = np.zeros((n, n))

T = 2000
trajectory = np.zeros((T, n))
for t in range(T):
    phases = upde_run(
        phases, omegas, knm, alpha,
        zeta=0.0, psi=0.0, dt=0.01, n_steps=1, method="rk4",
    )
    trajectory[t] = phases

res = correlation_dimension(trajectory, n_epsilons=40, max_pairs=40_000)
print(f"D2 = {res.D2:.3f}  scaling range = {res.scaling_range}")
```

### 6.2 Kaplan-Yorke dimension from a Lyapunov spectrum

```python
from scpn_phase_orchestrator.monitor.lyapunov import lyapunov_spectrum
from scpn_phase_orchestrator.monitor.dimension import kaplan_yorke_dimension

spec = lyapunov_spectrum(phases, omegas, knm, alpha, n_steps=500)
D_KY = kaplan_yorke_dimension(spec)
print(f"D_KY ≈ {D_KY:.3f}  (N = {len(spec)})")
```

### 6.3 Forcing a backend

```python
from scpn_phase_orchestrator.monitor import dimension as dim_mod

saved = dim_mod.ACTIVE_BACKEND
try:
    dim_mod.ACTIVE_BACKEND = "julia"
    C = correlation_integral(trajectory, epsilons)
finally:
    dim_mod.ACTIVE_BACKEND = saved
```

---

## 7. Tests

Three files (28 tests):

### 7.1 `tests/test_dimension_algorithm.py`

14 tests:

* `TestCorrelationIntegral` — monotone-in-ε, unit-interval bound,
  ``C(∞) = 1``, ``C(0) = 0``, ``T ≤ 1`` returns zero, Gaussian-cloud
  slope sanity.
* `TestKaplanYorke` — all-negative ``0``, all-positive ``N``,
  sort-invariance, empty input ``0``, Lorenz-like spectrum lands in
  ``(2, 3)``.
* `TestHypothesis` — random-spectrum KY bounded in ``[0, N]``
  across seeds.
* `TestDispatcherSurface` — backend chain non-empty, Python always
  present.

### 7.2 `tests/test_dimension_backends.py`

11 tests:

* `TestRustParity` — Hypothesis sweep of full-pairs CI at
  ``1e-12`` + KY at ``1e-12``.
* `TestJuliaParity` — two seeds CI + KY at ``1e-12``.
* `TestGoParity` — Hypothesis sweep CI + KY at ``1e-12``.
* `TestMojoParity` — two seeds CI + KY at ``1e-9``.
* `TestCrossBackendConsistency` — every `AVAILABLE_BACKENDS`
  entry under the tolerance matrix.

### 7.3 `tests/test_dimension_stability.py`

Three ``@pytest.mark.slow`` tests:

* 3-D Gaussian cloud ``D₂`` converges towards 3.
* Subsampled ``C(ε)`` on a 300×4 trajectory stays finite and
  bounded.
* ``D_KY`` on 20 random 32-element spectra is always in
  ``[0, 32]``.

Run all three:

```bash
pytest tests/test_dimension_algorithm.py tests/test_dimension_backends.py
pytest tests/test_dimension_stability.py -m slow
```

---

## 8. Failure modes and caveats

### 8.1 Subsampled mode diverges from Rust by seed

Rust's pair RNG is independent from the Python dispatcher's RNG.
Their subsampled outputs agree to sampling noise but are not
bit-exact. Parity tests only check the full-pairs branch; the
subsampled branch is exercised in `TestCorrelationIntegral` on the
Python backend and in the `TestCrossBackendConsistency` comparison
for the non-Rust backends.

### 8.2 GP bias is real

At practical sample sizes ``D₂`` is systematically under-estimated.
The SPO implementation reports the slope of the steepest plateau
window; for publication-grade ``D₂`` you should still cross-check
with the Takens-Theiler or Judd estimator on a longer trajectory.

### 8.3 `_attractor_diameter` uses its own RNG

Distinct from the pair RNG; seeded with `0` and samples up to 200
trajectory points. This is an implementation detail, not part of
the cross-backend contract (it runs only on the Python side).

### 8.4 ``T = 1``

No pairs exist; every backend returns a zero vector of length
``n_k``. The `correlation_dimension` wrapper returns the
dimensionless "empty" `CorrelationDimensionResult` with ``D2 = 0``.

### 8.5 Kaplan-Yorke does **not** re-run Lyapunov

It takes the spectrum as input. For the full pipeline, chain
`lyapunov_spectrum(...)` → `kaplan_yorke_dimension(...)`. If you
pass a mis-ordered spectrum the function sorts internally, so the
output is correct but the rust-side sort cost is not exposed in
the FFI timing.

---

## 9. Complexity

| Operation                 | Time                               | Space            |
| ------------------------- | ---------------------------------- | ---------------- |
| `correlation_integral`    | `O(P · d + P · K)` (``P`` pairs)   | `O(P)` distances |
| `correlation_dimension`   | `O(P · d + P · K) + O(K log K)`    | `O(K + P)`       |
| `kaplan_yorke_dimension`  | `O(N log N)` (sort dominates)      | `O(N)`           |

For full-pairs mode at ``T = 400, d = 3, K = 20`` this is
``~1.6 × 10⁵`` distances — ~3 ms on Rust, ~8 ms on Python.

---

## 10. References

* Grassberger, P., Procaccia, I. (1983). *Characterization of
  strange attractors.* Physical Review Letters **50** (5),
  346–349.
* Kaplan, J. L., Yorke, J. A. (1979). *Chaotic behavior of
  multidimensional difference equations.* Lecture Notes in
  Mathematics **730**, 228–237.
* Theiler, J. (1986). *Spurious dimension from correlation
  algorithms applied to limited time-series data.* Physical
  Review A **34** (3), 2427–2432.
* Pikovsky, A., Politi, A. (2016). *Lyapunov Exponents: A Tool to
  Explore Complex Dynamics.* Cambridge University Press.
* Takens, F. (1981). *Detecting strange attractors in turbulence.*
  In *Dynamical Systems and Turbulence*, Springer LNM **898**,
  366–381.

---

## 11. Changelog

* **2026-04-18** — Migrated to the AttnRes-level standard. Added
  Julia / Go / Mojo ports of the correlation-integral and
  Kaplan-Yorke kernels, Python bridges, and a 5-backend dispatcher
  in `monitor/dimension.py`. Python now owns the pair-subsample RNG
  for cross-backend parity; Rust retains its in-kernel RNG. 28 new
  tests + multi-backend benchmark. Full-pairs parity measured at
  ``0.0e+00`` across all four non-Python backends.
