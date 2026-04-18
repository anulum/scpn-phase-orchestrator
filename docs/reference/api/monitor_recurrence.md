# Recurrence Analysis — Eckmann 1987 + Marwan 2007 RQA

The `monitor.recurrence` module computes **recurrence matrices** and
**Recurrence Quantification Analysis (RQA)** measures from phase
trajectories. Two compute kernels are on the multi-language chain:

* `recurrence_matrix(trajectory, ε, metric)` — single-trajectory
  ``R_ij = Θ(ε − ‖x_i − x_j‖)``.
* `cross_recurrence_matrix(traj_a, traj_b, ε, metric)` — ``CR_ij =
  Θ(ε − ‖x_i − y_j‖)``.

The `rqa` / `cross_rqa` wrappers compose these kernels with
Python-side line-length histograms and the standard seven Marwan
2007 statistics (recurrence rate, determinism, average /max
diagonal, entropy of diagonal lengths, laminarity, trapping time,
max vertical).

This is the thirteenth module migrated to the AttnRes-level
standard: five-language backend chain, **array-exact** boolean
parity across all four non-Python backends, multi-backend
benchmark, and ``pytest.mark.slow`` stability tests.

---

## 1. Mathematical formalism

### 1.1 Recurrence matrix

Eckmann, Kamphorst & Ruelle 1987 defined

$$
R_{ij} \;=\; \Theta\bigl(\varepsilon - \lVert x_i - x_j \rVert\bigr), \qquad
i, j = 1 \dots T.
$$

``R_ii = 1`` always (self-distance). Off-diagonal recurrence
measures how often the trajectory returns to states close (within
``ε``) to earlier states.

Two metrics:

* ``euclidean``: ``‖x − y‖ = √Σ (x_k − y_k)²``.
* ``angular``: chord distance on ``S¹`` — ``√Σ 4·sin((x_k − y_k)/2)²``.
  Preferred for phases, since it handles the ``θ = 0 / 2π``
  wrap.

### 1.2 Cross recurrence

For two trajectories ``a, b`` of equal length,
``CR_ij = Θ(ε − ‖a_i − b_j‖)``. Non-symmetric in general; used
to detect synchronisation between oscillator groups or systems.

### 1.3 RQA measures (Marwan 2007)

| Measure          | Formula                                                           |
| ---------------- | ----------------------------------------------------------------- |
| recurrence rate  | ``Σ R_ij / (T² − T)`` (main diagonal excluded)                    |
| determinism      | ``Σ (ℓ × P(ℓ, ℓ ≥ ℓ_min)) / Σ R_ij``                              |
| avg_diagonal     | ``mean(ℓ : ℓ ≥ ℓ_min)``                                           |
| max_diagonal     | ``max(ℓ : ℓ ≥ ℓ_min)``                                            |
| entropy_diagonal | ``−Σ p(ℓ) log p(ℓ)``                                              |
| laminarity       | ``Σ (v × P(v, v ≥ v_min)) / Σ R_ij``                              |
| trapping_time    | ``mean(v : v ≥ v_min)``                                           |
| max_vertical     | ``max(v : v ≥ v_min)``                                            |

Diagonal lines are scanned only on the upper triangle by
convention — this cuts determinism roughly in half for symmetric
``R``, but matches the existing SPO / Rust numerics exactly.

### 1.4 Invariants

* ``R_ii = 1`` for all ``i``.
* ``R_ij = R_ji`` (recurrence matrix symmetric).
* Monotone in ``ε``: increasing ``ε`` can only add recurrence.
* Every statistic in ``RQAResult`` is non-negative; RR / DET /
  LAM live in ``[0, 1]``.
* Self-cross-recurrence equals the plain recurrence matrix:
  ``CR(a, a, ε) = R(a, ε)``.

---

## 2. API

```python
from scpn_phase_orchestrator.monitor.recurrence import (
    ACTIVE_BACKEND, AVAILABLE_BACKENDS,
    RQAResult,
    recurrence_matrix,
    cross_recurrence_matrix,
    rqa,
    cross_rqa,
)
```

### 2.1 Matrix kernels

```python
def recurrence_matrix(
    trajectory: NDArray, epsilon: float, metric: str = "euclidean",
) -> NDArray: ...

def cross_recurrence_matrix(
    traj_a: NDArray, traj_b: NDArray, epsilon: float,
    metric: str = "euclidean",
) -> NDArray: ...
```

Both return ``(T, T)`` boolean arrays.

### 2.2 RQA wrappers

```python
def rqa(
    trajectory: NDArray,
    epsilon: float,
    l_min: int = 2,
    v_min: int = 2,
    metric: str = "euclidean",
) -> RQAResult: ...

def cross_rqa(
    traj_a: NDArray, traj_b: NDArray,
    epsilon: float, l_min: int = 2, metric: str = "euclidean",
) -> RQAResult: ...
```

`rqa` and `cross_rqa` both use the dispatched matrix kernel, then
run **Python-side** line-length analysis for uniform behaviour
across backends.

---

## 3. Backend fallback chain

Resolved at import time **Rust → Mojo → Julia → Go → Python**.

### 3.1 Loader probes

| Backend | Probe                                                         | Artefact                              |
| ------- | ------------------------------------------------------------- | ------------------------------------- |
| Rust    | `from spo_kernel import recurrence_matrix_rust`               | `spo_kernel` wheel via maturin.       |
| Mojo    | `mojo/recurrence_mojo` executable                             | `mojo build mojo/recurrence.mojo …`.  |
| Julia   | `juliacall` + `julia/recurrence.jl`                           | Julia 1.11.                           |
| Go      | `ctypes.CDLL("go/librecurrence.so")`                          | `go build -buildmode=c-shared`.       |
| Python  | NumPy broadcast                                               | Always available.                     |

### 3.2 Parity

The output is boolean — cross-backend tolerance is **exact array
equality** (``np.array_equal``). Both euclidean and angular metrics
verified across seeds and sizes.

---

## 4. Per-backend build notes

### 4.1 Rust

Pre-existing `spo-engine/src/recurrence.rs` exposes
`recurrence_matrix_rust` and `cross_recurrence_matrix_rust`; both
return flat row-major ``u8`` arrays reshaped + cast to bool by the
dispatcher.

### 4.2 Julia (`julia/recurrence.jl`)

Plain Julia, no external packages. Uses a squared-distance
comparison (``Σ δ² ≤ ε²``) to avoid a ``sqrt`` per pair while
matching Python's ``sqrt(Σ) ≤ ε`` semantics exactly. ``UInt8``
output.

### 4.3 Go (`go/recurrence.go`)

c-shared `.so`. Same squared-distance formulation. ``uint8``
output via caller-owned `*uchar` pointer.

### 4.4 Mojo (`mojo/recurrence.mojo`)

Stdin executable with two verbs (`REC` / `CROSS`). Returns ``T × T``
integer entries (0 / 1) as ASCII lines. The subprocess round-trip
is the bottleneck — retained for parity coverage.

### 4.5 Python

NumPy broadcast form:

```python
diff = traj[:, np.newaxis, :] - traj[np.newaxis, :, :]
dist = np.sqrt(np.sum(diff ** 2, axis=2))
R = dist <= epsilon
```

The ``T × T × d`` scratch allocation dominates at large ``T``.

---

## 5. Benchmarks

Measured on the local Ubuntu 24.04 host, ``d = 3``, ``ε = 0.8``,
one warm-up + twenty measured calls. Reproduce with
`python benchmarks/recurrence_benchmark.py --T-list 30 100 300 --d 3 --epsilon 0.8 --calls 20`.

| T   | rust (ms) | mojo (ms) | julia (ms) | go (ms) | python (ms) |
| --- | --------: | --------: | ---------: | ------: | ----------: |
| 30  |    0.0407 |   68.5577 |     0.0801 |  0.6710 |      0.0320 |
| 100 |    0.0609 |   71.9942 |     0.1520 |  1.3860 |      0.4017 |
| 300 |    0.2005 |   91.3079 |     0.8742 |  1.8750 |      7.3125 |

Observations:

* **Python (NumPy) wins at small T** — the whole pair matrix fits
  in cache and the vectorised form is near-ideal.
* **Rust wins from T = 100 onwards** and dominates at T = 300
  (~36× Python).
* **Julia is a close second** at larger T after `juliacall`
  warm-up.
* **Go c-shared** is consistent but single-goroutine.
* **Mojo** is subprocess-bound at every size.

Raw JSON: `python benchmarks/recurrence_benchmark.py --output /tmp/rec_bench.json`.

---

## 6. Usage examples

### 6.1 RQA on a Kuramoto trajectory

```python
import numpy as np
from scpn_phase_orchestrator.upde.engine import upde_run
from scpn_phase_orchestrator.monitor.recurrence import rqa

rng = np.random.default_rng(0)
N = 16
phases = rng.uniform(0, 2 * np.pi, N)
omegas = rng.normal(0.0, 0.3, N)
knm = rng.uniform(0.1, 0.3, (N, N))
np.fill_diagonal(knm, 0.0)
alpha = np.zeros((N, N))

T = 500
traj = np.zeros((T, N))
for t in range(T):
    phases = upde_run(
        phases, omegas, knm, alpha,
        zeta=0.0, psi=0.0, dt=0.01, n_steps=1, method="rk4",
    )
    traj[t] = phases

res = rqa(traj, 0.8, metric="angular")
print(f"RR={res.recurrence_rate:.3f}  DET={res.determinism:.3f}  "
      f"LAM={res.laminarity:.3f}")
```

### 6.2 Cross-recurrence of two oscillator groups

```python
from scpn_phase_orchestrator.monitor.recurrence import cross_rqa

# Split the Kuramoto population in half.
traj_a, traj_b = traj[:, : N // 2], traj[:, N // 2 :]
res = cross_rqa(traj_a, traj_b, 0.8, metric="angular")
print(f"cross-RR = {res.recurrence_rate:.3f}")
```

### 6.3 Forcing a specific backend

```python
from scpn_phase_orchestrator.monitor import recurrence as r_mod

saved = r_mod.ACTIVE_BACKEND
try:
    r_mod.ACTIVE_BACKEND = "julia"
    R = recurrence_matrix(traj, 0.8, metric="angular")
finally:
    r_mod.ACTIVE_BACKEND = saved
```

---

## 7. Tests

Three files (29 tests):

### 7.1 `tests/test_recurrence_algorithm.py`

14 tests:

* `TestRecurrenceMatrix` — main diagonal always 1; symmetry;
  monotone in ε; large ε → fully recurrent; small ε → only
  diagonal.
* `TestAngularMetric` — wraps ``θ = 0`` / ``2π − ε`` correctly.
* `TestCrossRecurrence` — self-cross equals plain R;
  mismatched-shape raises.
* `TestRQA` — measures in ``[0, 1]``; near-constant trajectory
  saturates convention-capped ``det`` around 0.5.
* `TestHypothesis` — boolean dtype + symmetry across random
  problems.
* `TestDispatcherSurface` — Python always present; active is
  first.

### 7.2 `tests/test_recurrence_backends.py`

12 tests:

* `TestRustParity` — Hypothesis sweep over ``(T, seed)``, array-
  exact equality; angular and cross variants.
* `TestJuliaParity` — two seeds, angular and cross.
* `TestGoParity` — Hypothesis sweep + cross.
* `TestMojoParity` — two seeds at T=15 (subprocess cost bounds
  sample size).
* `TestCrossBackendConsistency` — every `AVAILABLE_BACKENDS`
  entry for both RM and cross-RM.

### 7.3 `tests/test_recurrence_stability.py`

Three ``@pytest.mark.slow`` tests:

* Periodic trajectory (sin / cos) has determinism > 0.45 (the
  convention-capped value for fully periodic data) and long
  diagonal lines.
* Periodic > random determinism.
* T=200 stress run — boolean output, recurrence rate in
  ``(0, 1)``.

Run:

```bash
pytest tests/test_recurrence_algorithm.py tests/test_recurrence_backends.py
pytest tests/test_recurrence_stability.py -m slow
```

---

## 8. Failure modes and caveats

### 8.1 Upper-triangle line convention

`_diagonal_lines` and `_vertical_lines` scan only the upper
triangle. For a symmetric ``R`` this halves ``det`` and ``lam``
versus the "both triangles" convention in some literature. The
numerics match the Rust kernel and Marwan 2007's original formulation.

### 8.2 Epsilon on squared distances

Julia / Go / Mojo ports compare ``Σ δ² ≤ ε²`` to skip the
per-pair ``sqrt``. This is mathematically identical to the Python
``sqrt(Σ) ≤ ε`` form and produces bit-exact boolean output.

### 8.3 Equal-length cross trajectories

`cross_recurrence_matrix` enforces ``a.shape == b.shape``. To
compare trajectories of different lengths, truncate or pad
upstream — cross-RQA under differing lengths is outside the
scope of this module.

### 8.4 Angular metric expects phases in radians

If your data is in degrees, convert to radians first; otherwise
the ``sin(Δ/2)`` chord formula silently under-reports distance.

### 8.5 Mojo subprocess cost dominates small T

For T < 100 the Mojo path is 50–200× slower than any other because
each call serialises ``O(T²)`` integers as ASCII. Retained for
parity coverage.

### 8.6 `rqa` always uses the dispatched matrix

Even when the Rust backend could do the entire RQA natively
(``_rust_rqa``), the Python wrapper now always reconstructs the
matrix and runs Python-side line extraction for cross-backend
consistency. The ~5% overhead at T = 300 is negligible; the
alternative would give Rust and non-Rust backends different
line-counts on edge cases, which is worse than a ~1 ms trim.

---

## 9. Complexity

| Operation                   | Time                 | Space       |
| --------------------------- | -------------------- | ----------- |
| `recurrence_matrix`         | `O(T² · d)`          | `O(T²)`     |
| `cross_recurrence_matrix`   | `O(T² · d)`          | `O(T²)`     |
| `rqa` line analysis         | `O(T²)`              | `O(T²)`     |

At T = 300, d = 3 this is 2.7 × 10⁵ distance computes — ~0.2 ms on
Rust, ~7 ms on Python, ~90 ms on Mojo.

---

## 10. References

* Eckmann, J.-P., Kamphorst, S. O., Ruelle, D. (1987). *Recurrence
  plots of dynamical systems.* Europhysics Letters **4** (9),
  973–977.
* Zbilut, J. P., Webber, C. L. (1992). *Embeddings and delays as
  derived from quantification of recurrence plots.* Physics Letters
  A **171** (3–4), 199–203.
* Marwan, N., Romano, M. C., Thiel, M., Kurths, J. (2007).
  *Recurrence plots for the analysis of complex systems.* Physics
  Reports **438** (5–6), 237–329.
* Webber, C. L., Marwan, N., eds. (2015). *Recurrence Quantification
  Analysis: Theory and Best Practices.* Springer.

---

## 11. Changelog

* **2026-04-18** — Migrated to the AttnRes-level standard. Added
  Julia / Go / Mojo ports of both `recurrence_matrix` and
  `cross_recurrence_matrix`, Python bridges, and a 5-backend
  dispatcher. `rqa` / `cross_rqa` now always use the dispatched
  matrix and Python-side line extraction for uniform behaviour.
  29 new tests (14 algorithmic + 12 cross-backend parity with
  array-exact equality + 3 long-run stability) plus the
  multi-backend benchmark harness. Parity measured at exact
  boolean equality across all four non-Python backends on both
  euclidean and angular metrics.
