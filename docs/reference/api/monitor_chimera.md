# Chimera State Detection — Kuramoto & Battogtokh 2002

The `monitor.chimera` module detects **chimera states** — the
coexistence of phase-locked and incoherent oscillator populations on
the same network — via the per-oscillator local order parameter.

Formally (Kuramoto & Battogtokh 2002, Nonlinear Phenomena in
Complex Systems **5** (4):380–385):

$$
R_i \;=\; \Bigl| \frac{1}{|N(i)|}\sum_{j \in N(i)} e^{\, i (\theta_j - \theta_i)} \Bigr|, \qquad
N(i) = \bigl\{ j : K_{ij} > 0 \bigr\}.
$$

An oscillator is **coherent** when ``R_i > 0.7``, **incoherent** when
``R_i < 0.3``, and **boundary** otherwise. The **chimera index** is
the fraction in the boundary band.

This is the eleventh module migrated to the AttnRes-level standard:
five-language backend chain, bit-exact parity across all four
non-Python backends, multi-backend benchmark, and
``pytest.mark.slow`` stability tests.

---

## 1. Mathematical formalism

### 1.1 Local order parameter

``R_i`` measures how phase-aligned oscillator ``i``'s neighbourhood
is around ``i``. With a perfectly-locked neighbourhood ``R_i = 1``;
with uniformly-distributed phases ``R_i → 1/(|N(i)|)`` (exact
``1/(N−1)`` on all-to-all coupling).

### 1.2 Partition thresholds

| Regime     | Condition       |
| ---------- | --------------- |
| Coherent   | ``R_i > 0.7``   |
| Incoherent | ``R_i < 0.3``   |
| Boundary   | otherwise       |

The thresholds come from the Kuramoto-Battogtokh convention and are
retained as module-level constants (`_COHERENT_THRESHOLD`,
`_INCOHERENT_THRESHOLD`) so downstream callers can override them
without touching the compute kernel.

### 1.3 Chimera index

``χ = (N − |coherent| − |incoherent|) / N`` — the boundary
fraction. ``χ = 0`` on a fully-partitioned state (pure coherent /
pure incoherent) and rises towards 1 when most oscillators are in
the transition band.

### 1.4 Invariants

* Bounded: ``0 ≤ R_i ≤ 1`` always.
* Global shift: adding a constant ``φ`` to every phase leaves
  every ``R_i`` unchanged.
* Isolated oscillator (``|N(i)| = 0``) → ``R_i = 0``.
* Perfect sync → every ``R_i = 1`` → all coherent → ``χ = 0``.

---

## 2. API

```python
from scpn_phase_orchestrator.monitor.chimera import (
    ACTIVE_BACKEND, AVAILABLE_BACKENDS,
    ChimeraState,
    local_order_parameter,
    detect_chimera,
)
```

### 2.1 `local_order_parameter`

```python
def local_order_parameter(
    phases: NDArray,    # (N,)
    knm: NDArray,       # (N, N) — K_ij > 0 defines neighbours
) -> NDArray: ...
```

Returns a ``(N,)`` array of ``R_i`` values. The compute surface for
the 5-backend chain.

### 2.2 `detect_chimera`

```python
def detect_chimera(phases: NDArray, knm: NDArray) -> ChimeraState: ...
```

Calls `local_order_parameter`, then classifies and builds the
:class:`ChimeraState`. The Python thresholding stays single-backend
— it is a handful of list comprehensions.

### 2.3 `ChimeraState`

```python
@dataclass(frozen=True)
class ChimeraState:
    coherent_indices: list[int]
    incoherent_indices: list[int]
    chimera_index: float
```

---

## 3. Backend fallback chain

Resolved at import time **Rust → Mojo → Julia → Go → Python**.
The Rust path reuses the existing `detect_chimera_rust` FFI and
extracts the `local_order` vector from its 4-tuple return value.

### 3.1 Loader probes

| Backend | Probe                                                          | Artefact                              |
| ------- | -------------------------------------------------------------- | ------------------------------------- |
| Rust    | `from spo_kernel import detect_chimera_rust`                   | `spo_kernel` wheel via maturin.       |
| Mojo    | `mojo/chimera_mojo` executable                                 | `mojo build mojo/chimera.mojo …`.     |
| Julia   | `juliacall` + `julia/chimera.jl`                               | Julia 1.11.                           |
| Go      | `ctypes.CDLL("go/libchimera.so")`                              | `go build -buildmode=c-shared`.       |
| Python  | Pure NumPy                                                     | Always available.                     |

### 3.2 Parity tolerances

| Backend | Tolerance | Reason                                    |
| ------- | --------- | ----------------------------------------- |
| Rust    | `1e-12`   | Shared f64; reuses existing FFI.          |
| Julia   | `1e-12`   | `juliacall` direct Float64.               |
| Go      | `1e-12`   | `ctypes` `double*`.                       |
| Mojo    | `1e-9`    | Subprocess text round-trip.               |
| Python  | exact     | Reference.                                |

Measured parity on a 20-oscillator sparse-``K`` problem: all four
non-Python backends within ``1.1e-16`` of the Python reference —
bit-equivalent under the test env.

---

## 4. Per-backend build notes

### 4.1 Rust

Pre-existing `spo-engine/src/chimera.rs`. The dispatcher reuses its
full 4-tuple return (`coherent_indices`, `incoherent_indices`,
`chimera_index`, `local_order`) and extracts only the ``local_order``
vector — Python handles partitioning for consistency with the other
backends.

### 4.2 Julia (`julia/chimera.jl`)

Pure Julia, no packages beyond the standard library. Explicit
``(sr, si)`` accumulators + ``cnt`` neighbour counter; `sqrt(sr² +
si²) / cnt` per oscillator. `@inbounds` on the outer loop.

### 4.3 Go (`go/chimera.go`)

c-shared `.so`. Takes caller-owned pointers via `unsafe.Slice`;
zero heap allocations per call.

### 4.4 Mojo (`mojo/chimera.mojo`)

Stdin executable with one verb (`CHI`). Mirrors the Julia kernel
line-for-line.

### 4.5 Python (`monitor/chimera.py`)

NumPy broadcast: builds the ``(N, N)`` complex unit-vector matrix
``exp(1j · (θ_j − θ_i))``, then masks and averages along ``j`` per
oscillator. Competitive for small ``N`` — at ``N = 64`` the
vectorised form is ~1 ms.

---

## 5. Benchmarks

Measured on the local Ubuntu 24.04 host, 16-thread x86_64, one
warm-up + ten measured calls, sparsity ``P(K_{ij} > 0) = 0.3``.
Reproduce with
`python benchmarks/chimera_benchmark.py --sizes 16 64 256 --calls 10`.

| N   | rust (ms) | mojo (ms) | julia (ms) | go (ms) | python (ms) |
| --- | --------: | --------: | ---------: | ------: | ----------: |
| 16  |    0.0453 |   43.2872 |    0.1080  |  0.6994 |      0.3749 |
| 64  |    0.0711 |   47.5277 |    0.0469  |  0.7299 |      0.7163 |
| 256 |    0.3241 |   91.8331 |    0.7149  |  1.6061 |      6.7363 |

Observations:

* **Rust wins everywhere.** Vectorised `sin` / `cos` plus
  allocation-free iteration.
* **Julia wins the ``N = 64`` cell** thanks to `@inbounds` +
  branch-prediction on the neighbour mask.
* **Python (NumPy)** is competitive at small ``N`` but the ``N³``
  complex-matrix construction dominates from ``N = 256``.
* **Go c-shared** has a fixed ctypes call floor that shows up at
  the small sizes.
* **Mojo subprocess cost** floors at ~43 ms; retained for parity
  coverage only.

Raw JSON: `python benchmarks/chimera_benchmark.py --output /tmp/chi_bench.json`.

---

## 6. Usage examples

### 6.1 Baseline chimera index

```python
import numpy as np
from scpn_phase_orchestrator.monitor.chimera import detect_chimera

rng = np.random.default_rng(0)
N = 64
phases = rng.uniform(0, 2 * np.pi, N)
knm = np.ones((N, N)) - np.eye(N)
state = detect_chimera(phases, knm)
print(f"χ = {state.chimera_index:.3f}  "
      f"(coh {len(state.coherent_indices)}, "
      f"incoh {len(state.incoherent_indices)})")
```

### 6.2 Time evolution on a Kuramoto-Battogtokh ring

```python
from scpn_phase_orchestrator.upde.engine import upde_run

N = 64
ring = np.zeros((N, N))
for i in range(N):
    for off in (-2, -1, 1, 2):
        ring[i, (i + off) % N] = 1.0
phases = np.concatenate([
    np.zeros(N // 2), rng.uniform(0, 2 * np.pi, N // 2)
])

chi_history = []
for _ in range(50):
    phases = upde_run(
        phases, np.zeros(N), ring, np.zeros((N, N)),
        zeta=0.0, psi=0.0, dt=0.01, n_steps=100, method="rk4",
    )
    chi_history.append(detect_chimera(phases, ring).chimera_index)
```

### 6.3 Forcing a backend

```python
from scpn_phase_orchestrator.monitor import chimera as ch_mod
saved = ch_mod.ACTIVE_BACKEND
try:
    ch_mod.ACTIVE_BACKEND = "julia"
    r = local_order_parameter(phases, knm)
finally:
    ch_mod.ACTIVE_BACKEND = saved
```

---

## 7. Tests

Three files (24 tests):

### 7.1 `tests/test_chimera_algorithm.py`

13 tests:

* `TestLocalOrderParameter` — perfect sync → ``1``; uniform on
  circle → ``1/(N−1)``; unit-interval bound; isolated oscillator
  → ``0``; empty input.
* `TestDetectChimera` — perfect sync → all coherent; antiphase
  → all incoherent; chimera index in ``[0, 1]``; partition totals
  to ``N``; empty input returns empty state.
* `TestHypothesis` — random sparse-``K`` inputs are bounded and
  finite across seeds and sizes.
* `TestDispatcherSurface` — active is first of
  `AVAILABLE_BACKENDS`; Python always present.

### 7.2 `tests/test_chimera_backends.py`

8 tests:

* `TestRustParity` — Hypothesis sweep over random ``(N, seed)``
  at `1e-12`.
* `TestJuliaParity` — two seeds at `1e-12`.
* `TestGoParity` — Hypothesis sweep at `1e-12`.
* `TestMojoParity` — two seeds at `1e-9`.
* `TestCrossBackendConsistency` — iterates every
  `AVAILABLE_BACKENDS` under the tolerance matrix + confirms
  `detect_chimera` routes through `local_order_parameter`.

### 7.3 `tests/test_chimera_stability.py`

Three ``@pytest.mark.slow`` tests:

* Global-shift invariance across a 40-oscillator random problem.
* Staged chimera on a narrow-kernel ring yields both classes for
  ≥ 7 / 10 seeds.
* ``N = 500`` stress run — finite, bounded output.

Run all three:

```bash
pytest tests/test_chimera_algorithm.py tests/test_chimera_backends.py
pytest tests/test_chimera_stability.py -m slow
```

---

## 8. Failure modes and caveats

### 8.1 Thresholds are conventions

``0.7`` / ``0.3`` come from Kuramoto & Battogtokh 2002. They are
not measurements. Workloads with heavy common drivers or narrow
chimeras may need project-specific cutoffs — override the module
constants or post-process ``local_order_parameter`` directly.

### 8.2 All-to-all coupling smears chimeras

On a fully-connected ``K``, the local ``R_i`` is nearly the same
for every oscillator (up to ``1/(N−1)``), so the coherent /
incoherent partition is empty even when physics is chaotic.
Genuine Kuramoto-Battogtokh chimeras require a non-local coupling
kernel (e.g. a ring of radius ``r < N/2``).

### 8.3 Rust 4-tuple overhead

The existing Rust FFI returns a 4-tuple including
``coherent_indices`` and ``incoherent_indices`` that the
dispatcher discards. For the dispatcher's hot path (just
``local_order``), a narrower FFI entry would trim per-call cost by
~0.01 ms on ``N = 64``. Not worth the maintenance overhead today.

### 8.4 Mojo subprocess cost dominates

For ``N ≤ 256`` the Mojo path is 100–200× slower than the others
because the subprocess fork + text round-trip is fixed cost. It is
retained for parity coverage only.

### 8.5 Isolated-neighbourhood definition

``K_ij > 0`` is the neighbour test. Negative / zero entries of
``K`` are treated as non-edges — the module does **not** interpret
negative couplings as inhibitory neighbours. Callers who want
signed-edge semantics should pre-threshold ``K`` to positive.

---

## 9. Complexity

| Operation                   | Time            | Space          |
| --------------------------- | --------------- | -------------- |
| `local_order_parameter`     | `O(N²)`         | `O(N)` scratch |
| `detect_chimera`            | `O(N²)` kernel + `O(N)` partition | `O(N)` |

At ``N = 256`` this is 65 536 multiplies — ~0.3 ms on Rust, ~7 ms
on Python, ~90 ms on Mojo.

---

## 10. References

* Kuramoto, Y., Battogtokh, D. (2002). *Coexistence of coherence
  and incoherence in nonlocally coupled phase oscillators.*
  Nonlinear Phenomena in Complex Systems **5** (4), 380–385.
* Abrams, D. M., Strogatz, S. H. (2004). *Chimera states for
  coupled oscillators.* Physical Review Letters **93** (17),
  174102.
* Martens, E. A., Thutupalli, S., Fourrière, A., Hallatschek, O.
  (2013). *Chimera states in mechanical oscillator networks.*
  Proceedings of the National Academy of Sciences **110** (26),
  10563–10567.
* Tinsley, M. R., Nkomo, S., Showalter, K. (2012). *Chimera and
  phase-cluster states in populations of coupled chemical
  oscillators.* Nature Physics **8** (9), 662–665.

---

## 11. Changelog

* **2026-04-18** — Migrated to the AttnRes-level standard. Added
  Julia / Go / Mojo ports of the local order parameter, Python
  bridges, and a 5-backend dispatcher in `monitor/chimera.py`. The
  coherent/incoherent classification stays Python-side for
  consistency across backends. 24 new tests (13 algorithmic + 8
  cross-backend parity + 3 long-run stability) plus the
  multi-backend benchmark harness. Parity measured at ``1.1e-16``
  across all four non-Python backends.
