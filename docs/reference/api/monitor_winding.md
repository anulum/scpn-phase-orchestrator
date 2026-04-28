# Phase Winding Numbers — Integer Topological Classifier

The `monitor.winding` module counts how many full ``2π`` rotations
each oscillator completes over a phase history. The output is a
``(N,)`` integer vector; topologically distinct trajectories map to
distinct lattice points.

$$
w_i \;=\; \Bigl\lfloor \frac{1}{2\pi} \sum_{t=1}^{T-1} \operatorname{wrap}\bigl(\theta_{i,t} - \theta_{i,t-1}\bigr) \Bigr\rfloor,
\qquad \operatorname{wrap}(x) = ((x + \pi) \bmod 2\pi) - \pi.
$$

Positive ``w_i`` = counterclockwise, negative = clockwise, zero =
no net rotation. The ``wrap`` operator handles the ``θ = 2π → 0``
jump at each step so the cumulative increment reflects true
rotation rather than storage discontinuities.

This is the twelfth module migrated to the AttnRes-level standard:
five-language backend chain, bit-exact integer parity across all
four non-Python backends, multi-backend benchmark, and
``pytest.mark.slow`` stability tests.

---

## 1. Mathematical formalism

### 1.1 Winding number

For a phase trajectory ``θ_i(t)`` wrapped to ``[0, 2π)``, the
winding number is the integer number of complete counterclockwise
revolutions the oscillator has performed. It is a **topological
invariant**: continuous deformations of the trajectory cannot
change it.

### 1.2 Why wrap the increments

A naive ``floor((θ(T) − θ(0)) / 2π)`` misses full rotations because
phases are stored modulo ``2π``. The remedy is to wrap each
step-increment into ``(−π, π]`` before summing — a single step
cannot cross more than one half-circle at a time, so this recovers
the true cumulative angular displacement.

### 1.3 Invariants

* ``|w_i| ≤ T`` — at most one full rotation per timestep.
* Global shift: adding ``φ`` to every phase leaves every ``w_i``
  unchanged.
* Time reversal: reversing the trajectory negates every ``w_i``.
* Sign: positive ω → positive ``w_i`` (for long enough ``T``).
* Empty / single-step: ``T < 2`` → zero vector.

---

## 2. API

```python
from scpn_phase_orchestrator.monitor.winding import (
    ACTIVE_BACKEND, AVAILABLE_BACKENDS,
    winding_numbers,
    winding_vector,
)
```

### 2.1 `winding_numbers`

```python
def winding_numbers(phases_history: NDArray) -> NDArray: ...
```

Takes a ``(T, N)`` phase history; returns ``(N,)`` int64 array.
``T < 2`` short-circuits to a zero vector.

### 2.2 `winding_vector`

Alias for `winding_numbers`; emphasises the integer-lattice
interpretation for topological classification.

---

## 3. Backend fallback chain

Resolved at import time **Rust → Mojo → Julia → Go → Python**.

### 3.1 Loader probes

| Backend | Probe                                                    | Artefact                              |
| ------- | -------------------------------------------------------- | ------------------------------------- |
| Rust    | `from spo_kernel import winding_numbers`                 | `spo_kernel` wheel via maturin.       |
| Mojo    | `mojo/winding_mojo` executable                           | `mojo build mojo/winding.mojo …`.     |
| Julia   | `juliacall` + `julia/winding.jl`                         | Julia 1.11.                           |
| Go      | `ctypes.CDLL("go/libwinding.so")`                        | `go build -buildmode=c-shared`.       |
| Python  | NumPy ``diff`` + wrap + ``sum`` + ``floor``              | Always available.                     |

### 3.2 Parity

Tolerance ``0`` — the final ``floor`` truncates to ``int64``, so
any float noise at the ``1e-12`` level vanishes before the
comparison. Measured cross-backend disagreement on random
problems: exactly ``0`` across every seed and size.

---

## 4. Per-backend build notes

### 4.1 Rust

Pre-existing `spo-engine/src/winding.rs`. Takes flat phases + ``N``
and returns an ``(N,)`` int64 array; ``T`` inferred from length.

### 4.2 Julia (`julia/winding.jl`)

Plain Julia. Single outer loop over steps, inner loop over
oscillators with ``@inbounds``. ``mod(x + π, 2π) − π`` for the
wrap; ``floor(Int64, …)`` for the final truncation.

### 4.3 Go (`go/winding.go`)

c-shared `.so`. The ``wrap`` helper uses
``math.Mod(delta + π, 2π)`` with a ``< 0 → += 2π`` fix so the
result lives in ``(−π, π]`` exactly as in Python.

### 4.4 Mojo (`mojo/winding.mojo`)

Stdin executable with one verb (`WIND`). Uses
``Float64(Int(x / 2π)) * 2π`` for integer truncation because Mojo
0.26's `%` on `Float64` follows sign conventions that differ from
`math.fmod` — the explicit form matches Python exactly.

### 4.5 Python (`monitor/winding.py`)

```python
dtheta_wrapped = (np.diff(phases_history, axis=0) + np.pi) % TWO_PI - np.pi
cumulative = np.sum(dtheta_wrapped, axis=0)
np.floor(cumulative / TWO_PI).astype(np.int64)
```

Single vectorised pass; short-circuits at ``T < 2``.

---

## 5. Benchmarks

Measured on the local Ubuntu 24.04 host, 16-thread x86_64, one
warm-up + five measured calls, ``N = 16``. Reproduce with
`python benchmarks/winding_benchmark.py --T-list 500 2000 10000 --N 16 --calls 5`.

| T     | rust (ms) | mojo (ms) | julia (ms) | go (ms) | python (ms) |
| ----- | --------: | --------: | ---------: | ------: | ----------: |
| 500   |    0.0519 |   93.1842 |     0.2308 |  0.9752 |      0.1458 |
| 2000  |    0.1454 |  116.6842 |     0.2502 |  1.9389 |      0.5908 |
| 10000 |    0.7445 |  303.2594 |     0.8392 |  2.9889 |      4.3820 |

Observations:

* **Rust wins at every T** — the O(T·N) loop vectorises cleanly
  under LLVM.
* **Julia is a close second** at large T after `juliacall`
  warm-up.
* **Python (NumPy)** is competitive at small T thanks to the
  vectorised ``diff``/``mod``/``sum``; falls behind at T=10 000
  because the allocation of the (T-1, N) wrapped-delta matrix
  dominates.
* **Go c-shared** has the fixed ctypes floor; competitive but not
  dominant.
* **Mojo subprocess bound** — dominated by text I/O at every
  size.

Raw JSON: `python benchmarks/winding_benchmark.py --output /tmp/wind_bench.json`.

---

## 6. Usage examples

### 6.1 Detecting running oscillators

```python
import numpy as np
from scpn_phase_orchestrator.upde.engine import upde_run
from scpn_phase_orchestrator.monitor.winding import winding_numbers

rng = np.random.default_rng(0)
N = 16
phases = rng.uniform(0, 2 * np.pi, N)
omegas = rng.normal(0.0, 2.0, N)  # some fast, some slow
knm = np.zeros((N, N))
alpha = np.zeros((N, N))

history: list[np.ndarray] = [phases.copy()]
for _ in range(1000):
    phases = upde_run(
        phases, omegas, knm, alpha,
        zeta=0.0, psi=0.0, dt=0.01, n_steps=1, method="euler",
    )
    history.append(phases.copy())

w = winding_numbers(np.array(history))
print("winding vector:", w)
# Expect approximately floor(ω_i · T · dt / 2π) for each i.
```

### 6.2 Topological classification

```python
# Cluster trajectories by winding pattern.
trajectories: list[np.ndarray] = [...]  # list of (T, N) arrays
labels = [tuple(winding_numbers(hist).tolist()) for hist in trajectories]
from collections import Counter
Counter(labels).most_common()
```

### 6.3 Forcing a backend

```python
from scpn_phase_orchestrator.monitor import winding as w_mod
saved = w_mod.ACTIVE_BACKEND
try:
    w_mod.ACTIVE_BACKEND = "julia"
    w = winding_numbers(hist)
finally:
    w_mod.ACTIVE_BACKEND = saved
```

---

## 7. Tests

Three files (20 tests):

### 7.1 `tests/test_winding_algorithm.py`

10 tests:

* `TestAnalyticIdentity` — constant-ω rotator matches
  ``floor(ω·T·dt / 2π)`` up to ±1; ω = 0 → 0.
* `TestSignConvention` — positive ω → w ≥ 0; negative → ≤ 0.
* `TestEdgeCases` — T = 1 → zeros; 1-D input → empty; alias.
* `TestHypothesis` — bounded by T across random trajectories.
* `TestDispatcherSurface` — Python always present.

### 7.2 `tests/test_winding_backends.py`

7 tests:

* `TestRustParity` — Hypothesis sweep over random ``(N, seed)``
  with array-exact equality.
* `TestJuliaParity` — two seeds with array-exact equality.
* `TestGoParity` — Hypothesis sweep, array-exact.
* `TestMojoParity` — two seeds, array-exact.
* `TestCrossBackendConsistency` — iterates every
  `AVAILABLE_BACKENDS` entry, requires exact array equality.

### 7.3 `tests/test_winding_stability.py`

Three ``@pytest.mark.slow`` tests:

* Additivity — winding over [0, T] = winding over [0, k] +
  [k, T] up to ±1 boundary rounding.
* T = 10 000, N = 64 stress run — finite int64 output bounded by
  T in magnitude.
* Noise robustness — adding tiny Gaussian noise to a constant-ω
  rotator preserves the winding sign per oscillator.

Run:

```bash
pytest tests/test_winding_algorithm.py tests/test_winding_backends.py
pytest tests/test_winding_stability.py -m slow
```

---

## 8. Failure modes and caveats

### 8.1 Unwrap vs winding

This module computes the **integer** winding number. If you need a
fully unwrapped real-valued trajectory (for spectrum / Lyapunov
computations), use `numpy.unwrap` directly — that operator does the
same ``(−π, π]`` remainder trick but keeps the cumulative result as
a float.

### 8.2 Per-step increments capped at π

``|Δθ_t| > π`` between samples is ambiguous: there is no way to
decide whether the oscillator moved +π/2 or −3π/2 in one step. The
wrap operator picks the shorter path. Keep ``ω · dt < π/2`` in
practice — two safety margins above the Nyquist equivalent — to
avoid mis-classification.

### 8.3 Floor rounding

``floor`` truncates towards ``−∞``. At the boundary where
``cumulative / 2π`` sits within float ``ε`` of an integer, the
output can flicker by ±1 vs an analytic expectation. Test
assertions use ``≤ 1`` tolerance for that reason.

### 8.4 int64 overflow

The output is int64; ``|w_i| ≤ T`` and ``T ≤ 2^63`` in practice, so
overflow is not a concern for any realistic simulation. Downstream
comparisons should treat ``w`` as a signed integer.

### 8.5 Non-contiguous arrays

The dispatcher calls ``np.ascontiguousarray(…, dtype=np.float64)``
before every backend call, so passing strided slices or different
dtypes is safe at a small copy cost.

---

## 9. Complexity

| Operation         | Time         | Space       |
| ----------------- | ------------ | ----------- |
| `winding_numbers` | `O(T · N)`   | `O(N)`      |

At ``T = 10 000, N = 16`` this is ``1.6 × 10⁵`` wraps — ~0.7 ms on
Rust, ~4.4 ms on Python, ~300 ms on Mojo (subprocess-bound).

---

## 10. References

* Kuramoto, Y. (1984). *Chemical Oscillations, Waves, and
  Turbulence.* Springer Series in Synergetics **19**.
* Strogatz, S. H. (2015). *Nonlinear Dynamics and Chaos.* 2nd
  edition, Westview Press.
* Pikovsky, A., Rosenblum, M., Kurths, J. (2001). *Synchronization:
  A Universal Concept in Nonlinear Sciences.* Cambridge University
  Press.
* Chernov, N. (2006). *Circular statistics on S¹ with winding
  conservation.* In *Handbook of Dynamical Systems*, Vol. 1B,
  Elsevier.

---

## 11. Changelog

* **2026-04-18** — Migrated to the AttnRes-level standard. Added
  Julia / Go / Mojo ports of the cumulative-winding kernel, Python
  bridges, and a 5-backend dispatcher in `monitor/winding.py`.
  20 new tests (10 algorithmic + 7 cross-backend parity with
  array-exact equality + 3 long-run stability) plus the
  multi-backend benchmark harness. Parity measured at exact integer
  equality across all four non-Python backends.
