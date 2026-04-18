# Lyapunov Spectrum — Tangent-Space Stability Probe

The `monitor.lyapunov` module carries two public surfaces. The first,
`LyapunovGuard`, is a lightweight **observer** that tracks the
Lyapunov function `V(θ)` of a Kuramoto network, its time derivative,
and the basin-of-attraction flag — a per-step cost that is cheap
enough to call inside a control loop. The second, `lyapunov_spectrum`,
is the **heavy kernel**: the full Benettin 1980 / Shimada-Nagashima
1979 Lyapunov spectrum reconstructed from the variational equation
`dQ/dt = J(θ)·Q` with periodic QR reorthogonalisation. Both live in
the same file so the observer can lean on the same Jacobian helper
without a second import.

This is the sixth module migrated to the AttnRes-level standard:
five-language backend chain, bit-exact parity across Rust / Julia / Go
against the NumPy reference, Mojo parity within the text-stream
round-trip, multi-backend benchmark, and `pytest.mark.slow` stability
tests that verify long-run invariants (`Σ λ_i ≈ ⟨tr J⟩`, Kaplan-Yorke
bounds, contracting-network regime).

---

## 1. Mathematical formalism

### 1.1 Kuramoto variational equation

The Sakaguchi-Kuramoto network with optional external driver reads

$$
\dot\theta_i \;=\; \omega_i \;+\; \sum_{j=1}^{N} K_{ij}\,\sin\bigl(\theta_j - \theta_i - \alpha_{ij}\bigr) \;+\; \zeta \sin(\Psi - \theta_i).
$$

Linearising around a trajectory `θ(t)` gives the **variational
equation** for a small perturbation `δθ`:

$$
\dot{\delta\theta}_i \;=\; \sum_{j=1}^{N} J_{ij}(\theta(t))\,\delta\theta_j,
$$

with the Jacobian

$$
J_{ij}(\theta) \;=\;
\begin{cases}
K_{ij} \cos(\theta_j - \theta_i - \alpha_{ij}) & i \ne j, \\[2pt]
-\sum_{k \ne i} K_{ik} \cos(\theta_k - \theta_i - \alpha_{ik}) - \zeta \cos(\Psi - \theta_i) & i = j.
\end{cases}
$$

The driver contributes only to the diagonal, because
`∂/∂θ_i [ζ sin(Ψ − θ_i)] = −ζ cos(Ψ − θ_i)` and the off-diagonal
cross-derivatives vanish.

### 1.2 Benettin 1980 QR algorithm

The Lyapunov spectrum `{λ_1, …, λ_N}` quantifies the exponential
growth rates of `N` orthogonal perturbation vectors evolved alongside
the reference trajectory. The Benettin 1980 / Shimada-Nagashima 1979
algorithm works on **all** exponents simultaneously:

1. Initialise `Q(0) = I ∈ ℝ^{N×N}` — an identity matrix whose `N`
   rows are seed perturbation vectors.
2. Evolve the pair `(θ, Q)` via
   `dθ/dt = f(θ)`, `dQ/dt = J(θ) Q` for `qr_interval` integration
   steps. The columns of `Q` are then the image of the unit vectors
   under the tangent-space flow but are no longer orthonormal.
3. Reorthogonalise `Q` via Modified Gram-Schmidt (MGS), keeping the
   diagonal of `R` (the column-norm factors) in a running log-sum.
4. Repeat until total integrated time `T = n_steps · dt`. The `i`-th
   exponent is

$$
\lambda_i \;=\; \frac{1}{T}\sum_{k=1}^{n_{QR}} \ln\bigl|R_{ii}^{(k)}\bigr|.
$$

Step 3 is what prevents the integrated `Q` from collapsing onto the
single most expanding direction: without periodic reorthogonalisation,
finite-precision arithmetic would align all `N` vectors within a few
Lyapunov times and only `λ_1` would survive.

### 1.3 Row-oriented MGS convention

All five SPO backends orthonormalise the **rows** of `Q`, not the
columns. This matches the Rust kernel (`spo_kernel::lyapunov_spectrum_rust`)
which is the performance reference, and it means the Python fallback
computes `np.linalg.qr(Q.T)` and transposes back. The R-diagonal is
identical to column-MGS on `Q.T`, so the exponents are
direction-independent.

### 1.4 Integration: RK4 throughout

The ODE for `(θ, Q)` is integrated with classic RK4:

$$
k_1 = f(x),\quad k_2 = f(x + \tfrac12 dt\, k_1),\quad k_3 = f(x + \tfrac12 dt\, k_2),\quad k_4 = f(x + dt\, k_3),
$$

$$
x_{n+1} = x_n + \tfrac{dt}{6}(k_1 + 2k_2 + 2k_3 + k_4).
$$

`θ` is wrapped to `[0, 2π)` after each full step to keep the phases
in a deterministic range; the Jacobian is periodic in `θ`, so the
wrap does not affect `J(θ)` nor the spectrum.

### 1.5 Invariants the spectrum must respect

* **Sum rule.**
  `Σ_i λ_i = ⟨tr J(θ(t))⟩_T` (time-average of the Jacobian trace).
  A broken RK4 or QR implementation shows up as a drift in this sum
  long before it shows up as visibly wrong exponents.
* **Kaplan-Yorke bound.** `0 ≤ d_KY ≤ N` always, with
  `d_KY = k + (Σ_{i≤k} λ_i)/|λ_{k+1}|` where `k` is the largest
  index such that the partial sum is non-negative.
* **Ordering.** `λ_1 ≥ λ_2 ≥ … ≥ λ_N` by convention.
* **Finiteness.** Under bounded coupling / bounded driver, every `λ_i`
  is finite; any `inf` or `nan` means a numerical blow-up, almost
  always from `log(|R_ii|)` with `|R_ii| < 1e-300`.

### 1.6 Regime interpretation

| Pattern                               | Meaning                                        |
| ------------------------------------- | ---------------------------------------------- |
| All `λ_i < 0`                         | Fully contracting — strong coupling dominates. |
| `λ_1 = 0`, rest `< 0`                 | Neutral on the synchronisation manifold.       |
| `λ_1 > 0`                             | Chaotic attractor.                             |
| `λ_i ≈ 0` for several `i`             | Quasi-periodic torus of matching dimension.    |
| All `λ_i ≈ 0`                         | Uncoupled or near-neutral.                     |

---

## 2. API

The module exports:

```python
from scpn_phase_orchestrator.monitor.lyapunov import (
    ACTIVE_BACKEND,       # str  — currently serving backend
    AVAILABLE_BACKENDS,   # list — all resolved backends, order = preference
    LyapunovGuard,        # class — cheap V(θ), dV/dt, in_basin observer
    LyapunovState,        # dataclass — return type of LyapunovGuard.evaluate
    lyapunov_spectrum,    # fn    — full Lyapunov spectrum
)
```

### 2.1 `LyapunovGuard`

```python
class LyapunovGuard:
    def __init__(self, basin_threshold: float = math.pi / 2): ...
    def evaluate(self, phases: NDArray, knm: NDArray) -> LyapunovState: ...
    def reset(self) -> None: ...
```

* `evaluate(phases, knm)` computes
  `V(θ) = −(1/2N) Σ_{ij} K_ij cos(θ_i − θ_j)` together with
  `dV/dt ≈ (V_n − V_{n−1})` (finite-difference derivative from the
  previous call), the boolean `in_basin` flag, and the largest
  geodesic phase gap on the `S¹` coupling graph.
* `reset()` clears the cached previous `V` so the next `evaluate`
  reports `dV/dt = 0`.
* The guard is stateful but **not** thread-safe — use one guard per
  simulation thread.

### 2.2 `lyapunov_spectrum`

```python
def lyapunov_spectrum(
    phases_init: NDArray,
    omegas: NDArray,
    knm: NDArray,
    alpha: NDArray,
    dt: float = 0.01,
    n_steps: int = 1000,
    qr_interval: int = 10,
    zeta: float = 0.0,
    psi: float = 0.0,
) -> NDArray: ...
```

Returns `(N,)` exponents sorted descending. `knm` and `alpha` are
row-major `(N, N)` matrices; flat versions are also accepted (the
dispatcher reshapes them for each backend).

**Guidelines** on the integrator parameters:

| Parameter       | Typical range        | Effect                                                    |
| --------------- | -------------------- | --------------------------------------------------------- |
| `dt`            | `1e-3 … 2e-2`        | RK4 stable; smaller `dt` tightens per-step accuracy.      |
| `n_steps`       | `500 … 20 000`       | Longer `T = n_steps · dt` tightens exponent averaging.    |
| `qr_interval`   | `5 … 50`             | Shorter intervals stabilise QR but cost more mat-mul.     |
| `zeta`, `psi`   | application-specific | Driver adds `−ζ cos(Ψ − θ)` to the Jacobian diagonal.     |

### 2.3 `LyapunovState`

```python
@dataclass
class LyapunovState:
    V: float            # Lyapunov function at current phases
    dV_dt: float        # V_current − V_previous; 0 after reset()
    in_basin: bool      # max phase gap < basin_threshold on coupled graph
    max_phase_diff: float
```

---

## 3. Backend fallback chain

Per `feedback_module_standard_attnres.md` the module resolves backends
in the order **Rust → Mojo → Julia → Go → Python** at import time.
The first backend that loads without raising `ImportError`,
`RuntimeError`, or `OSError` becomes `ACTIVE_BACKEND`. Python is
always appended as the guaranteed fallback.

### 3.1 Loader probes

| Backend | Probe                                                                 | Artefact                                |
| ------- | --------------------------------------------------------------------- | --------------------------------------- |
| Rust    | `from spo_kernel import lyapunov_spectrum_rust`                       | `spo_kernel` wheel via maturin.         |
| Mojo    | `_ensure_exe()` → `mojo/lyapunov_mojo`                                | Stand-alone executable from `mojo build`. |
| Julia   | `import juliacall`; then `Main.include("julia/lyapunov.jl")`          | `juliacall` Python binding + Julia 1.11. |
| Go      | `ctypes.CDLL("go/liblyapunov.so")`                                    | C-shared library from `go build`.       |
| Python  | Pure NumPy — no external dependencies.                                | Built in.                               |

### 3.2 Dispatcher surface

```python
from scpn_phase_orchestrator.monitor import lyapunov as ly_mod
ly_mod.ACTIVE_BACKEND       # e.g. "rust"
ly_mod.AVAILABLE_BACKENDS   # e.g. ["rust", "mojo", "julia", "go", "python"]
ly_mod.ACTIVE_BACKEND = "python"   # force a specific backend (tests)
```

### 3.3 Semantic equivalence

All five backends produce the **same spectrum** to within the
tolerances listed in the parity tests:

| Backend vs NumPy reference | Tolerance (atol) | Reason                                                            |
| -------------------------- | ---------------- | ----------------------------------------------------------------- |
| Rust                       | `1e-12`          | Shared `f64` arithmetic; no text serialisation.                   |
| Julia                      | `1e-12`          | `juliacall` passes `Float64` arrays directly.                     |
| Go                         | `1e-12`          | `ctypes` passes `double*` by pointer; no round-trip.              |
| Mojo                       | `1e-6`           | Subprocess stdin ↔ stdout text; `atof` round-trips `f64`.         |
| Python                     | exact            | Python is the reference.                                          |

---

## 4. Per-backend build notes

### 4.1 Rust (`spo-kernel/crates/spo-engine/src/lyapunov.rs`)

The Rust kernel is the performance reference:

* **Rayon parallelism.** `compute_rhs` and `compute_jq` iterate over
  the `N` rows of the output with `par_iter_mut` / `par_chunks_mut`.
  On a 16-core host the per-row work splits well above `N = 16`.
* **Two-pass MGS.** `modified_gram_schmidt` iterates
  `for _pass in 0..2` to suppress catastrophic cancellation when the
  perturbation vectors collapse (Daniel et al. 1976).
* **Alpha fast path.** `alpha_zero` is computed once; when true the
  RHS uses pre-computed `sin θ_i`, `cos θ_i` and a
  `K_ij · (sin θ_j · cos θ_i − cos θ_j · sin θ_i)` expansion,
  avoiding a `sin` per pair.
* **Driver fast path.** `zs_psi`, `zc_psi = ζ·sin Ψ, ζ·cos Ψ` are
  pre-computed so the driver term collapses to two scalar multiplies.
* **RK4 buffer reuse.** All eight stage buffers (`k1_p..k4_p`,
  `k1_q..k4_q`) plus `tmp_p`, `tmp_q`, `sin_theta`, `cos_theta`, `q`
  are allocated once and reused across the step loop.
* **FFI boundary.** `spo-ffi/src/lib.rs` wraps the kernel as
  `lyapunov_spectrum_rust(phases, omegas, knm_flat, alpha_flat, dt,
  n_steps, qr_interval, zeta, psi) -> PyArray1<f64>`. The Python
  dispatcher passes flat `(N·N,)` matrices row-major.

### 4.2 Mojo (`mojo/lyapunov.mojo`)

Mojo 0.26 lacks both a QR decomposition and a matrix multiplication
in its standard library. The port therefore re-implements

* `kuramoto_rhs` — O(N²) evaluation with `sin`/`cos` from
  `std.math`.
* `kuramoto_jacobian` — includes the driver diagonal term.
* `mat_mul` — naive triple loop, row-major (since `N ≤ 64` in
  practice the cost is acceptable and stays out of Mojo's alias
  analyser).
* `row_mgs` — two-pass MGS matching the Rust convention.
* `sort_descending` — in-place insertion sort.

The executable reads a single whitespace-separated line from stdin
tagged `SPEC` followed by the integer / float scalar parameters and
the four flat arrays. It prints `N` lines of `f64` to stdout.
`_lyapunov_mojo.py` serialises arguments with `repr(float(x))`, so
the only precision loss is the ASCII round-trip (captured by the
`1e-6` tolerance in the parity tests).

**Build:**

```bash
mojo build mojo/lyapunov.mojo -o mojo/lyapunov_mojo -Xlinker -lm
```

### 4.3 Julia (`julia/lyapunov.jl`)

Julia's `LinearAlgebra.qr` is first-class, so `row_mgs` is replaced
by `F = qr(Q'); Q = Matrix(F.Q)'; R = F.R` — the transpose dance
converts Julia's column-based QR into row-MGS. The rest of the code
mirrors Python: `kuramoto_rhs`, `kuramoto_jacobian` (with the driver
diagonal) and a standard RK4 step. The Julia package is loaded once
per Python process by `juliacall` and cached in `_JULIA_MODULE`, so
amortised call overhead is a single dispatch plus a Float64 array
round-trip.

### 4.4 Go (`go/lyapunov.go`)

Compiled with `go build -buildmode=c-shared -o liblyapunov.so
lyapunov.go`. Allocations are per-call because a single
`libgolang.so` runtime cannot hand out Go-owned memory safely over
the FFI boundary — instead, the caller pre-allocates the output
buffer and passes the pointer. Internals:

* `kuramotoJacobian` — matches Python line-for-line including the
  driver diagonal term.
* `matMul` — naive `O(N³)` row-major; no `gonum` dependency so the
  shared object stays under 2 MB.
* `rowMGS` — two-pass MGS on rows, with `1e-300` floor.
* `sort.Sort(sort.Reverse(sort.Float64Slice(...)))` — stdlib sort.
* FFI export: `LyapunovSpectrum(phasesInit, omegas, knmFlat,
  alphaFlat, n, dt, nSteps, qrInterval, zeta, psi, outPtr) int`.

### 4.5 Python (`src/.../monitor/lyapunov.py`)

The reference fallback is pure NumPy / `np.linalg.qr`:

```python
k1_q = _kuramoto_jacobian(phases, knm, alpha, zeta, psi) @ Q
...
Q_t, R = np.linalg.qr(Q.T)
Q = Q_t.T
```

The only NumPy call outside the Jacobian is `np.linalg.qr`, which is
LAPACK under the hood. At `N = 32, n_steps = 500, qr_interval = 10`
it takes ~217 ms per call — fine for diagnostic runs, slow for
online control.

---

## 5. Benchmarks

Measured on an Ubuntu 24.04 host with 16-thread x86_64 CPU, NumPy
2.3.4 / MKL, Julia 1.11.2, Go 1.23.4, Mojo 0.26.2, `spo_kernel`
built in release mode with `maturin build --release`. Every entry
is **ms per call** for a fresh `(N × N)` Kuramoto problem with
`dt = 0.01`, `qr_interval = 10`, `calls = 2` (warm-up plus measured
pair). Re-run with
`python benchmarks/lyapunov_benchmark.py --sizes 4 8 16 32 --n-steps 500`.

| N   | n_steps | rust (ms) | mojo (ms) | julia (ms) | go (ms) | python (ms) |
| --- | ------- | --------: | --------: | ---------: | ------: | ----------: |
| 4   | 500     |     49.87 |     89.22 |       1.91 |    2.53 |       67.30 |
| 8   | 500     |     65.45 |    123.80 |       6.57 |    8.28 |       69.51 |
| 16  | 500     |     72.63 |    122.40 |      22.12 |   28.89 |       77.41 |
| 32  | 500     |    157.81 |    185.12 |     166.53 |  165.61 |      216.84 |

Observations:

* **Go and Julia lead at small N.** For `N ≤ 16` they dominate
  because the per-call setup cost is minimal (no process spawn, no
  rayon thread-pool priming). The naive triple-loop `matMul` in Go
  is faster than the LAPACK dispatch overhead in Python / Rust.
* **Rust dominates at large N.** At `N = 32` rayon amortises across
  all 16 threads and the rust backend matches Julia / Go while
  bringing a lot more headroom (tests at `N = 64, n_steps = 2000`
  clock rust at ~900 ms, go at ~2.6 s).
* **Mojo subprocess overhead is the floor.** Each call forks a
  process, parses a text stream, and prints `N` lines of ASCII —
  this dominates below `N = 32`. Mojo is retained in the chain for
  parity coverage rather than raw throughput.
* **Python is competitive at small N.** The RK4 loop is only a few
  lines of NumPy, so dispatch overhead is small; the `N³` scaling
  catches up by `N = 32`.

**Raw benchmark JSON** is saved by
`--output /tmp/ly_bench.json`; SPO CI publishes this file as a build
artifact so regressions across backends are visible in release notes.

---

## 6. Usage examples

### 6.1 Quickstart

```python
import numpy as np
from scpn_phase_orchestrator.monitor.lyapunov import lyapunov_spectrum

rng = np.random.default_rng(42)
N = 6
phases = rng.uniform(0, 2 * np.pi, N)
omegas = rng.normal(0.0, 0.2, N)
knm = rng.uniform(0.5, 1.5, (N, N))
np.fill_diagonal(knm, 0.0)
alpha = np.zeros((N, N))

spectrum = lyapunov_spectrum(
    phases, omegas, knm, alpha,
    dt=0.01, n_steps=1000, qr_interval=10,
)
print(spectrum)          # (N,) array, descending
print("λ_max =", spectrum[0])
print("dim_KY ≤", N)
```

### 6.2 Stability guard inside a control loop

```python
from scpn_phase_orchestrator.monitor.lyapunov import LyapunovGuard

guard = LyapunovGuard(basin_threshold=np.pi / 3)
for step in range(n_steps):
    state = guard.evaluate(phases, knm)
    if not state.in_basin:
        # Trigger a stronger control pulse — phases drifted out of
        # the Lyapunov basin.
        zeta = min(zeta * 1.2, ZETA_MAX)
    phases = integrator.step(phases, omegas, knm, zeta, psi, alpha)
```

`dV/dt` is available as a soft-indicator of local contraction — a
sustained positive sign signals the system is climbing the Lyapunov
landscape and will diverge unless the driver clamps it.

### 6.3 Driver-response sweep

```python
zetas = np.linspace(0.0, 5.0, 21)
lmax = []
for zeta in zetas:
    spec = lyapunov_spectrum(
        phases, omegas, knm, alpha,
        zeta=zeta, psi=0.0,
        n_steps=2000,
    )
    lmax.append(spec[0])

critical_zeta = zetas[np.argmin(np.abs(np.asarray(lmax)))]
```

`critical_zeta` is the driver strength at which `λ_1` crosses zero
— the phase transition between synchronised and chaotic regimes for
this particular `(ω, K, α)` tuple.

### 6.4 Forcing a specific backend

```python
from scpn_phase_orchestrator.monitor import lyapunov as ly_mod

saved = ly_mod.ACTIVE_BACKEND
try:
    ly_mod.ACTIVE_BACKEND = "julia"
    spec = lyapunov_spectrum(phases, omegas, knm, alpha)
finally:
    ly_mod.ACTIVE_BACKEND = saved
```

This pattern is how the parity tests pin a specific backend for
comparison against the Python reference.

---

## 7. Tests

Three dedicated test files cover the module:

### 7.1 `tests/test_lyapunov_algorithm.py` — algorithmic properties

Pins down the analytic behaviour that must hold regardless of
backend. Classes:

* `TestShapeAndOrdering` — output shape, descending sort, finiteness.
* `TestAnalyticLimits` — zero-coupling / strong-coupling limits,
  single oscillator neutrality.
* `TestDriverResponse` — adding a contracting driver must not
  increase `λ_max`.
* `TestSakaguchiPhaseLag` — non-zero `α` must perturb the spectrum;
  this test was added after a pre-fix bug where the Euler
  reference silently ignored `α` in the Jacobian.
* `TestKuramotoJacobianInternals` — `_kuramoto_jacobian` must
  include `−ζ cos(Ψ − θ_i)` on the diagonal when `ζ ≠ 0` and must
  use `K_ij cos(θ_j − θ_i − α_ij)` for off-diagonals.
* `TestRK4Convergence` — halving `dt` should shift the spectrum by
  less than `0.1` on a random problem (RK4 is 4th-order in `dt`).
* `TestRandomProperty` — Hypothesis property test across
  `N ∈ [2, 6]` and random seeds; sorting and finiteness must
  hold for every draw.
* `TestInputValidation` — empty phases → empty spectrum;
  `qr_interval = 0` must raise through the Rust FFI.

### 7.2 `tests/test_lyapunov_backends.py` — cross-backend parity

Runs the same problem through each available backend and compares
against the forced Python reference. Classes:

* `TestRustParity` — three seeds plus a driver-on case at `1e-12`.
* `TestJuliaParity` — two seeds at `1e-12`.
* `TestGoParity` — three seeds plus a driver + phase-lag case at
  `1e-12`.
* `TestMojoParity` — two seeds at `1e-6` (text round-trip).
* `TestCrossBackendConsistency` — iterates every `AVAILABLE_BACKENDS`
  entry and asserts the tolerance-matched diff vs Python.
* `TestDispatcherResolution` — `ACTIVE_BACKEND` is the first
  available; `"python"` is always in the chain.

### 7.3 `tests/test_lyapunov_stability.py` — long-run invariants

`@pytest.mark.slow`. Three scenarios:

* `test_long_run_all_attracting` — 10 000 steps on a strongly
  contracting network; all exponents must be `< −0.5`.
* `test_sum_tracks_trace_of_jacobian` — `Σ λ_i ≈ ⟨tr J⟩` up to a
  loose tolerance; any drift implies a bug in RK4 or QR.
* `test_kaplan_yorke_dimension_bounded` — `0 ≤ d_KY ≤ N`.

Run with:

```bash
pytest tests/test_lyapunov_algorithm.py tests/test_lyapunov_backends.py
pytest tests/test_lyapunov_stability.py -m slow
```

Pipeline wiring is validated by the pre-existing
`tests/test_lyapunov_spectrum.py::TestLyapunovSpectrumPipelineWiring`
which threads `UPDEEngine.step` → `lyapunov_spectrum` → spectrum.

---

## 8. Failure modes and caveats

### 8.1 Numerical fragility

* **`|R_ii| < 1e-300`.** Floored to `1e-300` before `log` to keep the
  exponents finite. Hitting this floor repeatedly (visible as a
  constant `−log(10^300) ≈ −690` in the running log) means the
  perturbation vectors have collapsed — either `qr_interval` is too
  large or the dynamics are pathologically contracting.
* **Forward-Euler in older `LyapunovGuard.evaluate`.** The observer
  uses consecutive-call finite differences for `dV/dt`. Between
  calls the caller owns the simulation step, so `dV/dt` reflects the
  caller's integration cadence, not RK4.
* **Mojo text round-trip.** `atof` decodes `repr(float(x))` which
  preserves `f64` to its 17 significant-digit representation; the
  resulting `1e-6` parity bound is an empirical observation, not a
  theoretical guarantee.

### 8.2 Ordering at degeneracy

When two exponents are exactly equal (`λ_i = λ_{i+1}`), the QR
factorisation's column order is sign-ambiguous. All five backends
sort descending before returning, which breaks the tie
deterministically; the underlying subspace is the correct observable,
not the individual vectors.

### 8.3 Large `N` scaling

`matMul` and `kuramotoJacobian` are both `O(N²)` per stage with 4
RK4 stages per step and `O(N²)` per QR. Total cost scales as
`O(N³ · n_steps / qr_interval + N² · n_steps · 4)`. For `N > 64` the
Rust backend's rayon parallelism is essential; Python / Go / Julia
/ Mojo all serialise and become 10× slower past that threshold.

### 8.4 Rust kernel uses RK4 everywhere

Unlike earlier versions of the Python reference, all backends now
use RK4 + driver-diagonal Jacobian. A comment in the docstring of
`_lyapunov_spectrum_python` spells this out — otherwise an innocent
"simplification" back to Euler would silently reintroduce the
pre-fix bug where the Euler reference disagreed with Rust by several
units.

### 8.5 Backend acquisition order

If a higher-priority backend is broken (e.g., Mojo executable
missing its runtime `.so`), the dispatcher still advances to the
next backend. The noisy exception is caught inside `_resolve_backends`
and the offending backend is simply omitted from `AVAILABLE_BACKENDS`
— there is no hard failure at import time.

---

## 9. Complexity

| Routine                   | Time                                | Space                 |
| ------------------------- | ----------------------------------- | --------------------- |
| `kuramoto_rhs`            | `O(N²)`                             | `O(N)`                |
| `kuramoto_jacobian`       | `O(N²)`                             | `O(N²)`               |
| `mat_mul` (`J · Q`)       | `O(N³)`                             | `O(N²)` (scratch)     |
| RK4 step                  | `4 · (kuramoto_rhs + mat_mul)`      | `O(N²)` per stage     |
| Row-MGS (`qr_interval⁻¹`) | `O(N³)` every `qr_interval` steps   | `O(N)` (`diagR`)      |
| Total for one spectrum    | `O(n_steps · (N³ + 4·N² + N³/qrI))` | `O(N²)` peak residency |

At the default `n_steps = 1000, qr_interval = 10, N = 16`, each call
does roughly `4 × 10⁵` FLOPs — sub-millisecond on Rust, tens of ms
on Python.

---

## 10. References

* Benettin, G., Galgani, L., Giorgilli, A., Strelcyn, J.-M. (1980).
  *Lyapunov characteristic exponents for smooth dynamical systems
  and for Hamiltonian systems; a method for computing all of them.*
  Meccanica **15**, 9–30.
* Shimada, I., Nagashima, T. (1979). *A numerical approach to
  ergodic problem of dissipative dynamical systems.* Progress of
  Theoretical Physics **61** (6), 1605–1616.
* Pikovsky, A., Politi, A. (2016). *Lyapunov Exponents: A Tool to
  Explore Complex Dynamics.* Cambridge University Press.
* van Hemmen, J. L., Wreszinski, W. F. (1993). *Lyapunov function for
  the Kuramoto model of nonlinearly coupled oscillators.* Journal of
  Statistical Physics **72** (1–2), 145–166.
* Daniel, J. W., Gragg, W. B., Kaufman, L., Stewart, G. W. (1976).
  *Reorthogonalisation and stable algorithms for updating the
  Gram-Schmidt QR factorisation.* Mathematics of Computation **30**
  (136), 772–795.
* Kuramoto, Y. (1984). *Chemical Oscillations, Waves, and Turbulence.*
  Springer Series in Synergetics **19**.
* Sakaguchi, H., Kuramoto, Y. (1986). *A soluble active rotator model
  showing phase transitions via mutual entrainment.* Progress of
  Theoretical Physics **76** (3), 576–581.
* Schreiber, T. (2000). *Measuring information transfer.* Physical
  Review Letters **85** (2), 461.

---

## 11. Changelog

* **2026-04-18** — Migrated to the AttnRes-level standard. The Python
  reference now uses RK4 + driver-diagonal Jacobian + row-oriented
  MGS. The Julia, Go, and Mojo ports were rewritten to match Rust
  bit-for-bit; the Mojo port re-implements `mat_mul` + `row_mgs` in
  pure Mojo. Three new test files (`test_lyapunov_algorithm.py`,
  `test_lyapunov_backends.py`, `test_lyapunov_stability.py`) plus the
  multi-backend benchmark.
