# Ordinal-Pattern Transition Entropy — Explosive-Sync Compute Core

The `monitor.opt_entropy` module turns a scalar observable into two
ordinal-dynamics quantities with a five-language backend chain:

* `ordinal_pattern_sequence` — the **Bandt–Pompe ordinal pattern** of every
  sliding window, encoded as the Lehmer code of its stable ascending argsort
  permutation, an integer in `[0, D! − 1]`.
* `transition_entropy` — the **normalised Shannon entropy of the
  consecutive-pattern transition distribution**, in `[0, 1]`.

Transition entropy collapses as a dynamical system regularises ahead of a
first-order (explosive) synchronisation onset, which is why it is the compute
core of the explosive-synchronisation early-warning monitor in
[`monitor.explosive_sync`](monitor_explosive_sync.md). This module is the
compute primitive; the monitor is the detection layer built on top.

---

## 1. Mathematical formalism

### 1.1 Ordinal patterns (Bandt–Pompe)

Given a scalar series `x[0..T−1]`, an embedding dimension `D ∈ [2, 7]`, and a
delay `τ ≥ 1`, the `m`-th delay window is

$$
w_m \;=\; \bigl(x_m,\; x_{m+τ},\; x_{m+2τ},\; \ldots,\; x_{m+(D-1)τ}\bigr),
\qquad m = 0, \ldots, M-1,
$$

with `M = T − (D − 1)·τ` windows. The window is sorted ascending; ties are
broken by sample index (the original Bandt–Pompe convention, so a flat window
is the identity pattern). The sorting permutation `π` — the list of original
positions in ascending-value order — is the **ordinal pattern**.

### 1.2 Lehmer encoding

`π` is encoded into a single integer in `[0, D! − 1]` by its Lehmer code:

$$
\mathrm{code}(π) \;=\; \sum_{i=0}^{D-1}
\Bigl(\,\bigl|\{\,j > i : π_j < π_i\,\}\bigr|\,\Bigr)\;(D-1-i)!\,.
$$

The encoding is a bijection: the strictly increasing window maps to `0`, the
strictly decreasing window to `D! − 1`, and the six orderings of three
distinct samples cover `{0, …, 5}` exactly once. `ordinal_pattern_sequence`
returns the length-`M` vector of these codes.

### 1.3 Transition distribution and entropy

Consecutive ordinal patterns form `M − 1` directed transitions. Each is packed
into one integer key `code_m · D! + code_{m+1}`; the keys are counted by
ascending key order into frequencies `c_1, \ldots, c_L` over the `L` distinct
observed transitions, with `\sum_\ell c_\ell = M − 1 = S`. The transition
entropy is the Shannon entropy of that distribution, normalised by the entropy
of `L` equiprobable transitions:

$$
H_T \;=\; \frac{-\sum_{\ell=1}^{L} p_\ell \ln p_\ell}{\ln L},
\qquad p_\ell = c_\ell / S \;\in\; [0, 1].
$$

`transition_entropy` returns `H_T`, defined as `0.0` when fewer than two
transitions exist (`M < 2`) or only a single transition type is observed
(`L < 2`, a perfectly predictable regime). Summation is performed sequentially
in ascending-key order so that every language backend accumulates the floating
sum in the identical order.

Interpretation:

| `H_T` | Regime |
|---|---|
| ≈ 0 | Near-deterministic transitions — regular / locked dynamics. |
| 0 < `H_T` < 1 | Structured transitions — partial order. |
| ≈ 1 | Uniform transition spread — disordered / stochastic dynamics. |

### 1.4 Why transitions rather than static patterns

The static permutation-entropy distribution `p(π)` discards temporal order; a
shuffled series shares it. The *transition* distribution `p(π_i → π_j)`
encodes the dynamical rule. As a network approaches an explosive
synchronisation, each node's local dynamics become more predictable: the
transition graph concentrates onto a few edges and `H_T` drops sharply, often
before the macroscopic order parameter moves and before
variance / autocorrelation critical-slowing-down indicators react.

### 1.5 Why scale invariance matters

Ordinal patterns depend only on the *relative order* of samples, so
`transition_entropy` is invariant under any strictly increasing affine
rescaling `a·x + b` with `a > 0`. A node observable does not need calibration
or detrending of its absolute scale before entering the monitor — only its
ordinal dynamics are read.

---

## 2. Python API

```python
from scpn_phase_orchestrator.monitor.opt_entropy import (
    ordinal_pattern_sequence,
    transition_entropy,
    ACTIVE_BACKEND,
    AVAILABLE_BACKENDS,
)

codes = ordinal_pattern_sequence(series, dimension=3, delay=1)  # int64 (M,)
value = transition_entropy(series, dimension=3, delay=1)        # float in [0, 1]
```

Key parameters:

* `series`: finite real one-dimensional array of shape `(T,)`. Boolean
  aliases, numeric-string aliases, complex samples (including object-dtype
  complex aliases), non-finite values, and multi-dimensional inputs are rejected
  before backend dispatch.
* `dimension`: embedding dimension `D`, an integer in `[2, 7]`. Larger `D`
  resolves finer ordinal structure but needs `D!`-times more data to populate
  the transition graph; `D = 3` (default) suits the short windows the monitor
  uses.
* `delay`: positive embedding delay `τ` (default `1`). Larger `τ` probes
  slower timescales.
* A series shorter than one window returns an empty code vector and
  `transition_entropy` returns `0.0`.

---

## 3. Multi-backend fallback chain

```python
>>> from scpn_phase_orchestrator.monitor.opt_entropy import (
...     ACTIVE_BACKEND, AVAILABLE_BACKENDS,
... )
>>> ACTIVE_BACKEND, AVAILABLE_BACKENDS
('rust', ['rust', 'mojo', 'julia', 'go', 'python'])
```

| Position | Backend | Build | Implementation notes |
|---|---|---|---|
| 1 | Rust | `maturin develop --release -m spo-kernel/crates/spo-ffi/Cargo.toml` | Canonical fast path; PyO3 exports `ordinal_pattern_sequence`, `transition_entropy`. |
| 2 | Mojo | `mojo build mojo/opt_entropy.mojo -o mojo/opt_entropy_mojo -Xlinker -lm` | Text-stdin subprocess bridge (`OPS` / `OTE` dispatch); integer keys sorted by insertion sort. |
| 3 | Julia | `juliacall` + `julia/opt_entropy.jl` | One-based argsort/Lehmer with identical inversion count. |
| 4 | Go | `go build -buildmode=c-shared -o go/libopt_entropy.so go/opt_entropy.go` | ctypes bridge; `OrdinalPatternSequence` + `TransitionEntropy` C-ABI exports. |
| 5 | Python | always present | NumPy / list reference — the implementation every compiled backend mirrors. |

### 3.1 Parity budget

Ordinal pattern codes are integers and must match **exactly** across every
backend (no tolerance). The scalar transition entropy matches the Python
reference within:

| Backend | `ordinal_pattern_sequence` | `transition_entropy` |
|---|---|---|
| Rust | exact (0) | ≤ 1e-12 |
| Julia | exact (0) | ≤ 1e-12 |
| Go | exact (0) | ≤ 1e-12 |
| Mojo | exact (0) | ≤ 1e-9 (text round-trip) |
| Python | exact (reference) | 0 (reference) |

The Mojo budget is wider because each transition probability is re-parsed from
a decimal string before the `log` summation, flooring the scalar at ~1e-10.
The compiled-bridge parity tests assert the tight `1e-12` bound directly; the
public dispatcher boundary absorbs the Mojo floor with a `1e-9` check
(`_DISPATCH_ENTROPY_TOLERANCE`).

Every backend output is revalidated at the Python boundary before it leaves the
monitor: code vectors must reject boolean, numeric-string, and complex aliases,
remain finite and integer-valued, keep the correct length, and stay bounded in
`[0, D! − 1]`; the scalar must be a finite real in `[0, 1]` and match the exact
reference within the backend tolerance. Malformed or physics-divergent payloads
fall back to the NumPy reference instead of entering the monitor.
The direct Go, Julia, and Mojo bridge validators enforce the same series-input
and ordinal-code output contract before optional runtime loading and after
backend execution, so direct accelerator callers do not bypass the public
numeric-domain guard.

---

## 4. Measured benchmarks

Output from
`PYTHONPATH=src python benchmarks/opt_entropy_benchmark.py --sizes 256 1024 4096 --calls 20`
on the development host (x86_64, single thread, release Rust wheel, juliacall
bootstrapped, Mojo and Go built). Each measured call runs the **public
dispatcher**, which always computes the Python reference for verification, so
these figures price the safety boundary, not the bare kernel:

| N | rust (ms) | mojo (ms) | julia (ms) | go (ms) | python (ms) |
|---|---:|---:|---:|---:|---:|
| 256 | 3.32 | 203.60 | 7.31 | 9.24 | 3.29 |
| 1024 | 11.11 | 237.60 | 23.50 | 22.77 | 11.46 |
| 4096 | 53.60 | 280.76 | 68.57 | 81.49 | 38.10 |

Observations (honest, not aspirational):

* **The reference check dominates at these sizes.** Because the dispatcher
  recomputes the Python reference on every call to verify the backend, the
  `rust` row includes the Python cost plus the Rust round-trip; in this safety
  boundary snapshot, pure Python remains competitive and is the fastest row at
  `N = 4096`. Callers who have already validated a backend and want raw
  throughput should pin `ACTIVE_BACKEND` and bypass per-call reference
  recomputation in a tight loop.
* **Mojo is ruled out of hot loops:** the subprocess spawn and text protocol
  floor each call at roughly 200 ms on this host. It earns its second slot as a
  correctness cross-check, not as a production hot path.
* **Go and Julia track each other** within a factor of two; Julia's first call
  pays a JIT warm-up not shown here (one warm-up call precedes timing).

Reproduce the deterministic parity gate with:

```bash
python benchmarks/opt_entropy_benchmark.py --parity-gate --sizes 256 --calls 1
```

The gate records every declared backend slot in canonical order, compares each
available backend against the forced Python reference for both the exact
ordinal codes and the scalar entropy, and requires the Python reference, one
record per declared backend, unit-interval entropy, and tolerance-bounded
agreement. Unavailable toolchains stay as explicit records with a reason rather
than disappearing.

---

## 5. Mathematical invariants (tested)

* **Bounds.** `transition_entropy ∈ [0, 1]` over random series at every
  dimension (`test_transition_entropy_bounded`, slow).
* **Determinism limits.** A constant or monotone series has a single
  transition type, so `transition_entropy = 0`
  (`test_constant_series_zero_entropy`, `test_monotone_series_is_zero`).
* **Disorder limit.** White noise visits transitions near-uniformly, so
  `transition_entropy > 0.95` for a long sample (`test_white_noise_high_entropy`).
* **Regularisation lowers entropy.** A mostly periodic signal has lower
  transition entropy than the pure noise it was mixed from
  (`test_regularisation_lowers_entropy`) — the property the early-warning
  monitor relies on.
* **Affine scale invariance.** `transition_entropy(a·x + b) =
  transition_entropy(x)` for `a > 0`, and the code sequences are identical
  (`test_amplitude_scale_invariance`).
* **Lehmer bijection.** The six orderings of three distinct values map onto
  `{0, …, 5}` exactly once (`test_all_orderings_form_a_bijection_to_factorial_range`).

---

## 6. Usage patterns

### 6.1 Single-series complexity trace

```python
from scpn_phase_orchestrator.monitor.opt_entropy import transition_entropy

trace = [transition_entropy(window, dimension=3, delay=1) for window in windows]
# A sustained drop in `trace` flags a regularising (pre-locking) regime.
```

### 6.2 Backend switch for cross-checking

```python
from scpn_phase_orchestrator.monitor import opt_entropy as oe

saved = oe.ACTIVE_BACKEND
try:
    oe.ACTIVE_BACKEND = "julia"
    value = transition_entropy(series)
finally:
    oe.ACTIVE_BACKEND = saved
```

The dispatcher reads `ACTIVE_BACKEND` at call time, so this pattern is safe to
restore in a `finally` block.

---

## 7. Pipeline position

`opt_entropy` is a **passive compute primitive** — it reads a scalar series and
returns codes or a scalar with no side effects. Its sole downstream consumer in
the SPO tree is the explosive-synchronisation monitor, which calls
`transition_entropy` per node per sliding window.

```
                         ┌─────────────────────────┐
   x(t) (one node) ─────▶│  ordinal_pattern_sequence│────▶ codes ∈ [0, D!-1]
                         └─────────────────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
   x(t) (one node) ─────▶│    transition_entropy    │────▶ H_T ∈ [0, 1]
                         └─────────────────────────┘
                                      │
                                      ▼
                         monitor.explosive_sync.explosive_sync_warning
```

---

## 8. Implementation cross-reference

| File | Role |
|---|---|
| `src/scpn_phase_orchestrator/monitor/opt_entropy.py` | Dispatcher + NumPy reference |
| `src/scpn_phase_orchestrator/monitor/explosive_sync.py` | Early-warning monitor built on this primitive |
| `spo-kernel/crates/spo-engine/src/opt_entropy.rs` | Rust kernel |
| `spo-kernel/crates/spo-ffi/src/lib.rs` | PyO3 bindings `ordinal_pattern_sequence`, `transition_entropy` |
| `julia/opt_entropy.jl` | Julia `OptEntropy` module |
| `go/opt_entropy.go` | Go c-shared library |
| `mojo/opt_entropy.mojo` | Mojo text-stdin executable |
| `src/scpn_phase_orchestrator/experimental/accelerators/monitor/_opt_entropy_validation.py` | Shared backend validation + reference |
| `src/scpn_phase_orchestrator/experimental/accelerators/monitor/_opt_entropy_{go,julia,mojo}.py` | Native bridges |
| `tests/test_opt_entropy.py` | Algorithm, dispatch, boundary-hardening tests |
| `tests/test_opt_entropy_backends.py` | Per-backend parity tests |
| `tests/test_opt_entropy_stability.py` | Mathematical invariants (slow) |
| `tests/test_opt_entropy_validation_guards.py` | Backend validation guards |
| `tests/test_opt_entropy_benchmark.py` | Polyglot parity-gate contract |
| `benchmarks/opt_entropy_benchmark.py` | Multi-backend wall-clock harness |

---

## 9. References

* Bandt, C. & Pompe, B. 2002, *Phys. Rev. Lett.* 88, 174102 — "Permutation
  entropy: a natural complexity measure for time series."
* McCullough, M., Small, M., Stemler, T. & Iu, H. H.-C. 2015, *Chaos* 25,
  053101 — ordinal partition (pattern transition) networks for continuous
  dynamics.
* Gómez-Gardeñes, J., Gómez, S., Arenas, A. & Moreno, Y. 2011, *Phys. Rev.
  Lett.* 106, 128701 — explosive synchronisation as a first-order transition.

---

## 10. API reference

::: scpn_phase_orchestrator.monitor.opt_entropy
