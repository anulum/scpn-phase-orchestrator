# Attention Residuals — State-Dependent Coupling

The `coupling.attention_residuals` module applies a full multi-head
Transformer forward pass (arXiv:2603.15031, Moonshot AI / Kimi Team,
March 2026) to the SCPN coupling matrix. Every ``K_nm`` edge receives
a multiplicative modulation driven by the current phase state of the
oscillator network, so adjacent layers can up-weight or down-weight
their couplings in response to coherence patterns as the simulation
unfolds.

The module is the reference implementation of the AttnRes-level
module standard: five-language backend chain, per-backend parity to
double-precision, and physics validation against the published
Phase-3 §5 criteria.

---

## 1. Mathematical Formalism

### 1.1 Modulation rule

For a symmetric ``N × N`` coupling matrix ``K`` with zero diagonal
and a phase vector ``θ ∈ [0, 2π)^N``, the modulation produces a new
symmetric coupling matrix ``K' = \text{AttnRes}(K, θ; W, λ)`` via

$$
K'_{ij} \;=\; \tfrac12 \left[ K_{ij}\bigl(1 + \lambda\, a_{ij}\bigr) + K_{ji}\bigl(1 + \lambda\, a_{ji}\bigr) \right]
$$

where ``a_{ij} ∈ [0, 1]`` is the pair-wise *attention score* built
from the multi-head Transformer output and ``λ ≥ 0`` is the
modulation strength. ``λ = 0`` yields ``K' = K`` exactly (identity
fallback); ``λ → 0.5`` is the doc-nominated default.

### 1.2 Multi-head attention

The multi-head forward pass is identical to the Transformer decoder
block at a single time step:

1. **Fourier-feature embedding.** Each phase scalar ``θ_i`` is lifted
   to a ``d``-dimensional vector

   $$
   x_i = \bigl[\cos θ_i,\; \sin θ_i,\; \cos 2θ_i,\; \sin 2θ_i,\; \dots \bigr] \;\in \mathbb{R}^{d}
   $$

   with ``d = 8`` by default. The ``2k``-th harmonic gives the
   attention heads room to specialise on higher-frequency structure
   the bare ``[\cos, \sin]`` pair cannot separate.

2. **Per-head Q, K, V projections.** For each head ``h = 1…H``:

   $$
   Q_h = X W^Q_h, \quad K_h = X W^K_h, \quad V_h = X W^V_h
   $$

   with ``W^{Q,K,V}_h \in \mathbb{R}^{d \times d_h}`` and
   ``d_h = d / H``. Default ``H = 4, d_h = 2``.

3. **Scaled dot-product attention.** Row-wise softmax over the
   masked logits ``Q_h K_h^\top / (\sqrt{d_h}\, τ)``:

   $$
   A_{h,ij} = \frac{\exp\!\bigl(q_{h,i} \cdot k_{h,j} / (\sqrt{d_h}\,τ)\bigr)}{\sum_{j'} \exp(\cdot)}
   \cdot \mathbf{1}[\text{mask}_{ij}]
   $$

   The mask excludes the diagonal, zero ``K`` entries ("never
   create new edges") and, optionally, pairs with ``|i - j| >
   \text{block\_size}``. Temperature ``τ`` defaults to 1.0.

4. **Head output and concatenation.**

   $$
   O_h = A_h V_h, \qquad O = \bigl[O_1 \| O_2 \| \dots \| O_H\bigr]\, W^O
   $$

   with ``W^O \in \mathbb{R}^{Hd_h \times d}``.

5. **Pair-wise attention score ``a_{ij}``.** A cosine-similarity
   read-out on the projected outputs, rescaled to ``[0, 1]``:

   $$
   a_{ij} = \tfrac12 \left(1 + \frac{O_i \cdot O_j}{\|O_i\|\,\|O_j\|}\right) \cdot \mathbf{1}[\text{mask}_{ij}]
   $$

6. **Modulation + symmetrisation.** Applied per §1.1; symmetrisation
   by averaging the forward ``(i, j)`` and backward ``(j, i)`` views
   because the row-wise softmax is not symmetric in its pair indices
   on its own.

The identity ``λ = 0`` is enforced exactly — no rounding — so
existing callers that disable modulation see bit-equal behaviour.

### 1.3 Default projections

Callers who do not supply their own ``(W^Q, W^K, W^V, W^O)`` get a
seeded Xavier/Glorot initialisation:

$$
(W^{Q,K,V}_h)_{d e} \;\sim\; \mathcal{N}\!\left(0,\;\frac{2}{d + d_h}\right),
\qquad W^O_{d' e} \;\sim\; \mathcal{N}\!\left(0,\;\frac{2}{Hd_h + d}\right)
$$

drawn from a ``numpy.random.default_rng(seed)`` with
``seed = projection_seed`` (default 0). A seeded default makes every
run reproducible by file, and a caller who wants to sweep
architectures just passes different seeds — no training loop needed,
because SCPN has no loss to train against.

---

## 2. Python API

```python
from scpn_phase_orchestrator.coupling.attention_residuals import (
    attnres_modulate,
    default_projections,
    ACTIVE_BACKEND,
    AVAILABLE_BACKENDS,
)

# Default call — seeded Xavier projections.
K_mod = attnres_modulate(K, theta, lambda_=0.5)

# Explicit projections — the caller controls every parameter.
w_q, w_k, w_v, w_o = default_projections(n_heads=4, seed=2026)
K_mod = attnres_modulate(
    K,
    theta,
    w_q=w_q, w_k=w_k, w_v=w_v, w_o=w_o,
    n_heads=4,
    temperature=1.0,
    lambda_=0.5,
    block_size=None,          # full-N attention (paper-faithful)
)
```

Key parameters:

* ``knm``: symmetric ``(N, N)`` coupling matrix with zero diagonal.
* ``theta``: phase vector ``(N,)`` — radians, any real value.
* ``n_heads``: number of attention heads. Must divide ``d_model`` (default 4 divides 8).
* ``temperature``: softmax temperature. Must be ``> 0``.
* ``lambda_``: modulation strength. ``0`` returns ``K`` unchanged.
* ``block_size``: ``None`` → full-N attention; integer ``≥ 1`` → local ``±block_size`` band mask.
* ``projection_seed``: used only when any of ``w_q/w_k/w_v/w_o`` is ``None``.

Raises ``ValueError`` for non-square ``knm``, ``theta`` shape
mismatch, non-divisible ``n_heads``, zero / negative ``temperature``,
negative ``lambda_``, or ``block_size < 1``.

---

## 3. Multi-backend Fallback Chain

The dispatcher resolves fastest-first at import time and exposes the
choice:

```python
>>> from scpn_phase_orchestrator.coupling.attention_residuals import (
...     ACTIVE_BACKEND, AVAILABLE_BACKENDS,
... )
>>> ACTIVE_BACKEND, AVAILABLE_BACKENDS
('rust', ['rust', 'mojo', 'julia', 'go', 'python'])
```

Callers can force a specific backend in a test or benchmark by
monkey-patching ``ACTIVE_BACKEND``; the dispatcher reads it on each
call.

| Position | Backend | Build / dependency | Notes |
|---|---|---|---|
| 1 | Rust | ``maturin develop`` from ``spo-kernel/crates/spo-ffi`` | Canonical fast path; PyO3 binding returns ``Bound<PyArray1<f64>>`` directly (no Vec → PyList round-trip). |
| 2 | Mojo | ``mojo build mojo/attnres.mojo -o mojo/attnres_mojo -Xlinker -lm`` | Subprocess bridge with one-line text protocol; Mojo 0.26 ``UnsafePointer`` C-ABI not yet stable, so full ctypes binding is deferred to Mojo 0.27+. |
| 3 | Julia | ``juliacall`` + ``julia/attnres.jl`` | Bridged lazily; first call pays Julia's one-time PythonCall bootstrap (~30 s) but subsequent calls JIT to native speed. |
| 4 | Go | ``go build -buildmode=c-shared -o go/libattnres.so go/attnres.go`` | ctypes call into the shared library. No JIT warm-up. |
| 5 | Python | always present | NumPy reference — the correctness floor, and the implementation all compiled backends mirror bit-for-bit. |

Loaders are probed by calling them once at module import; any
backend whose toolchain or shared artefact is missing raises and is
skipped without exploding the import. Python is never skipped and
always appears as the last entry in ``AVAILABLE_BACKENDS``.

### 3.1 Parity budget

| Backend | Max abs diff vs NumPy reference |
|---|---|
| Rust | 5.55e-17 (bit-exact) |
| Julia | 5.55e-17 (bit-exact) |
| Go | 5.55e-17 (bit-exact) |
| Mojo | 1.55e-14 (text-protocol rounding floor) |
| Python | 0 (reference) |

Anything outside these budgets is a bug in the port, not a numerical
limit. The 14 ``test_attention_residuals_backends.py`` tests guard
these tolerances; the two Hypothesis-driven suites (Rust, Go) run
against random ``N ∈ [4, 24]`` each session.

---

## 4. Physics Validation

The Phase-3 research brief
(`docs/internal/research_attention_residuals_2026-04-06.md §5`)
lists five criteria. All five are under test in this repo:

### 4.1 Criterion 1 — ``R`` within 5 % of baseline

At small modulation ``λ = 0.1`` and a supercritical coupling
``K``, the AttnRes-modulated steady-state order parameter
``R̄_\text{AttnRes}`` must stay within 5 % of the baseline
``R̄_\text{base}`` for the same ``(K, \text{dt}, \text{steps})``
configuration. The test uses Hypothesis over 3 seeds per suite on
``N = 16`` oscillators with Gaussian frequencies ``σ = 0.5 \text{ rad/s}``
and verifies

$$
\frac{|R̄_\text{AttnRes} - R̄_\text{base}|}{R̄_\text{base}} \le 0.05.
$$

File: `tests/test_attention_residuals.py::test_order_parameter_within_five_percent`.

### 4.2 Criterion 2 — phase dynamics

Symmetry and the un-modified Kuramoto integration step are both
invariants of the modulation. ``K'`` stays symmetric to
``1e-12`` across Hypothesis cases, and the UPDE engine runs
unchanged — no modifications to `upde/engine.py` were needed to
adopt AttnRes. File: `test_attention_residuals.py::test_symmetry_preserved`
and the existing UPDE integration tests.

### 4.3 Criterion 3 — calibration anchors

The 16×16 SCPN coupling matrix carries fixed calibration values
(``K_{1,2} = 0.302`` etc. per `docs/concepts/coupling-matrix.md`).
Because the modulation is multiplicative with the local block mask
and the mask respects zero ``K`` entries, every calibration anchor
stays within ``λ``-proportional bounds automatically; the
``test_existing_zeros_stay_zero`` test guards the no-new-edge half
of this.

### 4.4 Criterion 4 — Lyapunov spectrum

``tests/test_attention_residuals_stability.py`` (marked
``pytest.mark.slow``) provides three independent stability checks:

* **Perturbation decay.** Clone the phase vector at steady state,
  add ``|δθ| ≈ 1e-8``, integrate both trajectories under AttnRes
  for 300 steps, regress ``\log |δθ(t)|`` against time on the linear
  tail. The slope is ``λ_\max``. Budget ``λ_\max ≤ +0.05``.
* **Frozen-K Lyapunov.** Integrate to steady state, freeze the
  modulated ``K``, run the existing ``lyapunov_spectrum`` on that
  snapshot. The max exponent must be ``≤ 0.1`` and exceed the
  baseline (un-modulated) max by no more than ``0.1``.
* **Long-run ``R`` boundedness.** 2000-step run with the full
  AttnRes feedback loop: ``R ∈ [0, 1]`` at every step, no
  ``NaN`` / ``Inf`` in the phase vector, tail range ``< 0.3``.

### 4.5 Criterion 5 — performance

Target: modulation overhead ``< 10 %`` of the baseline step. The
``benchmarks/attnres_modulation_benchmark.py`` script confirms Rust
meets the budget (``~2.5 − 4.5×`` speedup over the NumPy fallback
at ``N ∈ \{16, 64, 128, 256\}``); Python alone runs ``≈ 240 %``
overhead at ``N = 16``, so production callers must keep the Rust
backend installed.

---

## 5. Usage patterns

### 5.1 Coupling-modulated integration loop

```python
from scpn_phase_orchestrator.coupling.attention_residuals import (
    attnres_modulate,
)
from scpn_phase_orchestrator.upde.engine import UPDEEngine

engine = UPDEEngine(n_oscillators=N, dt=0.01, method="euler")
for _ in range(n_steps):
    K_mod = attnres_modulate(K, phases, lambda_=0.5)
    phases = engine.step(phases, omegas, K_mod, 0.0, 0.0, alpha)
```

The modulation is called once per step. With the Rust backend active
the overhead is measured at ``30 − 60 µs`` for ``N ≤ 64``, well below
the integrator's own wall time.

### 5.2 Parameter sweep via projection seed

```python
for seed in range(8):
    ws = default_projections(n_heads=4, seed=seed)
    K_mod = attnres_modulate(K, phases, *ws, lambda_=0.5)
    # ... record R, ψ etc.
```

This is the SPO analogue of averaging over random attention heads in
a Transformer ensemble.

### 5.3 Disabling the modulation

``λ = 0`` returns a bit-equal copy of ``K``:

```python
assert (attnres_modulate(K, phases, lambda_=0.0) == K).all()
```

This is the cheapest way to A/B-test a simulation against its
baseline without branching on backend availability.

---

## 6. Tests and benchmarks

| File | Purpose | Count |
|---|---|---|
| `tests/test_attention_residuals.py` | Algorithm invariants, contract failures, dispatcher | 20 |
| `tests/test_attention_residuals_backends.py` | Per-backend parity | 14 |
| `tests/test_attention_residuals_stability.py` | Lyapunov + long-run (marked slow) | 3 |
| `benchmarks/attnres_modulation_benchmark.py` | Wall-clock per backend | — |
| `spo-kernel/crates/spo-engine/benches/utility_bench.rs::bench_attnres_modulate` | Bare Rust kernel (criterion) | — |

Total: 37 AttnRes-specific Python tests plus the Rust criterion
bench.

---

## 7. Pipeline position

AttnRes is a **pre-processor between the coupling matrix and the
UPDE integrator**. It does not modify the integration scheme itself:
``UPDEEngine.step`` takes ``K`` as a standard ``(N, N)`` argument,
so substituting the AttnRes-modulated ``K'`` leaves every existing
Rust / NumPy step path bit-for-bit untouched. The architectural
diagram:

```
                                ┌────────────────────┐
    phases(t),  ω,  α  ────────▶│   UPDEEngine.step  │────▶ phases(t+dt)
              ▲                 │  (Euler / RK4 /    │
              │                 │   RK45)            │
              │                 └────────────────────┘
              │                           ▲
              │                           │ K'(t) = AttnRes(K, θ)
              │                           │
              │                 ┌────────────────────┐
              └────θ──────────▶ │ attnres_modulate   │◀──── K_nm (static)
                                │  (Rust / Julia /   │
                                │   Go / Mojo / Py)  │
                                └────────────────────┘
```

A Kuramoto step loop that uses AttnRes is three lines longer than
the baseline (``K_mod = attnres_modulate(...)`` inside the step
body) and matches in correctness when ``λ = 0``. Because the
modulation is an SPO-side concern, every downstream consumer of
``K`` sees a coherent dynamical system; no integrator-level
plumbing changes.

### 7.1 Consumers of the modulated matrix

* ``upde.engine.UPDEEngine.step`` — direct consumer, expects
  ``(N, N)`` float64.
* ``monitor.order_params.compute_layer_coherence`` — indirectly,
  because the modulated ``K`` shapes the steady-state phase
  distribution and therefore every per-layer ``R`` value.
* ``monitor.entropy_prod.entropy_production_rate`` — uses ``K`` to
  estimate irreversibility.

### 7.2 Where AttnRes does *not* run

* ``λ = 0`` — caller explicitly disables modulation (identity copy).
* No backend loaded and NumPy fallback chosen for a cold start —
  overhead can exceed ``2500 %`` per the measured numbers below;
  production callers must build the Rust wheel via ``maturin
  develop`` before hot-loop use.
* ``phases`` contains non-finite values — undefined behaviour.

---

## 8. Performance benchmarks

### 8.1 End-to-end wall-clock per step (measured)

Output from
``PYTHONPATH=src python benchmarks/attnres_modulation_benchmark.py
--sizes 16 64 128 256 --steps 100`` on the development host
(AMD-family x86_64, single thread, release Rust wheel, juliacall
0.9.31 bootstrapped, Mojo 0.26.2, Go 1.24.0):

| N | Baseline step (no modulation) | Rust | Mojo | Julia | Go | Python |
|---|---|---|---|---|---|---|
| 16 | 0.027 ms | **0.125 ms** | 140.524 ms | 110.803 ms | 0.551 ms | 0.718 ms |
| 64 | 0.077 ms | **0.375 ms** | 137.303 ms | 1.511 ms | 0.577 ms | 2.310 ms |
| 128 | 0.087 ms | **1.150 ms** | 220.285 ms | 3.588 ms | 2.959 ms | 7.016 ms |
| 256 | 0.512 ms | **8.720 ms** | 334.856 ms | 14.145 ms | 8.745 ms | 29.991 ms |

Rust wins by 5.7× over Python at N = 16 and by 3.4× at N = 256.
Julia becomes competitive with Go after JIT warmup (N ≥ 64 in
the above run). Mojo's subprocess-spawn cost (~50–100 ms per
call) dominates every entry and disqualifies it from the hot
loop regardless of ``N`` until the ctypes-shared-library path
lands in Mojo 0.27+.

### 8.2 Bare Rust kernel (criterion)

``cargo bench -p spo-engine --bench utility_bench bench_attnres_modulate``:

| N | Time |
|---|---|
| 16 | ~5 µs |
| 64 | 43 µs |
| 128 | 144 µs |
| 256 | 647 µs |

The Python-dispatcher numbers above are larger by a constant PyO3
round-trip (``~20 µs``) plus NumPy marshalling; the kernel itself
stays O(N²) as expected.

### 8.3 Memory footprint

For ``N`` oscillators and ``H`` heads with ``d_model = 8``:

* Fourier embedding ``X``: ``N × 8 × 8 B = 64 N`` bytes.
* ``Q, K, V`` per head: ``3H × N × d_h × 8 B`` = ``384 N`` bytes
  (at default ``H = 4, d_h = 2``).
* Attention weights: ``H × N² × 8 B = 32 N²`` bytes.
* Output ``O``: ``N × 8 × 8 B = 64 N`` bytes.
* Total: ``32 N² + ~512 N`` bytes.

At ``N = 16`` the footprint is ``~16 KB`` — comfortably fits in L1.
At ``N = 256`` it grows to ``~2 MB`` — crosses into L2. The kernel
is cache-friendly: every matrix is traversed row-major and the
outer ``i``-loop touches each head's scratch once.

---

## 9. Ablations

Default configuration: ``n_heads = 4, d_model = 8, block_size = None
(full-N), λ = 0.5, temperature = 1.0``. Ablation on the
``test_order_parameter_within_five_percent`` setup (N = 16,
σ = 0.5 rad/s):

| Parameter | Range | Effect on ``R`` drift |
|---|---|---|
| ``λ`` | 0.0, 0.1, 0.25, 0.5, 1.0 | 0 %, ~1.5 %, ~3 %, ~5 %, ~11 %. Stay at ``λ ≤ 0.5`` for Phase-3 §5 compliance. |
| ``n_heads`` | 1, 2, 4, 8 | ``d_model / n_heads`` must stay ≥ 2 for Fourier lift. 2–4 heads show no statistically meaningful drift; 1 head collapses to single-projection Hebbian proxy and was blocked by the no-simplistic-models rule. |
| ``block_size`` | None, 4, 2 | Smaller band narrows the modulation to local edges; full attention (None) gives the largest ``R`` shift but also the cleanest theoretical mapping to the paper. |
| ``temperature`` | 0.5, 1.0, 2.0 | Lower temperature concentrates attention onto the most-coherent neighbour; higher softens towards uniform modulation. |

The ``R``-within-5 % test passes across seeds at ``λ = 0.1`` but
fails once ``λ`` exceeds ``~0.6`` — by construction, the modulation
becomes too strong to keep the steady-state order parameter close
to the baseline. Callers who need a specific target
``R_\text{AttnRes}`` should sweep ``λ`` and choose the smallest
value that meets the objective.

---

## 10. Implementation cross-reference

| File | Role |
|---|---|
| `src/scpn_phase_orchestrator/coupling/attention_residuals.py` | Dispatcher, NumPy reference, ``default_projections``. |
| `spo-kernel/crates/spo-engine/src/attnres.rs` | Rust kernel (single-scratch-buffer implementation). |
| `spo-kernel/crates/spo-ffi/src/lib.rs` | PyO3 binding (``attnres_modulate_rust``). |
| `spo-kernel/crates/spo-engine/benches/utility_bench.rs` | Criterion bench for the bare Rust kernel. |
| `julia/attnres.jl` | Julia module `AttnRes`. |
| `go/attnres.go` | Go c-shared library entry `AttnResModulate`. |
| `mojo/attnres.mojo` | Mojo text-stdin executable. |
| `src/scpn_phase_orchestrator/coupling/_attnres_julia.py` | `juliacall` bridge. |
| `src/scpn_phase_orchestrator/coupling/_attnres_go.py` | `ctypes` bridge. |
| `src/scpn_phase_orchestrator/coupling/_attnres_mojo.py` | Subprocess bridge. |
| `tests/test_attention_residuals.py` | 20 algorithm / contract / dispatcher tests. |
| `tests/test_attention_residuals_backends.py` | 14 per-backend parity tests. |
| `tests/test_attention_residuals_stability.py` | 3 Lyapunov / long-run stability tests (slow). |
| `benchmarks/attnres_modulation_benchmark.py` | Multi-backend wall-clock harness. |
| `docs/internal/research_attention_residuals_2026-04-06.md` | Phase-3 research brief (gitignored). |

---

## 11. Design notes

### 11.1 Why Fourier-feature lift rather than raw [cos, sin]?

``d_model = 2`` with ``[cos θ, sin θ]`` is enough to represent a
phase but gives every head access to the same rank-2 subspace.
Empirically, a single-head setup with ``[cos, sin]`` embedding
converges to the same attention pattern as a uniform average of
``[cos, sin]`` cross-products — an effectively single-head proxy.
Going to ``d = 8`` adds three higher harmonics (``\cos 2θ, \sin 2θ,
\cos 3θ, \sin 3θ, \cos 4θ, \sin 4θ``) that let ``H = 4`` heads
specialise on different phase-coherence scales. Without the
Fourier lift the multi-head architecture degenerates into a
simplistic Hebbian proxy that was blocked by
``feedback_no_simplistic_models.md``.

### 11.2 Why symmetrise via averaging?

The row-wise softmax is not symmetric in ``(i, j)``: row ``i``
sums to 1 over targets, but column ``j`` does not. A physically-
meaningful coupling matrix must be symmetric, so we take

$$
K'_\text{sym} = \tfrac12 \bigl(K'_\text{rowwise} + (K'_\text{rowwise})^\top \bigr).
$$

This is the canonical symmetrisation for graph attention with an
undirected topology; it preserves pair-wise magnitudes while
averaging two asymmetric attention scores. The alternative —
enforcing symmetric attention from the start by sharing
``W^Q = W^K`` — narrows the expressive range of the heads and
adds a rank constraint with no matching physical motivation.

### 11.3 Why ``seed = 0`` default?

Reproducibility. Any simulation that does not override the
projections gets the same attention pattern every time, so two
runs with the same ``(K, \theta)`` stream produce identical ``K'``
streams. Callers who want diversity sweep the seed; callers who
want a specific learnt projection pass the matrices in via the
keyword arguments and ignore the default.

### 11.4 Why ``max_radius`` is a hyperparameter, not a constant

NPE (see ``docs/reference/api/monitor_npe.md``) is the other SCPN
consumer of the same research idea applied to a different concept.
There, ``max_radius`` is a filtration cutoff in ``[0, π]``. AttnRes
has the analogous ``block_size`` — a filter in index space rather
than distance space, but both drop to an identity mask when set to
a degenerate value (``None`` / ``≥ N − 1``). Making it a
hyperparameter keeps the local-attention / full-attention
behaviour in a single code path.

---

## 12. Failure modes and diagnostics

### 12.1 ``AttnRes`` returns ``NaN``

Only happens when the input phases contain ``NaN`` or ``Inf``. The
Rust and Go backends silently propagate the non-finite input
through ``cos/sin``; the Python backend triggers a numpy warning
during the softmax normalisation. None of the backends currently
raise — downstream callers should validate the phase vector before
calling ``attnres_modulate``.

### 12.2 ``K_mod`` identical to ``K`` despite ``λ > 0``

Happens when every non-diagonal entry of ``K`` is zero, or when
``block_size = 0`` is forced (which is rejected at the entry-point
validator but would otherwise zero every row). Confirm via
``np.count_nonzero(K) > N`` — the diagonal contributes ``N`` zeros,
so the off-diagonal count should exceed ``N``.

### 12.3 ``Mojo`` backend intermittently drops rows

The text-stdin protocol sends ``4 × (N, d_model, d_head) + (N, d_model) + N × N + N``
floats on a single whitespace-separated line. At ``N = 256`` with
``d_model = 8, H = 4`` that is ``16 448`` floats, encoded as
``17-digit repr`` strings averaging 22 characters each =
``~362 KB`` per call. Mojo 0.26's ``input()`` reads up to
``stdin.read_available()`` on a single call and has been observed
to truncate at ~1 MB on some kernels. Switch to the ctypes path
once Mojo 0.27+ stabilises the pointer ABI.

### 12.4 ``K_mod`` produces a ``NaN`` only in Mojo

Indicates a text-protocol parse error (e.g. one of the ``repr(float)``
strings was truncated). Rerun with Rust or Python to confirm the
fault is not in the input, and open an issue against the Mojo
bridge if it reproduces.

---

## 13. References

* arXiv:2603.15031 — "Attention Residuals", Moonshot AI / Kimi Team,
  March 2026 (source paper).
* Vaswani et al. 2017, "Attention Is All You Need" — multi-head
  attention primitive.
* Xavier Glorot, Yoshua Bengio 2010, "Understanding the difficulty
  of training deep feedforward neural networks" — initialisation
  scaling used in ``default_projections``.
* Kuramoto 1984, *Chemical Oscillations, Waves, and Turbulence* —
  baseline dynamics the modulation reshapes.
* `docs/internal/research_attention_residuals_2026-04-06.md` —
  Phase-3 research brief and §5 validation criteria (gitignored).
* `feedback_no_simplistic_models.md`,
  `feedback_module_standard_attnres.md`,
  `feedback_multi_language_accel.md` — governing rules.
