# Digital-Twin Confidence — Online Model–Observation Divergence

The `monitor.twin_confidence` module scores how well a running orchestrator
tracks its physical (or simulated) twin. At every control tick both sides emit a
phase vector and an order-parameter trajectory; the module turns the
*disagreement* between the two streams into a single calibrated confidence in
`[0, 1]` plus an operator status (`healthy` / `warning` / `critical`). It is
review-only: it never proposes or applies actuation, and is consumed as a health
observable by the digital-twin operator-evidence summary and the observability
exporters.

The heavy path is the `(js, w1)` divergence kernel — a phase-histogram
Jensen–Shannon divergence and an order-parameter Wasserstein-1 distance —
served by the five-language backend chain (Rust → Mojo → Julia → Go → NumPy).
The calibration, confidence map, operating bands, and audit records sit in
deterministic NumPy / Python on top.

---

## 1. Mathematical formalism

### 1.1 Phase Jensen–Shannon divergence

Model phases `θ^m ∈ ℝ^N` and observed phases `θ^o ∈ ℝ^N` are wrapped to
`[0, 2π)` and binned into `B = n_bins` equal-width histograms, normalised to
probability mass functions `p` and `q` (the uniform PMF is used when a stream is
empty). The bin index for a phase `φ` is

$$
b(\varphi) \;=\; \min\!\Bigl(\Bigl\lfloor \tfrac{\varphi - \lfloor \varphi / 2\pi\rfloor\,2\pi}{2\pi / B} \Bigr\rfloor,\; B-1\Bigr).
$$

The symmetric **Jensen–Shannon divergence** (natural log) with mixture
`m = ½(p + q)` is

$$
\mathrm{JS}(p, q) \;=\; \tfrac12 D_{\mathrm{KL}}(p \,\|\, m) + \tfrac12 D_{\mathrm{KL}}(q \,\|\, m),
\qquad D_{\mathrm{KL}}(p \,\|\, m) = \sum_{i:\,p_i>0} p_i \ln \tfrac{p_i}{m_i}.
$$

JS is bounded in `[0, ln 2]`, is symmetric, and is `0` iff `p = q`. The
`0 \ln 0 = 0` convention handles empty bins, and `m_i > 0` wherever `p_i > 0`,
so the ratio is always finite.

### 1.2 Order-parameter Wasserstein-1 distance

The model and observed order-parameter windows `R^m, R^o ∈ [0, 1]^W` are
compared with the closed-form one-dimensional Wasserstein-1 (earth-mover)
distance, which for equal-length equal-weight samples reduces to the mean
absolute difference of the order-sorted windows:

$$
W_1(R^m, R^o) \;=\; \frac{1}{W} \sum_{k=1}^{W} \bigl|\, R^m_{(k)} - R^o_{(k)} \,\bigr|,
$$

where `R_{(k)}` is the `k`-th order statistic. Because `R ∈ [0, 1]`,
`W_1 ∈ [0, 1]`; it is symmetric and `0` iff the two windows are permutations of
each other.

### 1.3 Calibration and operating bands

A `TwinConfidenceCalibrator` ingests `(js, w1)` pairs gathered during trusted
nominal operation and fits per-divergence means and population standard
deviations `(μ_{js}, σ_{js}, μ_{w1}, σ_{w1})` together with a normal-quantile
upper operating band `μ + z_\text{band}\,σ` (default `z_band = 3`). A divergence
above its band is flagged out-of-band in the score.

### 1.4 Confidence map

Each divergence is converted to a one-sided z-score against its baseline,

$$
z_{js} = \max\!\Bigl(0, \tfrac{\mathrm{JS} - \mu_{js}}{\max(\sigma_{js}, \varepsilon)}\Bigr),
\qquad
z_{w1} = \max\!\Bigl(0, \tfrac{W_1 - \mu_{w1}}{\max(\sigma_{w1}, \varepsilon)}\Bigr),
$$

combined into a composite Euclidean deviation `z = \sqrt{z_{js}^2 + z_{w1}^2}`,
and mapped to confidence

$$
c \;=\; \mathrm{clip}\bigl(e^{-z / s},\, 0,\, 1\bigr),
$$

with sensitivity `s > 0` (default `3`). The confidence is exactly `1` while both
divergences sit at or below their nominal means and decays smoothly as the twin
drifts. The status is `healthy` for `c ≥ warning_confidence`, `warning` for
`critical_confidence ≤ c < warning_confidence`, and `critical` below that
(defaults `0.6` / `0.3`).

---

## 2. API

```python
from scpn_phase_orchestrator.monitor.twin_confidence import (
    ACTIVE_BACKEND,             # str  — currently serving backend
    AVAILABLE_BACKENDS,         # list — resolved backends, order = preference
    TwinDivergence,             # dataclass — (js, w1, n_bins, backend)
    TwinConfidenceBaseline,     # dataclass — calibrated means/stds + bands
    TwinConfidenceCalibrator,   # class — accumulates nominal samples → baseline
    TwinConfidenceScore,        # dataclass — confidence, status, z-scores, hash
    phase_order_divergence,     # fn — compute (js, w1) for one tick
    score_twin_confidence,      # fn — score a divergence against a baseline
)
```

```python
def phase_order_divergence(
    model_phases, observed_phases, model_order, observed_order,
    *, n_bins: int = 36,
) -> TwinDivergence: ...

def score_twin_confidence(
    divergence, baseline,
    *, sensitivity: float = 3.0,
    warning_confidence: float = 0.6,
    critical_confidence: float = 0.3,
) -> TwinConfidenceScore: ...
```

`phase_order_divergence` validates that the phase vectors share a non-empty
length `N`, the order windows share a non-empty length `W`, every value is a
finite real (boolean and complex aliases rejected), the order values lie in
`[0, 1]`, and `n_bins` is a positive integer. `score_twin_confidence` requires
`sensitivity > 0`, both confidence thresholds in `[0, 1]`, and
`critical_confidence ≤ warning_confidence`. Every dataclass exposes a
deterministic JSON-safe `to_audit_record()`, and `TwinConfidenceScore` carries a
SHA-256 `score_hash` over its record.

---

### 2.4 Operator surfaces

`summarise_twin_confidence(scores)` aggregates a sequence of
`TwinConfidenceScore` into a `TwinConfidenceSummary` — tick count, per-status
counts, min/mean/latest confidence, worst and latest status, and a deterministic
hash. `twin_confidence_prometheus_text(summary, prefix="spo")` renders it as
Prometheus exposition text (confidence gauges, a per-status counter, and a
numeric worst-status level). `RuntimeObservability.twin_confidence_prometheus_text`
delegates to the same renderer with the runtime's metric prefix.
`studio.build_twin_confidence_studio_panel(score_records, summary_record)` uses
the same score and summary audit records for the review-only Studio surface,
validating hashes and aggregate consistency before rendering latest/worst status
evidence.

The `spo twin-confidence` CLI command scores an observation stream against a
calibrated baseline:

```bash
spo twin-confidence --calibration nominal.jsonl --observations live.jsonl
spo twin-confidence --calibration nominal.jsonl --observations live.jsonl --json-out
spo twin-confidence --calibration nominal.jsonl --observations live.jsonl --prometheus
spo twin-confidence --calibration nominal.jsonl --observations live.jsonl --fail-on-critical
```

Each JSONL line is one tick: a JSON object with `model_phases`,
`observed_phases`, `model_order`, and `observed_order` arrays. The calibration
file fits the baseline; the observation file is scored. Options expose `--n-bins`,
`--sensitivity`, `--warning-confidence`, `--critical-confidence`, and `--band-z`.
`--fail-on-critical` exits non-zero when the worst scored status is critical, so
the command can gate a deployment pipeline.

## 3. Backend fallback chain

The module resolves backends in the order **Rust → Mojo → Julia → Go → Python**
at import time; the first that loads becomes `ACTIVE_BACKEND`, and Python is
always appended as the guaranteed fallback.

| Backend | Probe | Artefact |
| ------- | ----- | -------- |
| Rust | `from spo_kernel import twin_divergence_rust` | `spo_kernel` wheel via maturin. |
| Mojo | `_ensure_exe()` → `mojo/twin_confidence_mojo` | Executable from `mojo build … -Xlinker -lm`. |
| Julia | `import juliacall`; `Main.include("julia/twin_confidence.jl")` | `juliacall` binding + Julia 1.12. |
| Go | `ctypes.CDLL("go/libtwin_confidence.so")` | C-shared library from `go build`. |
| Python | Pure NumPy — no external dependencies. | Built in. |

The backend kernel computes only the raw `(js, w1)` pair from flat `float64`
buffers; the public entry point validates inputs once, dispatches, and validates
the returned pair (shape `(2,)`, finite, `js ∈ [0, ln 2]`, `w1 ∈ [0, 1]` within
parity tolerance). Force a backend for tests/benchmarks via
`twin_confidence.ACTIVE_BACKEND = "go"`.

### 3.1 Semantic equivalence

| Backend vs NumPy reference | Tolerance (atol) | Reason |
| -------------------------- | ---------------- | ------ |
| Rust | `1e-12` | Shared `f64` arithmetic; no serialisation. |
| Julia | `1e-12` | `juliacall` passes `Float64` arrays directly. |
| Go | `1e-12` | `ctypes` passes `double*` by pointer. |
| Mojo | `1e-8` | Subprocess text protocol; `std.math.log` is an approximation, so the Jensen–Shannon term carries a ~`1e-9` floor while the Wasserstein term stays bit-exact. |
| Python | exact | Python is the reference. |

The polyglot parity gate
(`benchmarks/twin_confidence_benchmark.py --parity-gate`) confirmed
`max_abs_error` of `1.7e-18` (Rust), `1.7e-18` (Julia), `6.9e-18` (Go), and
`2.1e-11` (Mojo) on a 200-phase / 64-window problem.

---

## 4. Benchmarks

Two harnesses measure the kernel. Both ran on a loaded Ubuntu 24.04 workstation
(non-isolated; functional/local evidence per the benchmark-core-isolation
policy, not an isolated-core production claim), `spo_kernel` built with
`maturin develop`.

**Python dispatcher harness** — milliseconds per call for a fresh
`(N phases, W = 64 window, n_bins = 36)` problem, `calls = 25`. Re-run with
`python benchmarks/twin_confidence_benchmark.py --sizes 64 256 1024`.

| N | rust (ms) | mojo (ms) | julia (ms) | go (ms) | python (ms) |
| ---- | --------: | --------: | ---------: | ------: | ----------: |
| 64 | 0.180 | 24.58 | 0.234 | 0.198 | 0.168 |
| 256 | 0.299 | 22.30 | 0.855 | 0.802 | 0.564 |
| 1024 | 0.948 | 25.97 | 0.958 | 0.966 | 0.688 |

**Rust criterion harness** — pure-kernel time (no FFI / array marshalling),
`cargo bench -p spo-engine --bench twin_confidence_bench`:

| N | time (median) |
| ---- | ------------: |
| 64 | 1.59 µs |
| 256 | 3.83 µs |
| 1024 | 14.87 µs |

Observations:

* **NumPy is competitive at small N.** The histogram (`np.bincount`) and the
  two sorts are vectorised, so for `N ≤ 256` the FFI marshalling overhead of the
  compiled backends offsets their faster inner loops.
* **The Rust kernel is the raw-throughput floor.** Criterion shows the pure
  kernel at single-digit microseconds; the Python-harness figure adds NumPy
  array setup and the FFI round-trip.
* **Mojo subprocess overhead dominates.** Each call forks a process and parses a
  text stream; Mojo is retained for parity coverage rather than throughput.

---

## 5. Tests

* `tests/test_twin_confidence.py` — algorithm invariants (JS symmetry and
  boundedness, exact Wasserstein on a shift, phase-wrap invariance), every
  input-validation and kernel-output branch, the dispatcher, calibration, and
  the confidence map including band membership and the deterministic hash.
* `tests/test_twin_confidence_backends.py` — the shared backend-validation
  contract (always runs) plus per-backend parity against the NumPy reference and
  pairwise cross-backend agreement, each gated on toolchain availability.
* `tests/test_twin_confidence_stability.py` (`@pytest.mark.slow`) — divergence
  bounds over a random sweep, metric symmetry and identity of indiscernibles,
  monotone confidence decay, calibration robustness, long-run drift-freedom on
  identical streams, and overflow-free phase wrapping up to `2π·10⁹`.
* `tests/test_twin_confidence_cli.py` — the `spo twin-confidence` command across
  human / JSON / Prometheus output, `--fail-on-critical`, and every JSONL loader
  error path (malformed line, non-object tick, missing field, non-numeric field,
  empty stream).
* `tests/test_runtime_observability.py` — the `MetricsExporter.export_twin_confidence`
  and `RuntimeObservability.twin_confidence_prometheus_text` facade.

```bash
pytest tests/test_twin_confidence.py tests/test_twin_confidence_backends.py
pytest tests/test_twin_confidence_stability.py -m slow
```

---

## 6. Failure modes and caveats

* **Calibration trust.** The baseline is only as good as the window it was
  fitted on; a baseline gathered while the twin already drifts will read the
  drift as nominal. Fit during commissioning or a trusted healthy replay.
* **Degenerate baseline variance.** When a divergence is constant during
  calibration its `σ` is floored by `ε = 1e-12`, so a tiny runtime increase
  registers a large z-score. This is intentional fail-sensitive behaviour;
  widen the calibration window if it is too twitchy.
* **Mojo log approximation.** Mojo's `std.math.log` sets a ~`1e-9` floor on the
  Jensen–Shannon term; the order Wasserstein term is bit-exact. The `1e-8`
  parity budget is an empirical bound, not a theoretical guarantee.
* **Histogram resolution.** `n_bins` trades phase resolution against sample
  noise; the default `36` (10° per bin) suits a few hundred oscillators. Raise
  it for sharply multimodal phase populations, lower it for short windows.

---

## 7. References

* Lin, J. (1991). *Divergence measures based on the Shannon entropy.* IEEE
  Transactions on Information Theory **37** (1), 145–151.
* Endres, D. M., Schindelin, J. E. (2003). *A new metric for probability
  distributions.* IEEE Transactions on Information Theory **49** (7), 1858–1860.
* Villani, C. (2009). *Optimal Transport: Old and New.* Grundlehren der
  mathematischen Wissenschaften **338**, Springer.
* Ramdas, A., García Trillos, N., Cuturi, M. (2017). *On Wasserstein two-sample
  testing and related families of nonparametric tests.* Entropy **19** (2), 47.
