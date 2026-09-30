# Oscillators

Phase extraction from raw signals via canonical Physical (P),
Informational (I), and Symbolic (S) channels, plus named extension
channels. The P/I/S decomposition is the default abstraction that makes
SPO domain-agnostic, but deployments are not limited to three channels:
any signal that exhibits periodic or quasi-periodic behaviour can map
onto one or more named channels.

## Pipeline position

```
Raw signals ──→ PhysicalExtractor  ──→ PhaseState(θ, ω, quality)
Event streams ──→ InformationalExtractor ──→ PhaseState(θ, ω, quality)
Sequences ──→ SymbolicExtractor ──→ PhaseState(θ, ω, quality)
                                              │
                                              ↓
                                    PhaseQualityScorer
                                              │
                                    ┌─────────┼──────────┐
                                    ↓         ↓          ↓
                              θ array    ω array    quality mask
                                    │         │          │
                                    ↓         ↓          ↓
                           UPDEEngine.step(phases, omegas, knm * mask, ...)
```

Oscillators are the **input adapters** of the SPO pipeline. They convert
raw domain signals into the `(θ, ω)` vectors that the engine requires.
Quality scores gate which oscillators participate in coupling.

---

## The PIS Model

Every domain signal decomposes into one or more oscillator channels. The
canonical channels are:

| Channel | Signal type | Extraction method | Example domains |
|---------|------------|-------------------|-----------------|
| **P** (Physical) | Continuous waveforms | Hilbert transform | EEG, ECG, vibration, voltage, plasma |
| **I** (Informational) | Event streams, rates | Inter-event interval | Network traffic, API calls, manufacturing |
| **S** (Symbolic) | Categorical sequences | Ring mapping | Protocols, language, music, genetics |

Not every domain uses all three canonical channels. A pure physics
domain (tokamak plasma) might use only P. A pure IT domain
(microservices) might use only I. Larger deployments can add named
extension channels such as `thermal`, `market_sentiment`, or
`operator_intent` while preserving the same `PhaseState` contract. The
binding specification declares which channels are active.

---

## Phase State

### PhaseState (dataclass)

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `theta` | `float` | [0, 2π) | Phase angle |
| `omega` | `float` | R | Instantaneous frequency (rad/s) |
| `amplitude` | `float` | ≥ 0 | Extractor-dependent signal magnitude |
| `quality` | `float` | [0, 1] | Extraction confidence |
| `channel` | `str` | Identifier | Binding channel (`P`, `I`, `S`, or named extension) |
| `node_id` | `str` | — | Unique oscillator identifier |

`PhaseQualityScorer` requires plain real quality values and amplitude weights.
Text, boolean, complex and temporal aliases are rejected before conversion.
Nonfinite real qualities retain the documented skip/collapse/zero-mask policy.
Constructor and per-call quality thresholds must also be plain real numbers.

Quality scores gate downstream processing: low-quality oscillators
are downweighted in coupling and excluded from regime classification.

---

## Extractor Interface

All channel extractors implement the `PhaseExtractor` abstract base class:

```python
from numpy.typing import NDArray
import numpy as np

FloatArray = NDArray[np.float64]

class PhaseExtractor(ABC):
    @abstractmethod
    def extract(self, signal: FloatArray, sample_rate: float) -> list[PhaseState]: ...

    @abstractmethod
    def quality_score(self, phase_states: list[PhaseState]) -> float: ...
```

The `extract` method receives a raw signal window and sample rate,
and returns one or more `PhaseState` objects. The `quality_score`
method computes an aggregate quality for the extraction.

---

## Physical Extraction (P)

### PhysicalExtractor

```python
PhysicalExtractor(
    node_id: str = "phys_0", *,
    band: tuple[float, float] | None = None,
    filter_order: int = 4,
    edge_trim: int | None = None,
)
```

Uses the analytic signal (Hilbert transform) to decompose a real-valued
waveform into instantaneous phase and amplitude:

Hilbert, wavelet and zero-crossing extractors accept one-dimensional finite
integer or floating-point measurement arrays. Text, boolean, complex, object,
`datetime64` and `timedelta64` arrays are rejected before signal processing.
Boolean values in numeric lists are also rejected before NumPy promotes them
to numbers. Convert temporal quantities to explicitly chosen physical units
at ingestion; changing a duration dtype must not silently change the waveform.

```
z(t) = x(t) + i H[x(t)]
θ(t) = arg(z(t)),  A(t) = |z(t)|
ω = 2π × median(instantaneous frequency)
```

### Quality metric

The returned `PhaseState.quality` uses the coefficient of variation (CV) of the analytic signal envelope:

```
quality = clip(1.0 - CV(|z(t)|), 0, 1)
```

Clean sinusoids have near-constant envelope (CV ≈ 0, quality ≈ 1.0).
Variable envelopes have high CV and low quality, including deterministic
amplitude modulation with an exactly known phase. A mean envelope below
`1e-15` gives zero quality; the score does not measure SNR or observability.

### Validation

- Rejects empty signals, single-sample signals, and 2-D arrays
  with `ValueError("1-D with >= 2 samples")`
- Returns `channel = "P"`, `node_id` from constructor

### Rust acceleration

When `spo_kernel` is importable, uses `spo_kernel.physical_extract()`
for the core computation. Python fallback uses scipy Hilbert transform.
The Hilbert transform and optional filtering/trimming run before either core.
Both real environments are exercised by `tests/test_oscillator_physical.py`;
`native-tests/test_physical.py` additionally verifies the registered native
call and direct readonly-array admission. Scaled envelope statistics avoid
intermediate overflow without changing the CV equation or its absolute
`1e-15` quality cutoff. This protects representable analytic envelopes;
filtering/Hilbert preprocessing can still overflow on finite raw samples.
Public extraction refuses a non-finite analytic signal with `ValueError`
before entering either statistics core.

**Local test budget:** `extract(1s @ 1kHz)` averages below 5 ms in the
owning host-sensitive test. This is not a published latency guarantee.


### Reproducing local physical diagnostics

Run `python -m benchmarks.physical_extraction --repeats 20` in each actual
runtime. The command records native C-call identities, artifact/source hashes,
raw-waveform and analytic-signal hashes, NumPy/SciPy versions, host affinity,
load, all four extraction fields and a separately computed equation reference.
Every case must agree with its own runtime's reference before JSON is emitted.

The following medians in microseconds were recorded on 2026-09-30 on an Intel
i5-11600K, CPython 3.12, affinity CPUs 0–11, without isolation. Host load was
20.5–25.3 during the runs. These are functional diagnostics on a shared host;
they do not support a speedup or deployment latency claim. The native runtime
used NumPy 2.5.3/SciPy 1.18.1; the actual kernel-absent runtime used
NumPy 2.2.6/SciPy 1.15.3. Each row passed its separately computed numerical reference.

| Workload | Native runtime median (µs) | Kernel-absent runtime median (µs) |
|----------|----------------------------|----------------------------------|
| sinusoid_1000 | 195.4 | 397.3 |
| bandpass_1000 | 928.6 | 1233.9 |
| zero_envelope | 108.7 | 227.2 |
| tiny_envelope | 113.5 | 304.0 |
| trim_to_two | 139.9 | 342.8 |
| modulated_127 | 140.6 | 330.9 |
| large_modulated_127 | 125.3 | 496.8 |
| modulated_128 | 111.5 | 299.7 |
| large_modulated_128 | 115.6 | 317.3 |

All nine raw-waveform hashes matched between runtimes. The band-pass analytic
hash differed and its endpoint phase differed by about `6.72e-6` radians;
each runtime still matched its own filtering/Hilbert reference. The zero and
tiny analytic hashes also differed while their returned fields agreed.
Equal raw inputs therefore do not establish bitwise preprocessing parity
across different dependency versions. Compare matched preprocessing and
recorded runtime versions before making a numerical or timing claim.

::: scpn_phase_orchestrator.oscillators.physical

---

## Informational Extraction (I)

### InformationalExtractor

```python
InformationalExtractor(node_id: str = "info_0")
```

Converts event timestamps into phase oscillators:

1. Compute inter-event intervals: τ_k = t_k - t_{k-1}; retain positive intervals
2. Median frequency: f = median(1 / τ)
3. Angular frequency: ω = 2πf
4. Phase: θ = (2πf × total_duration) mod 2π
5. Amplitude: mean instantaneous frequency
6. Quality: 1/(1 + CV(τ)) where CV = std(τ)/mean(τ)

For an even interval count, median instantaneous frequency is not generally
the reciprocal of median interval. For timestamps `[0, 0.125, 0.375]`, the
frequencies are 8 and 4 Hz: f = 6 Hz, θ = π/2, amplitude = 6 and quality = 0.75.
Both Python and the optional Rust kernel use this median-frequency convention.
Duplicate events contribute no interval; unsorted timestamps raise `ValueError`.
Quality uses population standard deviation of the positive intervals.

### Edge cases

Event timestamps must be a one-dimensional array. Scalars, row or column
matrices, higher-rank arrays and empty matrices raise `ValueError` before phase
extraction. An empty one-dimensional array is a valid degenerate event train.
Refusal does not modify the input or prevent a later valid extraction with the
same extractor.

| Input | Result |
|-------|--------|
| Single timestamp | θ=0, ω=0, quality=0 |
| Identical timestamps | θ=0, ω=0, quality=0 |
| Two timestamps | Valid extraction from one interval |
| Regular events | quality ≈ 1.0 |
| Non-constant positive intervals | quality < 1.0 |

**Local test limit:** `extract(100 timestamps)` < 600 μs averaged over
1,000 extractions after warm-up. Hosted CI deselects this performance test;
the limit is not a hosted gate or a new benchmark result.

::: scpn_phase_orchestrator.oscillators.informational

---

## Symbolic Extraction (S)

### SymbolicExtractor

```python
SymbolicExtractor(n_states: int, node_id: str = "sym", mode: str = "ring", *,
                  initial_transition_quality: float = 0.5)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_states` | `int` | Vocabulary size (≥ 2) |
| `node_id` | `str` | Oscillator identifier |
| `mode` | `str` | `"ring"` or `"graph"` |
| `initial_transition_quality` | `float` | Finite initial quality in [0, 1]; default 0.5 |

### Ring mode

Maps state index s to phase: θ_s = 2πs / N (mod 2π).
Equispaced phases with gap = 2π/N.
Signed and out-of-vocabulary integer labels are cyclic aliases modulo N.

### Graph mode

Cumulative absolute linear index differences normalised to [0, 2π).
No adjacency graph is consulted. A singleton uses the ring mapping; a
stationary multi-state sequence has zero phase.

### Integer and backend contract

Signed and unsigned integer arrays retain their labels: narrower widths promote
to `int64` or `uint64` respectively, never by wrapping unsigned labels into
negative values. One-dimensional strided, reversed and read-only views are
accepted. Unaligned public input is copied before native access.

Graph distances and cumulative totals use exact integer arithmetic (Python
integers or Rust `u128`) before conversion to `float64`. This preserves full-span
64-bit label distances without signed overflow or saturating the walk total.
Output phases and frequencies still have `float64` precision; sub-ULP differences
and underflow are not exact rational results.

Ring phases and graph qualities are numerically, not bitwise, backend-equivalent.
Rust divides converted integer operands; Python rounds the integer quotient.
Above `2**53`, those operations can produce adjacent float64 values. For
`n_states = 2**53 + 1` and label 1, the native ring phase is one ULP above
the Python phase. Integer residues and graph distances remain exact.

`n_states` has no public upper bound beyond being an integer ≥ 2. The Rust path
uses the target's `usize` capacity; larger counts use the real Python mapping and
quality computation, even when `spo_kernel` is installed. Without the kernel,
all symbolic computations use Python. Cyclic ring quality always uses Python.

### Quality scoring

| Transition type | Quality |
|----------------|---------|
| Single step (k = 1) | 1.0 |
| Stalled (k = 0) | 0.2 |
| Large jump (k > 1) | max(0.1, 1 - (k-1)/N) |
| First state (no prior) | Configured initial quality; default 0.5 |

Graph mode uses k = |Δs|. Ring mode uses d = |Δs| mod N and
k = min(d, N-d), so full cycles score as stalls, including signed aliases.

### Omega derivation

ω is derived from consecutive phase differences wrapped to [-π, π), divided
by dt (1/sample_rate). The first frequency is zero; an exact half-turn is
negative. For forward single steps with N > 2: ω = 2π/(N·dt).

**Local test budget:** `extract(1000 states)` averages below 5 ms, or
50 ms when CI is set or the one-minute load exceeds
`max(2, cpu_count/4)`. This is the existing load-aware test limit, not a
published comparative benchmark; hosted CI deselects performance tests.

The real public-path diagnostic covers ring, graph, full-span signed/unsigned
labels, strides and counts above native capacity:

```bash
PYTHONPATH=src .venv/bin/python benchmarks/bench_symbolic.py --repeats 20
```

It reports actual native C-call identities outside timed loops, source hashes,
runtime versions, host context and per-workload median/P95 timings. Shared-host
runs are non-isolated diagnostics, not portable speedup or latency guarantees.
The [2026-09-30 raw observations](../data/symbolic_extraction_diagnostic_2026-09-30.json)
contain separate installed-native and genuinely kernel-absent runs.

::: scpn_phase_orchestrator.oscillators.symbolic
    options:
      docstring_style: numpy

---

## Wavelet-ridge extractor (physical channel)

`WaveletExtractor` is a band-adaptive alternative to the Hilbert extractor: it
computes a complex Morlet continuous wavelet transform across a log-spaced
frequency bank, selects the dominant energy ridge over a cone-of-influence-safe
interior region, and reads the analytic phase along that ridge. The terminal
phase is extrapolated from a COI-safe interior sample at the ridge frequency, so
the corrupted signal edge is avoided. The Morlet wavelet is
`ψ(t) = π^(−1/4)·exp(i·ω₀·t/s)·exp(−(t/s)²/2)/√s` with `ω₀ = 6`; a pure-NumPy
path is used because SciPy 1.15 removed `cwt`/`morlet2`. Prefer it over Hilbert
when a dominant oscillation sits in broadband noise or slow drift.

::: scpn_phase_orchestrator.oscillators.wavelet

---

## Zero-crossing extractor (physical channel)

`ZeroCrossingExtractor` recovers phase from interpolated zero crossings: each
crossing is a half-cycle (`π` of phase advance), absolute phase is anchored to
the crossing direction (a rising crossing ≡ 0, a falling crossing ≡ `π`, the
sine convention), and a Schmitt-trigger deadband (a fraction of the RMS)
suppresses spurious noise-induced crossings. Angular frequency comes from the
mean half-period and quality from the regularity of the half-period intervals.
Prefer it for sharply non-sinusoidal periodic signals where a single analytic
phase is ill-defined.

::: scpn_phase_orchestrator.oscillators.zero_crossing

---

## Quality Scoring

### PhaseQualityScorer

| Method | Signature | Description |
|--------|-----------|-------------|
| `score` | `(states) → float` | Amplitude-weighted mean quality |
| `detect_collapse` | `(states, threshold=0.1) → bool` | True if >50% below threshold |
| `downweight_mask` | `(states, min_quality=0.3) → NDArray[np.float64]` | Weight array, zeros below min |

### Downweight mask in pipeline

The mask is applied to the coupling matrix before engine evaluation:

```python
mask = scorer.downweight_mask(states, min_quality=0.3)
knm_gated = knm * mask[:, None] * mask[None, :]
# Low-quality oscillators decoupled from high-quality ones
```

This prevents noisy phase estimates from corrupting the synchronisation
dynamics. Only oscillators with quality ≥ min_quality participate.

**Performance:** `downweight_mask(100 states)` < 50 μs.

::: scpn_phase_orchestrator.oscillators.quality

---

## Base Types

::: scpn_phase_orchestrator.oscillators.base

## Phase Initialisation

Utilities for deterministic and random initial phase generation used in
simulation setup and reproducible experiment seeds.

::: scpn_phase_orchestrator.oscillators.init_phases

---

## Phase reduction

Model-free phase reduction: a dependency-light evaluator of a trained phase
autoencoder (see `nn.phase_autoencoder`) that recovers the asymptotic phase
`Θ(x)` and the phase-sensitivity function `Z(θ)` — the phase response curve —
from frozen NumPy weights, with no JAX on the control path.

::: scpn_phase_orchestrator.oscillators.phase_reduction

---

## Extractor factory

`build_extractor` maps a binding `extractor_type` — a channel alias
(`physical`/`informational`/`symbolic`) or a canonical algorithm name
(`hilbert`/`wavelet`/`zero_crossing`/`event`/`ring`/`graph`) — to the concrete
`PhaseExtractor` that implements it. Aliases resolve through
`resolve_extractor_type`; an unknown type raises `ValueError` (fail-closed)
rather than silently degrading to a default algorithm.

::: scpn_phase_orchestrator.oscillators.factory

---

## Cross-channel composition

A domain can use multiple channels simultaneously. The binding spec
declares which channels are active and how they map to oscillator
indices:

```yaml
layers:
  - name: voltage
    channel: P
    indices: [0, 1, 2, 3]
  - name: event_rate
    channel: I
    indices: [4, 5]
  - name: protocol_state
    channel: S
    indices: [6, 7]
```

All channels produce `PhaseState` with the same fields, so the engine
treats them uniformly. The `channel` field enables channel-aware
analysis (e.g., computing R separately for P and I oscillators).

## Rust FFI acceleration

`PhysicalExtractor` uses `spo_kernel.physical_extract()` when the
Rust extension is installed. SciPy computes filtering and the Hilbert transform
before the native core receives contiguous real and imaginary arrays. Rust then
computes phase, median angular frequency, scaled mean amplitude and envelope CV.

`tests/test_oscillator_physical.py` checks public analytical contracts in native
and kernel-absent environments. `native-tests/test_physical.py` proves registered
C-call dispatch and direct native admission. The diagnostic command above records
floating-point agreement with each runtime's reference; bitwise parity depends
on the preprocessing environment.

---

## Performance summary

| Operation | Budget | Rust | Notes |
|-----------|--------|------|-------|
| `PhysicalExtractor.extract(1s @ 1kHz)` | Local test: < 5 ms | Same public test budget | Hilbert transform; no published speedup |
| `InformationalExtractor.extract(100 ts)` | < 600 μs (local test limit, deselected in hosted CI) | — | native kernel or NumPy fallback |
| `SymbolicExtractor.extract(1000 states)` | Local test: < 5 ms; < 50 ms under CI/high load | — | ring mapping; not a published benchmark |
| `PhaseQualityScorer.downweight_mask(100)` | < 50 μs | 3.15 μs | public wrapper: 31.57 μs |

Quality timings measured on 2026-09-26 with the rebuilt release extension,
five repeats of 100 calls after warm-up on a shared host. Direct Rust mask
and the public wrapper include source-type validation. The supported per-call
threshold override selects the real Python mask path, measured at 32.40 μs
for the same applied threshold and input. This is a current measurement,
not an isolated performance comparison against an earlier release.
[Raw quality timing records and source hashes](../data/quality_measurement_types_benchmark_2026-09-26.json)
cover N=10/100/1000. Pure Python score was not measured on this kernel-installed
host; the public score uses the real Rust backend.

The direct `spo_kernel.PyPhaseQualityScorer` also rejects text, boolean,
complex and temporal aliases, while retaining real arrays and documented
nonfinite measurement policies. Its constructor retains the `0.1`/`0.3`
defaults and requires finite unit-interval thresholds. Native boundary tests
run under `native-tests/` in the kernel-built Unix and Windows FFI CI lanes.

## Domain examples

### Neuroscience (EEG)

```python
# 64-channel EEG → 64 P-channel oscillators
extractor = PhysicalExtractor(node_id="eeg")
for ch in range(64):
    states = extractor.extract(eeg_data[ch], fs=256.0)
    phases[ch] = states[0].theta
    omegas[ch] = states[0].omega
```

### Microservices (queue depths)

```python
# 12 services → 12 I-channel oscillators
extractor = InformationalExtractor(node_id="svc")
for svc in services:
    timestamps = svc.request_timestamps()
    states = extractor.extract(timestamps, sample_rate=0.0)
    phases[svc.id] = states[0].theta
```

### Genomic sequences

```python
# DNA codons → S-channel oscillators
extractor = SymbolicExtractor(n_states=64, mode="ring")
codon_indices = encode_codons(sequence)
states = extractor.extract(codon_indices, sample_rate=1.0)
```

## Original numerical input types

See [numerical source types](numerical_source_types.md) for text, boolean and
temporal refusal, numeric-object compatibility and current language measurements.
