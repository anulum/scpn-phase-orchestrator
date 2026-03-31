# Oscillators

Phase extraction from raw signals via three channels: Physical (P),
Informational (I), Symbolic (S). This three-channel decomposition is
the core abstraction that makes SPO domain-agnostic — any signal
that exhibits periodic or quasi-periodic behaviour maps onto one or
more channels.

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

Every domain signal decomposes into at most three oscillator channels:

| Channel | Signal type | Extraction method | Example domains |
|---------|------------|-------------------|-----------------|
| **P** (Physical) | Continuous waveforms | Hilbert transform | EEG, ECG, vibration, voltage, plasma |
| **I** (Informational) | Event streams, rates | Inter-event interval | Network traffic, API calls, manufacturing |
| **S** (Symbolic) | Categorical sequences | Ring mapping | Protocols, language, music, genetics |

Not every domain uses all three channels. A pure physics domain (tokamak
plasma) might use only P. A pure IT domain (microservices) might use
only I. The binding specification declares which channels are active.

---

## Phase State

### PhaseState (dataclass)

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `theta` | `float` | [0, 2π) | Phase angle |
| `omega` | `float` | R | Instantaneous frequency (rad/s) |
| `amplitude` | `float` | ≥ 0 | Signal strength (SNR proxy) |
| `quality` | `float` | [0, 1] | Extraction confidence |
| `channel` | `str` | P, I, S | Channel type |
| `node_id` | `str` | — | Unique oscillator identifier |

Quality scores gate downstream processing: low-quality oscillators
are downweighted in coupling and excluded from regime classification.

---

## Extractor Interface

All channel extractors implement the `PhaseExtractor` abstract base class:

```python
class PhaseExtractor(ABC):
    @abstractmethod
    def extract(self, signal: NDArray, sample_rate: float) -> list[PhaseState]: ...

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
PhysicalExtractor(node_id: str = "phys_0")
```

Uses the analytic signal (Hilbert transform) to decompose a real-valued
waveform into instantaneous phase and amplitude:

```
z(t) = x(t) + i H[x(t)]
θ(t) = arg(z(t)),  A(t) = |z(t)|
ω = 2π × median(instantaneous frequency)
```

### Quality metric

`_envelope_quality(signal, analytic)` returns quality based on the
coefficient of variation (CV) of the analytic signal envelope:

```
quality = clip(1.0 - CV(|z(t)|), 0, 1)
```

Clean sinusoids have near-constant envelope (CV ≈ 0, quality ≈ 1.0).
Noisy signals have variable envelope (high CV, low quality).

### Validation

- Rejects empty signals, single-sample signals, and 2-D arrays
  with `ValueError("1-D with >= 2 samples")`
- Returns `channel = "P"`, `node_id` from constructor

### Rust acceleration

When `spo_kernel` is importable, uses `spo_kernel.physical_extract()`
for the core computation. Python fallback uses scipy Hilbert transform.
Parity verified in `tests/test_oscillator_physical.py::test_rust_python_parity`.

**Performance:** `extract(1s @ 1kHz)` < 5 ms.

::: scpn_phase_orchestrator.oscillators.physical

---

## Informational Extraction (I)

### InformationalExtractor

```python
InformationalExtractor(node_id: str = "info_0")
```

Converts event timestamps into phase oscillators:

1. Compute inter-event intervals: τ_k = t_k - t_{k-1}
2. Median frequency: f = 1 / median(τ)
3. Angular frequency: ω = 2πf
4. Phase: θ = (2πf × total_duration) mod 2π
5. Amplitude: mean instantaneous frequency
6. Quality: 1/(1 + CV(τ)) where CV = std(τ)/mean(τ)

### Edge cases

| Input | Result |
|-------|--------|
| Single timestamp | θ=0, ω=0, quality=0 |
| Identical timestamps | θ=0, ω=0, quality=0 |
| Two timestamps | Valid extraction from one interval |
| Regular events | quality ≈ 1.0 |
| Irregular events | quality < 0.9 |

**Performance:** `extract(100 timestamps)` < 500 μs.

::: scpn_phase_orchestrator.oscillators.informational

---

## Symbolic Extraction (S)

### SymbolicExtractor

```python
SymbolicExtractor(n_states: int, node_id: str = "sym", mode: str = "ring")
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_states` | `int` | Vocabulary size (≥ 2) |
| `node_id` | `str` | Oscillator identifier |
| `mode` | `str` | `"ring"` or `"graph"` |

### Ring mode

Maps state index s to phase: θ_s = 2πs / N (mod 2π).
Equispaced phases with gap = 2π/N.

### Graph mode

Cumulative transition distances normalised to [0, 2π).

### Quality scoring

| Transition type | Quality |
|----------------|---------|
| Single step (|Δs| = 1) | 1.0 |
| Stalled (Δs = 0) | 0.2 |
| Large jump (|Δs| = k) | max(0.1, 1 - (k-1)/N) |
| First state (no prior) | 0.5 |

### Omega derivation

ω is derived from consecutive phase differences divided by dt
(1/sample_rate). For ring mode with single steps: ω = 2π/(N·dt).

**Performance:** `extract(1000 states)` < 1 ms.

::: scpn_phase_orchestrator.oscillators.symbolic

---

## Quality Scoring

### PhaseQualityScorer

| Method | Signature | Description |
|--------|-----------|-------------|
| `score` | `(states) → float` | Amplitude-weighted mean quality |
| `detect_collapse` | `(states, threshold=0.1) → bool` | True if >50% below threshold |
| `downweight_mask` | `(states, min_quality=0.3) → NDArray` | Weight array, zeros below min |

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
Rust extension is installed. The Rust path computes the Hilbert
transform and phase extraction in a single pass, avoiding Python/NumPy
overhead for large signals.

Parity is verified in `tests/test_oscillator_physical.py::test_rust_python_parity`
with tolerance atol=1e-10 for phase, rtol=0.01 for frequency.

---

## Performance summary

| Operation | Budget | Rust | Notes |
|-----------|--------|------|-------|
| `PhysicalExtractor.extract(1s @ 1kHz)` | < 5 ms | < 1 ms | Hilbert transform |
| `InformationalExtractor.extract(100 ts)` | < 500 μs | — | numpy operations |
| `SymbolicExtractor.extract(1000 states)` | < 1 ms | — | ring mapping |
| `PhaseQualityScorer.downweight_mask(100)` | < 50 μs | — | array comparison |

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
