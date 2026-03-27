# Oscillators

Phase extraction from raw signals via three channels: Physical (P),
Informational (I), Symbolic (S). This three-channel decomposition is
the core abstraction that makes SPO domain-agnostic — any signal
that exhibits periodic or quasi-periodic behaviour maps onto one or
more channels.

## The PIS Model

Every domain signal decomposes into at most three oscillator channels:

| Channel | Signal type | Extraction method | Example domains |
|---------|------------|-------------------|-----------------|
| **P** (Physical) | Continuous waveforms | Hilbert transform, wavelet ridge, zero-crossing | EEG, ECG, vibration, voltage, plasma |
| **I** (Informational) | Event streams, rates | Inter-event interval, queue depth oscillation | Network traffic, API calls, manufacturing |
| **S** (Symbolic) | Categorical sequences | Ring mapping, graph embedding | Protocols, language, music, genetics |

Not every domain uses all three channels. A pure physics domain (tokamak
plasma) might use only P. A pure IT domain (microservices) might use
only I. The binding specification declares which channels are active.

## Phase State

Each extracted oscillator is represented by a `PhaseState` carrying:

- $\theta$ — phase angle in $[0, 2\pi)$
- $\omega$ — instantaneous frequency (rad/s)
- amplitude — signal strength (used for quality weighting)
- quality — extraction confidence in $[0, 1]$ (low SNR → low quality)
- channel — `"P"`, `"I"`, or `"S"`
- node_id — unique identifier for this oscillator

Quality scores gate downstream processing: low-quality oscillators
are downweighted in coupling and excluded from regime classification.

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

### Physical Extraction (P)

Uses the analytic signal (Hilbert transform) to decompose a real-valued
waveform into instantaneous phase and amplitude:

$$z(t) = x(t) + i \mathcal{H}[x(t)]$$
$$\theta(t) = \arg(z(t)), \quad A(t) = |z(t)|$$

For narrowband signals, this is exact. For broadband signals, a
bandpass filter is applied first (Savitzky-Golay or Butterworth),
and quality is penalised proportional to the out-of-band energy.

Alternative methods: wavelet ridge extraction (Morlet wavelet with
ridge-following) for non-stationary signals, zero-crossing detection
for digital/square-wave signals.

### Informational Extraction (I)

Converts event timestamps or rate time series into phase oscillators.
The inter-event interval $\tau_k = t_k - t_{k-1}$ defines an
instantaneous frequency $\omega_k = 2\pi / \tau_k$. Phase is
accumulated as $\theta(t) = \sum_{k: t_k \leq t} 2\pi \cdot (t - t_k) / \tau_k$.

Quality degrades when the event rate is too low (< 2 events per
expected cycle) or too irregular (coefficient of variation > 1.5).

### Symbolic Extraction (S)

Maps a categorical sequence onto the unit circle. For a vocabulary
of size $V$, symbol $s$ maps to phase $\theta_s = 2\pi s / V$.
For graph-structured vocabularies, uses spectral embedding of the
adjacency matrix to produce a phase assignment that preserves
topological structure.

Quality is 1.0 for deterministic sequences and degrades with
entropy: $q = 1 - H(s) / \log V$.

## Quality Scoring

The `PhaseQualityScorer` provides three operations:

1. **Weighted score** — amplitude-weighted average quality across all oscillators.
   High-amplitude oscillators contribute more because they have higher SNR.

2. **Collapse detection** — returns `True` if >50% of oscillators have quality
   below a threshold (default 0.1). Collapse indicates that the input signal
   has degraded to the point where phase extraction is unreliable.

3. **Downweight mask** — produces a weight array in $[0, 1]$ that zeros out
   oscillators below a minimum quality threshold. Used to gate coupling:
   low-quality oscillators should not influence high-quality ones.

## API Reference

### Base Types

::: scpn_phase_orchestrator.oscillators.base

### Physical Channel

::: scpn_phase_orchestrator.oscillators.physical

### Informational Channel

::: scpn_phase_orchestrator.oscillators.informational

### Symbolic Channel

::: scpn_phase_orchestrator.oscillators.symbolic

### Quality Scoring

::: scpn_phase_orchestrator.oscillators.quality
