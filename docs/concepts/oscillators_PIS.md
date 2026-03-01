# Oscillators: P / I / S Channels

Three extraction channels convert domain signals into phase states.

## P -- Physical Channel

**Input:** Continuous waveform (voltage, pressure, displacement).

**Method:** Hilbert transform. The analytic signal `a(t) = s(t) + i*H[s(t)]` yields instantaneous phase `theta = arg(a)` and amplitude `|a|`. Instantaneous frequency from the phase gradient.

**Quality:** SNR estimate: `sig_power / (sig_power + noise_power)`, saturating at 1.

**When to use:** The signal is a real-valued time series with a dominant oscillatory component. EEG, vibration, AC current, heartbeat waveform.

**Extractor:** `PhysicalExtractor` in `oscillators.physical`.

## I -- Informational Channel

**Input:** Event timestamps (sorted ascending). Spike trains, request arrivals, heartbeat R-peaks.

**Method:** Inter-event intervals yield instantaneous frequency `f = 1/IEI`. Phase via cumulative integral `theta = cumsum(2*pi*f*dt)`, taken modulo 2*pi.

**Quality:** Inverse coefficient of variation of intervals: `1 / (1 + CV)`. Regular events score high; bursty or irregular events score low.

**When to use:** The observable is a point process, not a continuous signal. Packet arrivals, neuron spikes, heartbeat R-peaks, job completions.

**Extractor:** `InformationalExtractor` in `oscillators.informational`.

## S -- Symbolic Channel

**Input:** Sequence of discrete state indices `s in {0, 1, ..., N-1}`.

**Method:**
- *Ring mode:* `theta = 2*pi*s/N`. Maps discrete state to equally spaced phases on the unit circle.
- *Graph mode:* normalise sequential position to `[0, 2*pi)`. Frequency from phase differences between successive states.

**Quality:** Transition regularity. Single-step transitions score 1.0. Stalls (repeated state) score 0.2. Large jumps penalised proportionally to `step/N`.

**When to use:** The observable is a finite state machine, protocol state, or categorical variable that cycles. TCP states, workflow stages, Markov chain position, graph random walk.

**Extractor:** `SymbolicExtractor` in `oscillators.symbolic`.

## Mixed-Channel Systems

A single domain can use all three channels. The `oscillator_families` dict in the binding spec maps family names to channels:

```yaml
oscillator_families:
  heart_wave:
    channel: P
    extractor_type: physical
  spike_train:
    channel: I
    extractor_type: informational
  protocol_state:
    channel: S
    extractor_type: symbolic
    config:
      n_states: 8
```

All channels output `PhaseState` with the same fields. The UPDE engine does not distinguish channels -- it operates on the unified phase vector.

## Quality Gating

Extracted phases with `quality < min_quality` (default 0.3) are down-weighted by `PhaseQualityScorer.downweight_mask()`. Collapsed oscillators (quality < 0.1 for majority of states) trigger `detect_collapse()`.
