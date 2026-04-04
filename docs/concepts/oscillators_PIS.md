# Oscillators: P / I / S Channels

Three extraction channels convert domain signals into phase states.
This is the universal interface through which any coupled-cycle system
enters the SCPN Phase Orchestrator. The choice of channel determines
how raw data becomes a phase on `[0, 2pi)`, a frequency in rad/s, and
a confidence score in `[0, 1]`.

The key insight: once a signal is expressed as a `PhaseState`, the
UPDE engine treats it identically regardless of origin. An EEG alpha
wave (P), a packet arrival process (I), and a protocol state machine
(S) all become entries in the same phase vector, coupled through the
same K_nm matrix. Domain knowledge lives in the extractor and the
binding spec, not in the engine.

---

## P — Physical Channel

**Input:** Continuous waveform (voltage, pressure, displacement,
acceleration, temperature oscillation — any real-valued time series
with a dominant oscillatory component).

**Method:** Hilbert transform via the analytic signal:

```
a(t) = s(t) + i * H[s(t)]
theta(t) = arg(a(t))         — instantaneous phase
A(t) = |a(t)|                — instantaneous amplitude (envelope)
omega(t) = d(unwrap(theta))/dt  — instantaneous frequency
```

The Hilbert transform `H[s]` computes the imaginary part of the
analytic signal. For a narrowband signal (e.g., EEG filtered to the
alpha band 8-12 Hz), this yields a clean, slowly varying phase.
For broadband signals, bandpass filtering before extraction is
essential — the Hilbert transform of a broadband signal produces
a noisy, rapidly varying phase.

**Implementation detail:** `PhysicalExtractor` uses `scipy.signal.hilbert`
on the input signal. The signal is zero-padded to the next power of 2
for FFT efficiency. Phase is unwrapped before frequency computation
to avoid discontinuities, then re-wrapped to `[0, 2pi)` for output.

**Quality metric:** SNR estimate:

```
quality = sig_power / (sig_power + noise_power)
```

Where `sig_power` is the variance of the bandpass-filtered signal and
`noise_power` is the variance of the residual (original minus filtered).
Quality saturates at 1.0. A quality of 0.5 means the signal and noise
have equal power — the phase estimate is unreliable.

**When to use:**
- EEG, MEG, LFP (neural oscillations)
- ECG (heartbeat waveform)
- Accelerometer, gyroscope (mechanical vibration)
- AC voltage/current (power systems)
- Acoustic signals (machinery monitoring)
- Any signal where the observable is a continuous oscillating quantity

**When NOT to use:**
- Event streams (use I-channel instead)
- Discrete state sequences (use S-channel instead)
- Signals with no oscillatory component (e.g., monotonic trends)

**Extractor:** `PhysicalExtractor` in `oscillators.physical`.

**Rust path:** `spo-oscillators::physical` provides `physical_extract()`
via the `spo_kernel` FFI. The Rust implementation uses the same
algorithm but avoids Python overhead for large arrays.

---

## I — Informational Channel

**Input:** Event timestamps (sorted ascending). Each timestamp marks
the occurrence of a discrete event: a neural spike, a network packet
arrival, a heartbeat R-peak, a job completion, a user click.

**Method:** The core idea is that a regular point process has an
inherent frequency (its rate), and the phase between events increases
linearly from 0 to 2pi:

```
IEI_k = t_{k+1} - t_k           — inter-event interval
f_k = 1 / IEI_k                 — instantaneous frequency
omega = 2 * pi * median(f_k)    — angular frequency (robust to outliers)
```

For a given observation time `t` between events `t_k` and `t_{k+1}`:

```
theta(t) = 2 * pi * (t - t_k) / (t_{k+1} - t_k)
```

This is the "ring projection" — it maps the time between consecutive
events to a linear phase on `[0, 2pi)`. At the event itself, phase
resets to 0 (or equivalently, wraps from 2pi to 0).

**Quality metric:** Inverse coefficient of variation:

```
CV = std(IEI) / mean(IEI)
quality = 1 / (1 + CV)
```

Regular events (constant IEI, CV=0) score quality=1.0. Bursty or
irregular events (high CV) score low. A Poisson process (CV=1) scores
quality=0.5 — barely usable.

**When to use:**
- Neural spike trains
- Network packet arrivals
- Heartbeat R-peaks (when only peak times are available, not the full
  waveform — otherwise use P-channel on the ECG signal)
- Job completion events in distributed systems
- User interaction events (clicks, keystrokes)
- Any observable that is a point process

**Implementation detail:** `InformationalExtractor` handles edge cases:
- Fewer than 2 events: returns quality=0 (no phase estimate possible).
- Events with zero interval: treated as simultaneous, IEI floored at
  1e-12 to prevent division by zero.
- Non-sorted input: sorted internally with a warning.

**Extractor:** `InformationalExtractor` in `oscillators.informational`.

**Rust path:** `spo-oscillators::informational` provides `ring_phase()`
and `event_phase()` via FFI.

---

## S — Symbolic Channel

**Input:** Sequence of discrete state indices `s in {0, 1, ..., N-1}`,
observed at discrete time steps. Protocol states, workflow stages,
Markov chain positions, categorical labels.

**Method:** Two modes:

### Ring Mode (default)

Maps discrete state to equally spaced phases on the unit circle:

```
theta = 2 * pi * s / N
```

State 0 maps to phase 0, state 1 to `2pi/N`, etc. This preserves
the circular topology — state N-1 is adjacent to state 0.

Frequency is computed from phase differences between consecutive
observations:

```
omega = (theta_current - theta_previous) / dt
```

where the phase difference is wrapped to `[-pi, pi]` before division.

### Graph Mode

For state machines where not all transitions are equally "distant"
(e.g., a TCP state machine where SYN_SENT → ESTABLISHED is one step
but SYN_SENT → CLOSED is a reset), graph mode normalises the
sequential position along the observed path:

```
theta = 2 * pi * path_position / path_length
```

This requires a graph of valid transitions, provided in the binding
spec `config.adjacency` field.

**Quality metric:** Transition regularity:

| Transition type | Quality |
|----------------|---------|
| Single step to adjacent state | 1.0 |
| Stall (repeated state) | 0.2 |
| Multi-step jump (`|s_new - s_old| > 1`) | `1.0 - jump_size / N` |

Low quality indicates the state machine is behaving unexpectedly
(stalling, skipping states), which makes the phase estimate
unreliable.

**When to use:**
- TCP/protocol state machines
- Workflow/pipeline stages
- Markov chain position tracking
- Categorical variable cycling (e.g., traffic light states)
- Game/simulation state
- Any observable that is a finite state machine with cyclic structure

**Extractor:** `SymbolicExtractor` in `oscillators.symbolic`.

**Rust path:** `spo-oscillators::symbolic` provides `graph_walk_phase()`
via FFI.

---

## Mixed-Channel Systems

A single domain can use all three channels simultaneously. The binding
spec `oscillator_families` dict maps family names to channels:

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

All channels output `PhaseState` with identical fields. The UPDE engine
operates on the unified phase vector — it does not distinguish channels.
Cross-channel coupling (e.g., a P-channel EEG oscillator coupled to an
I-channel spike train) is handled naturally through K_nm entries that
span oscillator families.

### Cross-Channel Coupling Considerations

Coupling between channels of different types requires care:

- **P-P coupling**: natural. Both have continuous phase dynamics.
- **I-I coupling**: natural. Both have event-driven phase.
- **P-I coupling**: the I-channel phase is piecewise linear (resets at
  each event), while the P-channel phase is smooth. The coupling term
  `sin(theta_P - theta_I)` handles this correctly, but the effective
  coupling strength may need to be lower to avoid jitter from I-channel
  resets.
- **S-X coupling**: symbolic phase is quantised. Coupling to continuous
  channels works but introduces discrete jumps in the coupling force.
  Higher `n_states` reduces quantisation effects.

### Initial Phase Extraction

The `extract_initial_phases()` utility extracts `PhaseState` from all
oscillator families in a binding spec given raw signal data:

```python
from scpn_phase_orchestrator.oscillators import extract_initial_phases

states = extract_initial_phases(
    binding_spec=spec,
    signals={"heart_wave": ecg_data, "spike_train": spike_times},
    sample_rates={"heart_wave": 256.0, "spike_train": 30000.0},
)
```

---

## Quality Gating

Extracted phases with `quality < min_quality` (default 0.3) are
down-weighted by `PhaseQualityScorer.downweight_mask()`. The weight
array multiplies into K_nm row-wise, effectively decoupling unreliable
oscillators without removing them from the state vector.

Collapsed oscillators (quality < 0.1 for majority of states) trigger
`detect_collapse()`, which the supervisor interprets as a DEGRADED or
CRITICAL condition depending on the scope of collapse.

### Quality Calibration

The default thresholds (0.3 for gating, 0.1 for collapse) are
empirical. Domain-specific calibration:

1. Run the system on representative data with quality logging enabled.
2. Plot quality distributions per channel.
3. Set `min_quality` at the point where phase estimates become visually
   unreliable (e.g., Hilbert phase no longer tracks the dominant
   oscillation).
4. Set collapse threshold at the point where the majority of oscillators
   produce noise rather than signal.

See `docs/ASSUMPTIONS.md` § Quality Gating for the current defaults
and their provenance.

---

## Channel Selection Guide

| Observable | Channel | Reason |
|-----------|---------|--------|
| EEG voltage | P | Continuous waveform with dominant oscillation |
| Neural spike times | I | Point process, not continuous |
| Heart rate from R-peaks | I | Event timestamps |
| ECG waveform | P | Continuous, use Hilbert on QRS complex |
| Network packet arrivals | I | Event timestamps |
| TCP state transitions | S | Finite state machine |
| Machine vibration | P | Continuous mechanical oscillation |
| Workflow stage | S | Discrete categorical |
| Stock price | P | Continuous (after detrending) |
| Trade events | I | Point process |
| Traffic light state | S | Cyclic discrete states |
| Respiratory waveform | P | Continuous |
| Servo motor position | P | Continuous angular signal |

When in doubt: if the signal is continuous and oscillatory, use P.
If it is a series of event times, use I. If it is a sequence of
discrete states, use S.

---

## References

- **[gabor1946]** D. Gabor (1946). Theory of communication. *J. IEE* 93, 429-457. — Analytic signal and instantaneous phase (P channel).
- **[pikovsky2001]** A. Pikovsky, M. Rosenblum & J. Kurths (2001). *Synchronization: A Universal Concept in Nonlinear Sciences*. Cambridge UP. — Phase extraction from time series, quality metrics.
- **[lachaux1999]** J.-P. Lachaux et al. (1999). Measuring phase synchrony in brain signals. *Human Brain Mapping* 8, 194-208. — PLV definition used in quality gating.
- **[daley2003]** D. J. Daley & D. Vere-Jones (2003). *An Introduction to the Theory of Point Processes*. Springer. — Point process theory underlying I-channel extraction.
