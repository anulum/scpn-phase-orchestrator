# Amplitude Envelope Solver

## 1. Mathematical Formalism

### Sliding-Window RMS Envelope

The amplitude envelope $E(t)$ is computed as the root-mean-square (RMS)
of the amplitude time series $a(t)$ over a sliding window of length $w$:

$$E(t) = \sqrt{\frac{1}{w} \sum_{k=t-w+1}^{t} a(k)^2}$$

The implementation uses a **cumulative sum** approach for $O(T)$ total
computation (independent of $w$):

$$\text{cs}(i) = \sum_{k=0}^{i-1} a(k)^2$$

$$E(t) = \sqrt{\frac{\text{cs}(t+1) - \text{cs}(t-w+1)}{w}}$$

The first $w-1$ samples are padded with the first valid RMS value:

$$E(t) = E(w-1) \quad \text{for } t < w-1$$

This ensures the output has the same length as the input.

### Modulation Depth

The modulation depth $M$ quantifies the dynamic range of the envelope:

$$M = \frac{E_{\max} - E_{\min}}{E_{\max} + E_{\min}} \in [0, 1]$$

- $M = 0$: constant envelope (no modulation)
- $M = 1$: full modulation (envelope reaches zero)
- $M = 0.5$: moderate modulation (3:1 peak-to-trough ratio)

This is the standard AM modulation index used in communications theory
and neuroscience (amplitude modulation of neural oscillations).

### 2-D Case

For multi-channel input $(T, N)$, the RMS is computed independently per
channel (column-wise). The cumulative sum operates along axis 0.

---

## 2. Theoretical Context

### Amplitude Dynamics in Coupled Oscillators

In the Kuramoto model, each oscillator has a fixed unit amplitude.
Real oscillators (neural, electronic, mechanical) have **variable amplitude**
governed by dynamics separate from the phase:

$$\dot{r}_i = (\mu - r_i^2) r_i + \sum_j K^r_{ij} (r_j - r_i)$$

(Stuart-Landau amplitude equation, where $\mu$ controls the Hopf bifurcation.)

The `extract_envelope` function extracts the amplitude time course from
either direct amplitude measurements or from the analytic signal of a
phase-reconstructed time series. It acts as a **low-pass filter** on the
amplitude dynamics, smoothing out fast oscillations and revealing the
slow envelope modulation.

### Applications

- **Sleep staging:** Slow-wave sleep (N3) exhibits high-amplitude, low-frequency
  cortical oscillations. The envelope tracks the waxing and waning of
  sleep spindles (Berry et al. 2012).
- **Entrainment verification:** Successful auditory entrainment shows a stable
  envelope at the stimulus frequency (Nozaradan et al. 2011).
- **Amplitude-phase coupling:** The envelope of high-frequency activity
  (gamma, 30-100 Hz) is modulated by the phase of low-frequency activity
  (theta, 4-8 Hz) — this is the PAC phenomenon (Canolty et al. 2006).
- **Stuart-Landau dynamics:** The `StuartLandauEngine` outputs amplitude
  trajectories that can be fed directly to `extract_envelope`.

### Relationship to Hilbert Envelope

The Hilbert transform provides an instantaneous amplitude envelope via
$|z(t)| = |x(t) + i \hat{x}(t)|$ where $\hat{x}$ is the Hilbert
transform. The RMS envelope is a smoothed version: for window $w = 1$,
$E(t) = |a(t)|$ (pointwise). For $w > 1$, it averages over the window,
reducing noise at the cost of temporal resolution.

The Hilbert envelope is available via `autotune.phase_extract.extract_phases()`.

### Signal Processing Foundations

The RMS envelope is a special case of the **moving average** filter family.
For a signal $x(t)$, the moving RMS with window $w$ is:

$$\text{RMS}_w(t) = \left(\frac{1}{w} \sum_{k=0}^{w-1} x(t-k)^2\right)^{1/2}$$

This is equivalent to:
1. Square the signal: $y(t) = x(t)^2$
2. Apply a boxcar (rectangular) moving average: $\bar{y}(t) = \frac{1}{w}\sum y(t-k)$
3. Take the square root: $\text{RMS}(t) = \sqrt{\bar{y}(t)}$

The boxcar filter has transfer function:

$$H(f) = \frac{\sin(\pi f w)}{\pi f w} \cdot e^{-i\pi f(w-1)}$$

The first null is at $f = 1/w$, so the RMS envelope suppresses
oscillations faster than $1/w$ Hz (in samples). For a 100 Hz sampled
signal with $w = 50$: frequencies above 2 Hz are attenuated.

### Choice of Window Size

The window size $w$ controls the trade-off between:
- **Temporal resolution:** Small $w$ tracks fast amplitude changes
  but includes more noise
- **Smoothness:** Large $w$ gives a clean envelope but blurs fast
  transients (e.g. spindle onset)

Practical guidelines:
- **Sleep staging:** $w \approx$ 1-2 seconds (100-200 samples at 100 Hz)
- **PAC analysis:** $w \approx$ 2-3 cycles of the low-frequency rhythm
- **Entrainment:** $w \approx$ 5-10 stimulus periods
- **General monitoring:** $w = T/20$ as a starting point

### Cumulative Sum Implementation

The $O(T)$ cumulative sum approach avoids the naive $O(T \cdot w)$
sliding window. It computes:

$$\text{cs}(0) = 0, \quad \text{cs}(i) = \text{cs}(i-1) + a(i-1)^2$$

Then for any window position $t$:

$$\sum_{k=t}^{t+w-1} a(k)^2 = \text{cs}(t+w) - \text{cs}(t)$$

This is numerically stable for $T < 10^7$ with float64. For very long
signals ($T > 10^7$), compensated summation (Kahan) may be needed to
avoid catastrophic cancellation, but this is not implemented as such
signal lengths are rare in oscillator monitoring.

---

## 3. Pipeline Position

```
StuartLandauEngine.step() ──→ [θ, r]
                                   │
                                   │ r = amplitudes
                                   ↓
     ┌──── extract_envelope(r_history, window) ────┐
     │                                              │
     │  Input:  (T,) or (T, N) amplitude series     │
     │  Param:  window (RMS smoothing length)       │
     │  Method: cumulative sum (Rust or NumPy)      │
     │  Output: (T,) or (T, N) envelope             │
     │                                              │
     └──────────────────────────────────────────────┘
                                   │
                                   ↓
     ┌──── envelope_modulation_depth(envelope) ─────┐
     │                                               │
     │  Output: M ∈ [0, 1] (modulation index)       │
     │                                               │
     └───────────────────────────────────────────────┘
                                   │
                                   ↓
              EVSMonitor / PAC analysis / sleep staging
```

### Upstream Sources

| Source | Produces | Type |
|--------|----------|------|
| `StuartLandauEngine` | Amplitude trajectory $r(t)$ | `NDArray (T,N)` |
| `phase_extract.extract_phases()` | Hilbert amplitudes | `NDArray (T,)` |
| Raw sensor data | Measured amplitudes | `NDArray (T,)` or `(T,N)` |

### Integration Pattern: Real-Time Monitoring

In the full SPO pipeline, the envelope is computed at each monitoring
interval (every $M$ integration steps):

```python
# Inside the main loop
for step in range(total_steps):
    phases, amplitudes = engine.step(...)
    amplitude_buffer.append(amplitudes)

    if step % monitor_interval == 0:
        history = np.array(amplitude_buffer[-window:])
        envelope = extract_envelope(history[:, 0], window=len(history) // 5)
        M = envelope_modulation_depth(envelope)

        if M > 0.8:
            supervisor.flag_regime("unstable_amplitude")
```

This streaming pattern ensures that the envelope computation does not
block the integration loop (the Rust path takes < 0.01 ms for typical
buffer sizes of 100-1000 samples).

### Downstream Consumers

| Consumer | Uses | Purpose |
|----------|------|---------|
| `EVSMonitor` | Envelope stability | Entrainment verification |
| `PAC` analysis | High-freq envelope | Phase-amplitude coupling |
| `sleep_staging` | Envelope features | AASM classification |
| Visualisation | Smoothed amplitude | Time series plots |

---

## 4. Features

- **$O(T)$ computation** via cumulative sum (independent of window size)
- **1-D and 2-D input** — single channel or multi-channel
- **Front-padding** with first valid value (output length = input length)
- **Rust FFI acceleration** for 1-D case via `spo_kernel`
- **Modulation depth** as a single-number summary of envelope dynamics
- **Non-negative output** guaranteed (RMS of real values)
- **Window validation** — raises `ValueError` for `window < 1`
- **Empty input handling** — returns empty array for empty input
- **EnvelopeState dataclass** for snapshot statistics

---

## 5. Usage Examples

### Basic: Extract Envelope from Amplitude Trajectory

```python
import numpy as np
from scpn_phase_orchestrator.upde.envelope import extract_envelope, envelope_modulation_depth

# Simulated AM signal: carrier modulated by slow envelope
T = 1000
t = np.arange(T) * 0.01
carrier = np.sin(2 * np.pi * 10 * t)           # 10 Hz carrier
modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)  # 0.5 Hz modulator
signal = modulation * carrier
amplitudes = np.abs(signal)

envelope = extract_envelope(amplitudes, window=50)
M = envelope_modulation_depth(envelope)
print(f"Modulation depth: {M:.4f}")  # Expected: ~1.0 (full modulation)
```

### Stuart-Landau Amplitude Tracking

```python
import numpy as np
from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine
from scpn_phase_orchestrator.upde.envelope import extract_envelope

N = 8
engine = StuartLandauEngine(N, dt=0.01, mu=1.0)

phases = np.random.default_rng(42).uniform(0, 2 * np.pi, N)
amplitudes_init = np.ones(N) * 0.5
knm = np.full((N, N), 0.3); np.fill_diagonal(knm, 0.0)

# Collect amplitude history
n_steps = 2000
amp_history = np.zeros((n_steps, N))
state = np.concatenate([phases, amplitudes_init])

for t in range(n_steps):
    state = engine.step(state, np.ones(N), knm, 0.0, 0.0, np.zeros((N, N)))
    amp_history[t] = state[N:]  # amplitudes are second half of state

# Extract per-channel envelope
envelope = extract_envelope(amp_history, window=100)
print(f"Envelope shape: {envelope.shape}")  # (2000, 8)
print(f"Mean amplitude: {np.mean(envelope[-100:]):.4f}")
```

### Multi-Channel with Modulation Depth

```python
import numpy as np
from scpn_phase_orchestrator.upde.envelope import (
    extract_envelope, envelope_modulation_depth,
)

# 4-channel amplitude data
T, N = 500, 4
rng = np.random.default_rng(42)
amplitudes = np.abs(rng.standard_normal((T, N)))

envelope = extract_envelope(amplitudes, window=20)
M = envelope_modulation_depth(envelope)
print(f"Overall modulation depth: {M:.4f}")

# Per-channel
for ch in range(N):
    env_ch = extract_envelope(amplitudes[:, ch], window=20)
    M_ch = envelope_modulation_depth(env_ch)
    print(f"  Channel {ch}: M = {M_ch:.4f}")
```

### Window Size Selection

```python
import numpy as np
from scpn_phase_orchestrator.upde.envelope import extract_envelope

T = 1000
signal = np.sin(np.arange(T) * 0.1) + 0.3 * np.random.default_rng(42).standard_normal(T)
amplitudes = np.abs(signal)

for w in [1, 10, 50, 100, 200]:
    env = extract_envelope(amplitudes, window=w)
    print(f"window={w:4d}: env_std={np.std(env):.4f}, env_mean={np.mean(env):.4f}")
# Larger window → smoother envelope, less temporal resolution
```

---

## 6. Technical Reference

### Functions

::: scpn_phase_orchestrator.upde.envelope

### Dataclass: EnvelopeState

```python
@dataclass(frozen=True)
class EnvelopeState:
    mean_amplitude: float      # Mean of the envelope
    amplitude_spread: float    # Std of the envelope
    modulation_depth: float    # (max - min) / (max + min)
    subcritical_count: int     # Number of channels below threshold
```

### Rust Engine Functions

```rust
// 1-D sliding-window RMS
pub fn extract_envelope(amplitudes: &[f64], window: usize) -> Vec<f64>

// Modulation depth: (max - min) / (max + min)
pub fn envelope_modulation_depth(envelope: &[f64]) -> f64
```

### Auto-Select Logic

- `extract_envelope`: Rust for 1-D input, Python for 2-D (Rust handles
  only 1-D; 2-D falls back to NumPy cumsum per column)
- `envelope_modulation_depth`: Rust for all inputs (flattened)

---

## 7. Performance Benchmarks

Measured on Intel Core i5-11600K @ 3.90 GHz, 32 GB DDR4-2400.
Window = 50 samples. Averaged over 100-1000 iterations.

### extract_envelope (1-D)

| T (samples) | Python (ms) | Rust (ms) | Speedup |
|-------------|-------------|-----------|---------|
| 100 | 0.0681 | 0.0029 | **23.2x** |
| 1,000 | 0.0845 | 0.0064 | **13.1x** |
| 10,000 | 0.1789 | 0.0759 | **2.4x** |
| 100,000 | 7.3790 | 4.5608 | **1.6x** |

### envelope_modulation_depth

| T (samples) | Python (ms) | Rust (ms) | Speedup |
|-------------|-------------|-----------|---------|
| 100 | 0.0139 | 0.0014 | **9.9x** |
| 1,000 | 0.0150 | 0.0039 | **3.8x** |
| 10,000 | 0.0206 | 0.0408 | 0.5x |

### Scaling Analysis

`extract_envelope` is $O(T)$ for both paths (cumulative sum). The Rust
speedup is largest at small $T$ (23x at $T=100$) where Python function
call overhead dominates, and decreases at large $T$ where NumPy's
C-level cumsum is competitive.

`envelope_modulation_depth` is $O(T)$ (single pass for min/max). At
$T > 10{,}000$, NumPy's vectorised `np.max/np.min` outperforms Rust
due to SIMD optimisation in NumPy's C backend.

### Test Coverage

- **Rust tests:** 10 (envelope module in spo-engine)
  - Constant signal, modulation depth constant, modulation depth range,
    empty signal, window one, output length, window equals length,
    window larger than length, sinusoidal envelope, modulation depth zeros
- **Python tests:** 16 (`tests/test_envelope.py`)
  - Empty signal, single value, output length, RMS values non-negative,
    window 1, AM signal tracking, 2-D per-column, window larger,
    constant returns zero, full modulation, partial modulation,
    empty/zero returns, range check, frozen dataclass, pipeline end-to-end,
    performance benchmark
- **Source lines:** 168 (Rust) + 75 (Python) = 243 total

---

## 8. Citations

1. **Berry, R. B., Brooks, R., Gamaldo, C. E., et al.** (2012).
   "The AASM Manual for the Scoring of Sleep and Associated Events:
   Rules, Terminology and Technical Specifications." Version 2.0.
   American Academy of Sleep Medicine.

2. **Nozaradan, S., Peretz, I., Missal, M., & Mouraux, A.** (2011).
   "Tagging the neuronal entrainment to beat and meter."
   *Journal of Neuroscience* 31(28):10234-10240.
   DOI: [10.1523/JNEUROSCI.0411-11.2011](https://doi.org/10.1523/JNEUROSCI.0411-11.2011)

3. **Canolty, R. T., Edwards, E., Dalal, S. S., et al.** (2006).
   "High gamma power is phase-locked to theta oscillations in human
   neocortex."
   *Science* 313(5793):1626-1628.
   DOI: [10.1126/science.1128115](https://doi.org/10.1126/science.1128115)

4. **Tort, A. B. L., Komorowski, R., Eichenbaum, H., & Kopell, N.** (2010).
   "Measuring phase-amplitude coupling between neuronal oscillations
   of different frequencies."
   *Journal of Neurophysiology* 104(2):1195-1210.
   DOI: [10.1152/jn.00106.2010](https://doi.org/10.1152/jn.00106.2010)

5. **Pikovsky, A., Rosenblum, M., & Kurths, J.** (2001).
   *Synchronization: A Universal Concept in Nonlinear Sciences.*
   Cambridge University Press.
   DOI: [10.1017/CBO9780511755743](https://doi.org/10.1017/CBO9780511755743)

6. **Boashash, B.** (1992).
   "Estimating and interpreting the instantaneous frequency of a signal."
   *Proceedings of the IEEE* 80(4):520-568.
   DOI: [10.1109/5.135376](https://doi.org/10.1109/5.135376)

7. **Cohen, M. X.** (2014).
   *Analyzing Neural Time Series Data: Theory and Practice.*
   MIT Press.
   — Chapter 13: Hilbert transform and analytic signal.

8. **Oppenheim, A. V. & Willsky, A. S.** (1997).
   *Signals and Systems.* 2nd ed. Prentice Hall.
   — Chapter 8: modulation and sampling.

---

## Edge Cases and Limitations

### Window Larger Than Signal

When $w > T$, no valid RMS value can be computed. The Rust implementation
returns a vector of zeros; the Python implementation returns the input
padded appropriately. Both paths return an array of length $T$.

### All-Zero Input

$\text{RMS}(0, 0, \ldots, 0) = 0$. Modulation depth of an all-zero
envelope is 0.0 (denominator $E_{\max} + E_{\min} = 0$).

### Negative Amplitudes

The function accepts negative values (it squares them). The output is
always non-negative. If input represents signed signals (not amplitudes),
apply `np.abs()` first for meaningful results.

### Very Large Window

A window equal to the signal length returns a single repeated value
(the global RMS). This is mathematically correct but loses all temporal
information. For meaningful envelopes, $w \ll T$ (typically $w < T/10$).

### 2-D Input: No Rust Path

The Rust FFI only handles 1-D input. For 2-D $(T, N)$ input,
`extract_envelope` uses the Python (NumPy) path. Per-channel Rust
acceleration is possible by calling the Rust function in a loop, but
the overhead of $N$ FFI calls may negate the benefit for small $T$.

---

## Appendix A: Relationship to Other Envelope Methods

| Method | Module | Complexity | Temporal Resolution | Smoothness |
|--------|--------|------------|--------------------|-----------:|
| RMS envelope | `envelope.extract_envelope` | $O(T)$ | $w$ samples | High |
| Hilbert envelope | `phase_extract.extract_phases` | $O(T \log T)$ | Instantaneous | Low |
| Peak detection | (manual) | $O(T)$ | Variable | Medium |
| Wavelet envelope | (scipy) | $O(T \log T)$ | Scale-dependent | Scale-dependent |

The RMS envelope is preferred for:
- Fast computation ($O(T)$ vs $O(T \log T)$)
- Predictable smoothing (controlled by $w$)
- No edge artefacts (unlike Hilbert, which has Gibbs phenomena at boundaries)

The Hilbert envelope is preferred for:
- Instantaneous amplitude (no smoothing delay)
- Consistency with analytic signal theory
- Phase-amplitude coupling analysis (PAC)

## Appendix B: Numerical Precision

### Cumulative Sum Stability

For the cumulative sum $\text{cs}(i) = \sum_{k=0}^{i-1} a(k)^2$:

- **Float64 precision:** $\sim 15.9$ significant digits
- **Worst case:** If all $a(k)^2 = 1$ and $T = 10^7$, then
  $\text{cs}(T) = 10^7$, and the subtraction $\text{cs}(t+w) - \text{cs}(t)$
  loses $\sim 7$ digits, leaving $\sim 9$ digits of precision.
  This is adequate for all practical applications.
- **Catastrophic cancellation:** Only occurs when $w \ll T$ and
  $\text{cs}(t+w) \approx \text{cs}(t)$, which happens when
  $a(k) \approx 0$ for all $k$ in the window — in which case
  $E(t) \approx 0$ and the relative error is irrelevant.

### Rust vs Python Numerical Parity

Both implementations use the same cumulative sum algorithm with
float64 arithmetic. The results are bitwise identical for 1-D input
(verified in tests). The 2-D Python path uses `np.cumsum(axis=0)`
which may differ by $O(\epsilon)$ from a manual loop due to
summation order.

## Appendix C: EnvelopeState Usage in the Supervisor

```python
from scpn_phase_orchestrator.upde.envelope import (
    extract_envelope, envelope_modulation_depth, EnvelopeState,
)
import numpy as np

# From Stuart-Landau trajectory
amplitudes = np.random.default_rng(42).uniform(0.5, 1.5, 1000)
envelope = extract_envelope(amplitudes, window=50)

state = EnvelopeState(
    mean_amplitude=float(np.mean(envelope)),
    amplitude_spread=float(np.std(envelope)),
    modulation_depth=envelope_modulation_depth(envelope),
    subcritical_count=int(np.sum(envelope < 0.3)),
)

print(f"Mean: {state.mean_amplitude:.4f}")
print(f"Spread: {state.amplitude_spread:.4f}")
print(f"Modulation: {state.modulation_depth:.4f}")
print(f"Subcritical: {state.subcritical_count}")

# The supervisor uses these fields for regime detection:
# - high modulation_depth → unstable amplitude → degraded regime
# - high subcritical_count → amplitude collapse → critical regime
```

## Appendix D: Common Pitfalls

### Pitfall 1: Window Too Large

If `window >= len(amplitudes)`, only one RMS value is computed and
replicated across the entire output. The resulting envelope is a flat
line — useless for temporal analysis. Always ensure `window < T / 2`.

### Pitfall 2: Mixing Phases and Amplitudes

`extract_envelope` expects **amplitudes** (non-negative), not phases.
Passing raw phases (which wrap at $2\pi$) produces meaningless results.
If you have phase trajectories, first convert to amplitudes via
`np.abs(np.exp(1j * phases))` (which is always 1.0 for unit-amplitude
oscillators) or use the Stuart-Landau amplitude output.

### Pitfall 3: Forgetting the Padding

The first `window - 1` samples are padded with the first valid RMS
value. This means the envelope at $t < w-1$ is constant and does not
reflect the actual dynamics. For accurate analysis, discard the first
`window` samples or account for the padding in your statistics.

### Pitfall 4: Modulation Depth of a Single Sample

`envelope_modulation_depth` on a single-element array returns 0.0
(max = min). This is correct but may be surprising. Ensure at least
2 distinct values for a meaningful modulation depth.
