# Phase Extraction via Hilbert Transform

## 1. Mathematical Formalism

### The Analytic Signal

Given a real-valued signal $x(t)$, the analytic signal is:

$$z(t) = x(t) + i\mathcal{H}[x](t)$$

where $\mathcal{H}$ denotes the Hilbert transform:

$$\mathcal{H}[x](t) = \frac{1}{\pi} \text{P.V.} \int_{-\infty}^{\infty} \frac{x(\tau)}{t - \tau} d\tau$$

The Hilbert transform shifts each frequency component by $-\pi/2$
(positive frequencies) or $+\pi/2$ (negative frequencies), creating
a quadrature signal.

### Instantaneous Phase

$$\theta(t) = \arg(z(t)) = \arctan\left(\frac{\mathcal{H}[x](t)}{x(t)}\right) \bmod 2\pi$$

The instantaneous phase is the angle of the analytic signal in the
complex plane. It represents the position of the oscillator on the
unit circle at each time point.

### Instantaneous Amplitude (Envelope)

$$A(t) = |z(t)| = \sqrt{x(t)^2 + \mathcal{H}[x](t)^2}$$

The amplitude envelope measures the modulation of the oscillation.
For a pure sinusoid, $A(t)$ is constant. For modulated signals
(AM, FM), $A(t)$ tracks the modulation.

### Instantaneous Frequency

$$f(t) = \frac{1}{2\pi} \frac{d}{dt}\text{unwrap}(\theta(t))$$

Computed via finite differences on the unwrapped phase:

$$f(t_k) \approx \frac{\text{unwrap}(\theta(t_{k+1})) - \text{unwrap}(\theta(t_k))}{2\pi \Delta t}$$

The unwrapping removes $2\pi$ jumps before differentiation.

### Dominant Frequency

$$f_{\text{dom}} = \arg\max_f |\hat{X}(f)|, \quad f > 0$$

where $\hat{X}(f) = \text{FFT}(x)$ is the discrete Fourier transform.
The DC component ($f = 0$) is excluded.

### FFT-Based Bandpass Filter

For narrowband phase extraction, an optional bandpass filter zeroes
FFT coefficients outside $[f_{\text{low}}, f_{\text{high}}]$:

$$\hat{X}_{\text{filtered}}(f) = \begin{cases} \hat{X}(f) & f_{\text{low}} \leq f \leq f_{\text{high}} \\ 0 & \text{otherwise} \end{cases}$$

followed by inverse FFT. This is a zero-phase (non-causal) filter
suitable for offline analysis.

---

## 2. Theoretical Context

### Why Hilbert Transform for Phase?

The Hilbert transform is the standard method for extracting
instantaneous phase from narrowband signals (Pikovsky et al. 2001).
Alternatives include:

| Method | Advantage | Limitation |
|--------|-----------|------------|
| **Hilbert transform** | Well-defined for narrowband | Ambiguous for broadband |
| Wavelet transform | Time-frequency resolution | Spectral leakage |
| Short-time Fourier transform | Simple | Fixed window size |
| Protophase (direct angle) | No transform needed | Only for oscillatory signals |

The Hilbert transform is preferred for SCPN because the oscillators
are assumed to be narrowband (quasi-sinusoidal) after bandpass
filtering.

### Bedrosian Theorem

The Hilbert transform gives a physically meaningful instantaneous
phase only when the signal is "narrowband" — the amplitude envelope
$A(t)$ varies slowly compared to the carrier frequency. Formally,
the Bedrosian theorem (1963) states:

$$\mathcal{H}[A(t) \cos(\phi(t))] = A(t) \sin(\phi(t))$$

if and only if the spectrum of $A(t)$ and the spectrum of
$\cos(\phi(t))$ do not overlap.

### Computational Implementation

The SciPy `hilbert` function computes the analytic signal via:
1. FFT of the input signal: $\hat{X}_k = \text{FFT}(x)$
2. Construct one-sided spectrum:
   $\hat{Z}_k = \begin{cases} \hat{X}_0 & k = 0 \\ 2\hat{X}_k & 1 \leq k < N/2 \\ \hat{X}_{N/2} & k = N/2 \\ 0 & k > N/2 \end{cases}$
3. Inverse FFT: $z(t) = \text{IFFT}(\hat{Z})$

This is an $O(N \log N)$ operation using the Cooley-Tukey FFT
algorithm. The one-sided spectrum construction is equivalent to
applying the Hilbert transform in the frequency domain.

### Phase Extraction Quality Metrics

The quality of the extracted phase depends on:

1. **Signal-to-Noise Ratio (SNR):** Phase estimates degrade for
   SNR < 10 dB. Below 0 dB, the instantaneous frequency becomes
   meaningless.

2. **Bandwidth relative to centre frequency ($B/f_c$):** The
   Bedrosian condition requires $B/f_c < 1$. For alpha-band EEG
   ($f_c = 10$ Hz, $B = 4$ Hz), $B/f_c = 0.4$ — acceptable.

3. **Signal length:** FFT resolution $\Delta f = f_s/N$. For 1-second
   windows at 250 Hz, $\Delta f = 0.25$ Hz — sufficient for most
   neural oscillations.

### Applications in Neuroscience

- **EEG phase extraction:** Alpha (8-12 Hz), theta (4-8 Hz), gamma
  (30-80 Hz) bands extracted for phase-amplitude coupling analysis
- **tACS phase tracking:** Online phase extraction for closed-loop
  transcranial stimulation
- **Brain-computer interfaces:** SSVEP phase decoding for user intent
- **Sleep staging:** Phase coherence (ITPC) across trials drives the
  EVS entrainment score

### Applications in Engineering

- **Vibration analysis:** Instantaneous frequency tracks machine
  health (bearing faults, gear mesh)
- **Power grid monitoring:** Phase extraction from voltage waveforms
  for synchronisation monitoring
- **Acoustic entrainment:** Fluctara uses phase extraction to verify
  that binaural beats entrain endogenous oscillations

### Historical Context

- **Gabor, D.** (1946): "Theory of communication." Introduced the
  analytic signal and instantaneous frequency for communication
  theory.
- **Bedrosian, E.** (1963): "A product theorem for Hilbert transforms."
  Established conditions for meaningful phase extraction.
- **Pikovsky, A., Rosenblum, M., & Kurths, J.** (2001):
  *Synchronization.* Chapter 6: Phase extraction from data.
- **Boashash, B.** (1992): "Estimating and interpreting the
  instantaneous frequency of a signal." Comprehensive review.
- **Marple, S. L.** (1999): "Computing the discrete-time 'analytic
  signal' via FFT." The algorithm used in SciPy.

### Rust Path (Disabled)

The Rust implementation uses a naive DFT ($O(N^2)$) instead of FFT
($O(N \log N)$). Benchmarking showed it is **60x slower** than the
Python/SciPy FFT path. The Rust module exists for correctness testing
and for environments without SciPy.

---

## 3. Pipeline Position

```
 Raw time series (EEG, audio, sensor)
                    │
                    ↓
 ┌── extract_phases(signal, fs, bandpass) ────────┐
 │                                                 │
 │  Step 1: Optional bandpass filter (FFT-based)  │
 │  Step 2: Hilbert transform → analytic signal   │
 │  Step 3: Phase = arg(z) mod 2π                 │
 │  Step 4: Amplitude = |z|                        │
 │  Step 5: Inst. frequency = d(unwrap(θ))/dt     │
 │  Step 6: Dominant freq = argmax(|FFT|)          │
 │                                                 │
 │  Output: PhaseResult(phases, amplitudes,        │
 │          inst_freq, dominant_freq)              │
 │                                                 │
 └──────────────────┬──────────────────────────────┘
                    │
                    ↓
 Oscillator initialisation: phases → UPDEEngine
 Frequency identification: dominant_freq → omegas
 EVS monitoring: phases → EVSMonitor.evaluate()
```

### Input Contracts

| Parameter | Type | Shape | Range | Meaning |
|-----------|------|-------|-------|---------|
| `signal` | `NDArray[float64]` | `(N,)` | any | 1-D real time series |
| `fs` | `float` | scalar | $> 0$ | Sampling frequency (Hz) |
| `bandpass` | `tuple[float, float] \| None` | — | $[0, f_s/2]$ | Optional bandpass (Hz) |

### Output Contract

```python
@dataclass
class PhaseResult:
    phases: NDArray          # (N,), in [0, 2π)
    amplitudes: NDArray      # (N,), ≥ 0
    instantaneous_freq: NDArray  # (N,), Hz
    dominant_freq: float     # Hz, > 0
```

---

## 4. Features

- **Hilbert transform** via SciPy FFT — $O(N \log N)$
- **Instantaneous phase** — position on the unit circle per sample
- **Instantaneous amplitude** — envelope of the oscillation
- **Instantaneous frequency** — phase derivative, in Hz
- **Dominant frequency** — FFT-based peak detection
- **Optional bandpass filter** — FFT-based zero-phase filtering
- **Phase wrapping** — output in $[0, 2\pi)$
- **Minimum length check** — requires $\geq 4$ samples
- **Rust engine available** — disabled (60x slower, naive DFT)

---

## 5. Usage Examples

### Basic: Extract Phase from Sine Wave

```python
import numpy as np
from scpn_phase_orchestrator.autotune.phase_extract import extract_phases

fs = 1000.0  # 1 kHz sampling
t = np.arange(1000) / fs
signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine

result = extract_phases(signal, fs)
print(f"Dominant freq: {result.dominant_freq:.1f} Hz")  # 10.0
print(f"Phase at t=0: {result.phases[0]:.4f}")
print(f"Amplitude mean: {result.amplitudes.mean():.4f}")  # ≈ 1.0
```

### With Bandpass Filter

```python
import numpy as np
from scpn_phase_orchestrator.autotune.phase_extract import extract_phases

# Multifrequency signal: 10 Hz + 40 Hz
fs = 1000.0
t = np.arange(2000) / fs
signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 40 * t)

# Extract phase at 10 Hz only
result = extract_phases(signal, fs, bandpass=(8.0, 12.0))
print(f"Dominant freq (filtered): {result.dominant_freq:.1f} Hz")
```

### Feed into UPDEEngine

```python
import numpy as np
from scpn_phase_orchestrator.autotune.phase_extract import extract_phases
from scpn_phase_orchestrator.upde.engine import UPDEEngine

# Extract phases from multiple channels
signals = np.random.default_rng(42).normal(0, 1, (8, 1000))
fs = 250.0
phases_init = np.zeros(8)
omegas = np.zeros(8)

for ch in range(8):
    result = extract_phases(signals[ch], fs, bandpass=(8, 12))
    phases_init[ch] = result.phases[-1]  # Last phase value
    omegas[ch] = result.dominant_freq * 2 * np.pi  # Convert to rad/s

eng = UPDEEngine(8, dt=1.0/fs)
# ... simulate with extracted initial conditions
```

### Amplitude Modulation Detection

```python
import numpy as np
from scpn_phase_orchestrator.autotune.phase_extract import extract_phases

fs = 1000.0
t = np.arange(5000) / fs
# AM signal: carrier 40 Hz, modulator 4 Hz
carrier = np.sin(2 * np.pi * 40 * t)
modulator = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)
signal = carrier * modulator

result = extract_phases(signal, fs, bandpass=(35, 45))
# Amplitude should track the 4 Hz modulation
print(f"Amplitude range: [{result.amplitudes.min():.3f}, {result.amplitudes.max():.3f}]")
```

---

## 6. Technical Reference

### Function: extract_phases

::: scpn_phase_orchestrator.autotune.phase_extract

### Internal: _bandpass_filter

```python
def _bandpass_filter(x: NDArray, fs: float, low: float, high: float) -> NDArray
```

FFT-based zero-phase bandpass. Zeroes all frequency bins outside
$[f_{\text{low}}, f_{\text{high}}]$, then inverse FFT. Non-causal —
suitable for offline analysis only.

### Rust Engine Functions (Disabled)

```rust
pub fn compute_analytic_signal(signal: &[f64]) -> Vec<f64>  // naive DFT
pub fn extract_phase_amplitude(analytic: &[f64], n: usize) -> (Vec<f64>, Vec<f64>)
pub fn instantaneous_frequency(analytic: &[f64], n: usize, fs: f64) -> Vec<f64>
pub fn find_dominant_freq(signal: &[f64], fs: f64) -> f64
```

All functions use naive $O(N^2)$ DFT. Disabled by default.

### Auto-Select: DISABLED

```python
# Rust path disabled: 60x slower than SciPy FFT
# _HAS_RUST = False (hardcoded)
```

---

## 7. Performance Benchmarks

Measured on Intel Core i5-11600K @ 3.90 GHz, 32 GB DDR4-2400.
Python (SciPy hilbert + numpy FFT) only.

| Signal Length | Python (ms) |
|---------------|-------------|
| 100 | 0.227 |
| 500 | 0.256 |
| 1,000 | 0.305 |

### Why Rust is 60x Slower

The Rust implementation computes the DFT as a naive double sum:

$$\hat{X}_k = \sum_{n=0}^{N-1} x_n \cdot e^{-2\pi i k n / N}$$

This is $O(N^2)$ versus the Cooley-Tukey FFT at $O(N \log N)$.
For $N = 1000$: $10^6$ vs $10^4$ operations — approximately 100x
difference, partially offset by Rust's lower per-operation cost.

To achieve competitive Rust performance, the implementation would
need an FFT library (e.g., `rustfft`, `fftw-sys`). This is on
the roadmap.

### Memory Usage

- Analytic signal: $N$ complex values (16 bytes each)
- FFT workspace: $N$ complex values
- Total: ~32 KB for $N = 1000$

### Test Coverage

- **Rust tests:** 6 (phase_extract module in spo-engine)
- **Python tests:** 9 (`tests/test_phase_extract.py`)
- **Source lines:** 209 (Rust) + 79 (Python) = 288 total

---

## 8. Citations

1. **Gabor, D.** (1946).
   "Theory of communication. Part 1: The analysis of information."
   *Journal of the Institution of Electrical Engineers* 93(26):429-441.
   DOI: [10.1049/ji-3-2.1946.0074](https://doi.org/10.1049/ji-3-2.1946.0074)

2. **Bedrosian, E.** (1963).
   "A product theorem for Hilbert transforms."
   *Proceedings of the IEEE* 51(5):868-869.
   DOI: [10.1109/PROC.1963.2308](https://doi.org/10.1109/PROC.1963.2308)

3. **Pikovsky, A., Rosenblum, M., & Kurths, J.** (2001).
   *Synchronization: A Universal Concept in Nonlinear Sciences.*
   Cambridge University Press. ISBN: 978-0-521-59285-7.

4. **Boashash, B.** (1992).
   "Estimating and interpreting the instantaneous frequency of a
   signal."
   *Proceedings of the IEEE* 80(4):520-568.
   DOI: [10.1109/5.135376](https://doi.org/10.1109/5.135376)

5. **Marple, S. L.** (1999).
   "Computing the discrete-time 'analytic signal' via FFT."
   *IEEE Transactions on Signal Processing* 47(9):2600-2603.
   DOI: [10.1109/78.782222](https://doi.org/10.1109/78.782222)

6. **Cohen, L.** (1995).
   *Time-Frequency Analysis.*
   Prentice Hall. ISBN: 978-0-13-594532-2.

7. **Feldman, M.** (2011).
   *Hilbert Transform Applications in Mechanical Vibration.*
   John Wiley & Sons. ISBN: 978-0-470-97827-6.

8. **Huang, N. E. et al.** (1998).
   "The empirical mode decomposition and the Hilbert spectrum for
   nonlinear and non-stationary time series analysis."
   *Proceedings of the Royal Society A* 454(1971):903-995.
   DOI: [10.1098/rspa.1998.0193](https://doi.org/10.1098/rspa.1998.0193)

---

## Edge Cases and Limitations

### Signal Too Short (< 4 samples)

Raises `ValueError`. The Hilbert transform needs at least 4 points
for meaningful results (FFT with at least 2 positive frequency bins).

### Broadband Signals (Violation of Bedrosian Condition)

For broadband signals, the instantaneous phase is not well-defined.
The extracted phase oscillates rapidly between components. Always
bandpass filter before phase extraction for multifrequency signals.

### DC Offset

A signal with large DC offset will have its dominant frequency at
0 Hz. The `extract_phases` function excludes DC by searching
`freqs[1:]` for the dominant frequency. However, the instantaneous
phase and amplitude are affected by DC.

**Solution:** Remove the mean before extraction:
`signal = signal - np.mean(signal)`.

### Constant Signal

A constant signal ($x(t) = c$) has zero Hilbert transform, giving
$z(t) = c + 0i$. The phase is 0 or $\pi$ depending on the sign
of $c$, and the amplitude is $|c|$. The dominant frequency is
undefined (FFT peak at DC).

---

## Troubleshooting

### Issue: Phase Jumps Between 0 and 2π

**Diagnosis:** This is correct behaviour — phase wrapping. Use
`np.unwrap` for continuous phase when computing derivatives or
visualising.

### Issue: Instantaneous Frequency Negative

**Diagnosis:** Can occur at phase discontinuities or for broadband
signals. Not physically meaningful.

**Solution:** Bandpass filter the signal before extraction.

### Issue: Dominant Frequency Wrong

**Diagnosis:** The FFT resolution is $\Delta f = f_s / N$. For
short signals, the frequency bins are coarse and the peak may
be mislocated.

**Solution:** Use longer signals or zero-pad to increase FFT resolution.

---

## Integration with Other SPO Modules

### With FrequencyIdentifier

`extract_phases` provides per-channel dominant frequencies.
`identify_frequencies` (DMD-based) provides the global modal
frequencies. The two complement each other:

```python
# Per-channel: Hilbert-based
result = extract_phases(signal_ch0, fs)
ch0_freq = result.dominant_freq

# Global: DMD-based
from scpn_phase_orchestrator.autotune.freq_id import identify_frequencies
freq_result = identify_frequencies(data_all_channels, fs)
global_freqs = freq_result.frequencies
```

### With EVSMonitor

Phase extraction feeds the EVS entrainment battery:

```python
# Extract phases from multiple trials
phases_trials = np.zeros((n_trials, n_tp))
for trial in range(n_trials):
    result = extract_phases(eeg_trials[trial], fs, bandpass=(8, 12))
    phases_trials[trial] = result.phases[:n_tp]

# Evaluate entrainment
evs_result = evs_monitor.evaluate(phases_trials, pause_idx, 10.0, 5.0)
```

### With UPDEEngine

Phase extraction initialises the simulation:

```python
# Extract initial phases from real EEG
init_phases = np.zeros(N)
init_omegas = np.zeros(N)
for ch in range(N):
    result = extract_phases(eeg[ch], fs, bandpass=(8, 12))
    init_phases[ch] = result.phases[-1]
    init_omegas[ch] = result.dominant_freq * 2 * np.pi

eng = UPDEEngine(N, dt=1.0/fs)
phases = eng.run(init_phases, init_omegas, knm, 0.0, 0.0, alpha, n_steps=500)
```

---

## Mathematical Appendix: Hilbert Transform Properties

### Linearity

$$\mathcal{H}[\alpha x + \beta y] = \alpha \mathcal{H}[x] + \beta \mathcal{H}[y]$$

### Self-Inverse (Up to Sign)

$$\mathcal{H}[\mathcal{H}[x]](t) = -x(t)$$

Applying the Hilbert transform twice negates the signal. This means
$\mathcal{H}$ is an involution: $\mathcal{H}^{-1} = -\mathcal{H}$.

### Trigonometric Identities

$$\mathcal{H}[\cos(\omega t)] = \sin(\omega t)$$
$$\mathcal{H}[\sin(\omega t)] = -\cos(\omega t)$$

These follow from the $-\pi/2$ phase shift property and are the
foundation for the analytic signal construction.

### Energy Conservation (Parseval)

$$\int_{-\infty}^{\infty} |\mathcal{H}[x](t)|^2 dt = \int_{-\infty}^{\infty} |x(t)|^2 dt$$

The Hilbert transform preserves signal energy. The analytic signal
has exactly twice the energy of the original: $\int |z|^2 = 2\int |x|^2$.

### Phase-Amplitude Coupling (PAC)

A common analysis in neuroscience extracts the phase of a low-frequency
oscillation and the amplitude of a high-frequency oscillation,
then measures their coupling:

```python
# Low-frequency phase (theta: 4-8 Hz)
theta_result = extract_phases(signal, fs, bandpass=(4, 8))
theta_phase = theta_result.phases

# High-frequency amplitude (gamma: 30-80 Hz)
gamma_result = extract_phases(signal, fs, bandpass=(30, 80))
gamma_amp = gamma_result.amplitudes

# Modulation index: peaked distribution = strong PAC
n_bins = 18
bins = np.linspace(0, 2 * np.pi, n_bins + 1)
amp_per_bin = [gamma_amp[(theta_phase >= bins[i]) &
               (theta_phase < bins[i+1])].mean()
               for i in range(n_bins)]
```

### Hilbert Transform of White Noise

For white Gaussian noise, the Hilbert transform produces another
white Gaussian noise signal with the same variance but uncorrelated.
The analytic signal of noise has uniformly distributed phase and
Rayleigh-distributed amplitude:

$$P(\theta) = \frac{1}{2\pi}, \quad P(A) = \frac{A}{\sigma^2} e^{-A^2/2\sigma^2}$$

This is the null distribution against which entrainment (non-uniform
phase) is tested by the EVS monitor.
