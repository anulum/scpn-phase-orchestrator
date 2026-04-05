# Frequency Identification via Dynamic Mode Decomposition

## 1. Mathematical Formalism

### Dynamic Mode Decomposition (DMD)

DMD extracts spatiotemporal coherent structures from data by
approximating the dynamics with a linear operator:

$$X' \approx A X$$

where $X \in \mathbb{R}^{n \times (T-1)}$ contains the first $T-1$
time snapshots and $X' \in \mathbb{R}^{n \times (T-1)}$ contains
the shifted snapshots (columns 2 through $T$).

### SVD-Based Exact DMD

The operator $A$ is computed via the SVD of $X$:

1. **SVD:** $X = U \Sigma V^*$
2. **Rank truncation:** Keep $r$ modes where $\sigma_i > \tau \cdot \sigma_1$
3. **Reduced operator:** $\tilde{A} = U_r^* X' V_r \Sigma_r^{-1}$
4. **Eigendecomposition:** $\tilde{A} W = W \Lambda$

The eigenvalues $\lambda_k$ of $\tilde{A}$ encode the dynamics:

### Frequency Extraction

$$f_k = \frac{|\text{Im}(\ln \lambda_k)|}{2\pi \Delta t}$$

where $\Delta t = 1/f_s$ is the sampling interval. The magnitude
$|\lambda_k|$ indicates the growth/decay rate of each mode:
- $|\lambda_k| = 1$: neutrally stable (sustained oscillation)
- $|\lambda_k| < 1$: decaying mode
- $|\lambda_k| > 1$: growing mode (unstable)

### Rank Truncation

The rank threshold $\tau$ (default 0.01) determines how many
singular values are retained:

$$r = \max\{k : \sigma_k > \tau \cdot \sigma_1\}$$

Smaller $\tau$ retains more modes (potentially noisy), larger $\tau$
keeps only the dominant structures. For oscillatory systems with
clear frequency peaks, $\tau = 0.01$ works well.

### Channel-to-Mode Assignment

After extracting global DMD frequencies, each channel is assigned
to the nearest DMD mode based on its per-channel dominant frequency
(from Hilbert phase extraction):

$$\text{assignment}(i) = \arg\min_k |f_k^{\text{DMD}} - f_i^{\text{channel}}|$$

This maps the multichannel data to the SCPN layer structure: each
oscillator operates at the frequency of its nearest DMD mode.

---

## 2. Theoretical Context

### Why DMD for Frequency Identification?

Standard Fourier analysis identifies frequencies per channel
independently. DMD identifies the **global spatiotemporal modes**
that explain the multichannel dynamics simultaneously. This has
two advantages:

1. **Shared frequencies:** DMD finds frequencies shared across
   channels, even if individual channel spectra are noisy
2. **Mode amplitudes:** DMD provides the spatial structure (which
   channels participate in each mode)

### Relation to Koopman Theory

DMD is a data-driven approximation to the Koopman operator —
the infinite-dimensional linear operator that governs the evolution
of observables of a nonlinear dynamical system. For a system
$x_{t+1} = F(x_t)$, the Koopman operator $\mathcal{K}$ satisfies:

$$g(x_{t+1}) = \mathcal{K} g(x_t)$$

for any observable $g$. DMD approximates the Koopman eigenvalues
(frequencies) and eigenfunctions (spatial modes) from data.

### DMD vs FFT vs Welch

| Method | Input | Output | Multivariate | Noise |
|--------|-------|--------|-------------|-------|
| **FFT** | Single channel | Frequency spectrum | No | Moderate |
| **Welch PSD** | Single channel | Power spectral density | No | Good |
| **DMD** | Multichannel | Frequencies + spatial modes | Yes | Good |
| **Multitaper** | Single channel | PSD | No | Excellent |

DMD is the natural choice for multichannel oscillatory data because
it simultaneously identifies frequencies and their spatial patterns.

### Rust Path (Interface Mismatch)

The Rust implementation (`freq_id.rs`) uses autocorrelation-based
frequency detection rather than DMD. The interfaces are incompatible:
- Rust: `autocorrelation_frequencies(signal, fs) -> Vec<f64>`
  (returns frequency list)
- Python: `identify_frequencies(data, fs) -> FrequencyResult`
  (returns frequencies + amplitudes + layer assignment)

The Rust module provides a lightweight alternative for single-channel
frequency detection in embedded contexts. For the full multichannel
DMD pipeline, the Python path is required.

### Historical Context

- **Schmid, P. J.** (2010): "Dynamic mode decomposition of numerical
  and experimental data." The foundational DMD paper.
- **Tu, J. H. et al.** (2014): "On dynamic mode decomposition:
  Theory and applications." Exact DMD algorithm used here.
- **Kutz, J. N. et al.** (2016): *Dynamic Mode Decomposition:
  Data-Driven Modeling of Complex Systems.* SIAM textbook.
- **Mezić, I.** (2005): "Spectral analysis of nonlinear flows."
  Koopman operator theory underlying DMD.
- **Brunton, S. L. & Kutz, J. N.** (2019): *Data-Driven Science
  and Engineering.* Modern reference for DMD and SINDy.
- **Cooley, J. W. & Tukey, J. W.** (1965): "An algorithm for the
  machine calculation of complex Fourier series." The FFT algorithm
  used in the per-channel phase extraction.

### DMD Limitations

1. **Assumes linear dynamics:** For strongly nonlinear systems,
   DMD modes may not correspond to physical oscillations
2. **Snapshot spacing matters:** $\Delta t$ must satisfy Nyquist
   ($f_s > 2 f_{\max}$)
3. **Requires sufficient snapshots:** $T \geq 2r + 1$ for $r$ modes
4. **Sensitivity to transients:** Startup transients can corrupt
   mode estimates — discard initial data

---

## 3. Pipeline Position

```
 Multichannel data (EEG, sensor array)
              │  shape: (n_channels, n_samples)
              ↓
 ┌── identify_frequencies(data, fs, n_modes) ─────┐
 │                                                  │
 │  Step 1: SVD of data matrix X                   │
 │  Step 2: Rank truncation (threshold τ)          │
 │  Step 3: Reduced DMD operator Ã                 │
 │  Step 4: Eigenvalues → frequencies + amplitudes │
 │  Step 5: Per-channel assignment via Hilbert      │
 │                                                  │
 │  Output: FrequencyResult(frequencies, amplitudes,│
 │          layer_assignment)                        │
 │                                                  │
 └──────────────────┬───────────────────────────────┘
                    │
                    ↓
 omegas = frequencies * 2π → UPDEEngine
 layer_assignment → CouplingBuilder (which layers share frequency)
```

### Input Contracts

| Parameter | Type | Shape | Range | Meaning |
|-----------|------|-------|-------|---------|
| `data` | `NDArray[float64]` | `(n_ch, T)` | any | Multichannel time series |
| `fs` | `float` | scalar | $> 0$ | Sampling frequency (Hz) |
| `n_modes` | `int \| None` | scalar | $\geq 1$ | DMD modes (auto if None) |
| `rank_threshold` | `float` | scalar | $(0, 1)$ | SVD truncation threshold |

### Output Contract

```python
@dataclass
class FrequencyResult:
    frequencies: NDArray     # (r,), sorted by amplitude (descending)
    amplitudes: NDArray      # (r,), |λ_k|
    layer_assignment: list[int]  # (n_ch,), channel → mode index
```

---

## 4. Features

- **SVD-based exact DMD** — numerically stable, handles rank-deficient
  data
- **Automatic rank selection** — from singular value threshold
- **Frequency + amplitude extraction** — from DMD eigenvalues
- **Channel-to-mode assignment** — maps multichannel data to SCPN
  layers via Hilbert-based per-channel frequency
- **Amplitude-sorted output** — dominant modes first
- **Configurable rank threshold** — trades off noise rejection vs.
  mode discovery
- **Rust engine for single-channel** — autocorrelation-based frequency
  detection (different algorithm, no Python FFI binding)
- **Minimum data check** — requires $T \geq 3$

---

## 5. Usage Examples

### Basic: Identify Frequencies

```python
import numpy as np
from scpn_phase_orchestrator.autotune.freq_id import identify_frequencies

fs = 250.0  # 250 Hz sampling
t = np.arange(2000) / fs
# 3-channel data: 10 Hz, 10 Hz, 25 Hz
data = np.array([
    np.sin(2 * np.pi * 10 * t),
    np.sin(2 * np.pi * 10 * t + 0.5),  # Same freq, different phase
    np.sin(2 * np.pi * 25 * t),
])

result = identify_frequencies(data, fs)
print(f"Frequencies: {result.frequencies[:5].round(1)}")
print(f"Amplitudes: {result.amplitudes[:5].round(3)}")
print(f"Layer assignment: {result.layer_assignment}")
# Channel 0 and 1 → same mode (10 Hz), Channel 2 → 25 Hz mode
```

### With Noisy Data

```python
import numpy as np
from scpn_phase_orchestrator.autotune.freq_id import identify_frequencies

rng = np.random.default_rng(42)
fs = 250.0
t = np.arange(2000) / fs
signal = np.sin(2 * np.pi * 10 * t) + 0.5 * rng.normal(0, 1, len(t))
data = np.stack([signal, signal + rng.normal(0, 0.3, len(t))])

result = identify_frequencies(data, fs, rank_threshold=0.05)
print(f"Dominant freq: {result.frequencies[0]:.1f} Hz")
```

### Map to SCPN Oscillators

```python
import numpy as np
from scpn_phase_orchestrator.autotune.freq_id import identify_frequencies
from scpn_phase_orchestrator.upde.engine import UPDEEngine

data = np.random.default_rng(42).normal(0, 1, (16, 2000))
fs = 250.0

result = identify_frequencies(data, fs)

# Map DMD frequencies to oscillator natural frequencies
omegas = np.zeros(16)
for ch in range(16):
    mode_idx = result.layer_assignment[ch]
    omegas[ch] = result.frequencies[mode_idx] * 2 * np.pi

eng = UPDEEngine(16, dt=1.0/fs)
# ... simulate with identified frequencies
```

### Compare DMD vs Per-Channel FFT

```python
import numpy as np
from scpn_phase_orchestrator.autotune.freq_id import identify_frequencies
from scpn_phase_orchestrator.autotune.phase_extract import extract_phases

fs = 250.0
t = np.arange(2000) / fs
data = np.array([
    np.sin(2 * np.pi * 10 * t),
    np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 40 * t),
])

# DMD: global modes
dmd_result = identify_frequencies(data, fs)

# Per-channel FFT
for ch in range(2):
    hilbert_result = extract_phases(data[ch], fs)
    print(f"Ch {ch} FFT dominant: {hilbert_result.dominant_freq:.1f} Hz")

print(f"DMD modes: {dmd_result.frequencies[:3].round(1)}")
```

---

## 6. Technical Reference

### Function: identify_frequencies

::: scpn_phase_orchestrator.autotune.freq_id

### Dataclass: FrequencyResult

```python
@dataclass
class FrequencyResult:
    frequencies: NDArray     # DMD frequencies (Hz), sorted by amplitude
    amplitudes: NDArray      # |λ_k|, sorted descending
    layer_assignment: list[int]  # channel → nearest mode index
```

### DMD Algorithm Steps

```python
# 1. Form data matrices
X = data[:, :-1]      # (n_ch, T-1)
Xp = data[:, 1:]      # (n_ch, T-1)

# 2. SVD and truncation
U, S, Vt = np.linalg.svd(X, full_matrices=False)
r = np.sum(S > rank_threshold * S[0])
U_r, S_r, Vt_r = U[:,:r], S[:r], Vt[:r,:]

# 3. Reduced DMD operator
A_tilde = U_r.conj().T @ Xp @ Vt_r.conj().T @ np.diag(1/S_r)

# 4. Eigenvalues → frequencies
evals, _ = np.linalg.eig(A_tilde)
freqs = np.abs(np.log(evals).imag / (2 * np.pi * dt))
amps = np.abs(evals)
```

### Rust Engine Function (Different Algorithm)

```rust
// Autocorrelation-based frequency detection (single-channel)
pub fn autocorrelation_frequencies(
    signal: &[f64], fs: f64,
) -> Vec<f64>
```

This is a lightweight alternative that finds peaks in the
autocorrelation function. It does not support multichannel data
or spatial mode extraction. No Python FFI binding is provided
due to the interface mismatch.

---

## 7. Performance Benchmarks

Measured on Intel Core i5-11600K @ 3.90 GHz, 32 GB DDR4-2400.
Python (numpy SVD + eigendecomposition) only.

| n_channels × T | Python (ms) |
|-----------------|-------------|
| 4 × 100 | 0.423 |
| 8 × 500 | 0.703 |
| 16 × 1000 | 2.1 |
| 32 × 2000 | 8.5 |

### Scaling Analysis

The cost is dominated by:
1. SVD of $X$ (n_ch × T): $O(\min(n_{\text{ch}}^2 T, n_{\text{ch}} T^2))$
2. Eigendecomposition of $\tilde{A}$ (r × r): $O(r^3)$
3. Per-channel Hilbert extraction: $O(n_{\text{ch}} \cdot T \log T)$

For typical use ($n_{\text{ch}} = 16$, $T = 1000$), the total is
~2 ms — well within real-time budgets.

### Memory Usage

- Data matrices $X, X'$: $2 \times n_{\text{ch}} \times T$ floats
- SVD workspace: $O(n_{\text{ch}} \times T)$
- Reduced operator: $r \times r$ complex matrix
- Total for 16 × 1000: ~256 KB

### Test Coverage

- **Rust tests:** 6 (freq_id module in spo-engine)
  - Autocorrelation basic, empty input, constant signal,
    multi-frequency detection, Nyquist boundary, determinism
- **Python tests:** 9 (`tests/test_freq_id.py`)
  - Single frequency detection, multi-frequency, noisy data,
    rank threshold effect, layer assignment, edge cases,
    pipeline wiring with phase_extract
- **Source lines:** 177 (Rust) + 93 (Python) = 270 total

---

## 8. Citations

1. **Schmid, P. J.** (2010).
   "Dynamic mode decomposition of numerical and experimental data."
   *Journal of Fluid Mechanics* 656:5-28.
   DOI: [10.1017/S0022112010001217](https://doi.org/10.1017/S0022112010001217)

2. **Tu, J. H., Rowley, C. W., Luchtenburg, D. M., Brunton, S. L.,
   & Kutz, J. N.** (2014).
   "On dynamic mode decomposition: Theory and applications."
   *Journal of Computational Dynamics* 1(2):391-421.
   DOI: [10.3934/jcd.2014.1.391](https://doi.org/10.3934/jcd.2014.1.391)

3. **Kutz, J. N., Brunton, S. L., Brunton, B. W., & Proctor, J. L.**
   (2016).
   *Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems.*
   SIAM. ISBN: 978-1-611-97449-2.

4. **Mezić, I.** (2005).
   "Spectral analysis of nonlinear flows."
   *Journal of Fluid Mechanics* 540:199-227.
   DOI: [10.1017/S0022112005005507](https://doi.org/10.1017/S0022112005005507)

5. **Brunton, S. L. & Kutz, J. N.** (2019).
   *Data-Driven Science and Engineering: Machine Learning, Dynamical
   Systems, and Control.* Cambridge University Press.
   ISBN: 978-1-108-42209-3.

6. **Rowley, C. W., Mezić, I., Bagheri, S., Schlatter, P., &
   Henningson, D. S.** (2009).
   "Spectral analysis of nonlinear flows."
   *Journal of Fluid Mechanics* 641:115-127.
   DOI: [10.1017/S0022112009992059](https://doi.org/10.1017/S0022112009992059)

7. **Cooley, J. W. & Tukey, J. W.** (1965).
   "An algorithm for the machine calculation of complex Fourier series."
   *Mathematics of Computation* 19(90):297-301.
   DOI: [10.2307/2003354](https://doi.org/10.2307/2003354)

8. **Pikovsky, A., Rosenblum, M., & Kurths, J.** (2001).
   *Synchronization: A Universal Concept in Nonlinear Sciences.*
   Cambridge University Press. ISBN: 978-0-521-59285-7.

---

## Edge Cases and Limitations

### n_samples < 3

Raises `ValueError`. DMD requires at least 3 snapshots to form
the shifted data matrices.

### Single Channel

DMD with a single channel ($n_{\text{ch}} = 1$) reduces to
scalar dynamics — the SVD is trivial and the DMD eigenvalue is
the ratio of consecutive samples. For single-channel frequency
identification, use `extract_phases` (Hilbert) instead.

### Rank Threshold Too High

If $\tau$ is set too high (e.g., 0.5), many modes are truncated
and the frequency identification becomes coarse. For exploratory
analysis, use $\tau = 0.001$. For noise rejection, use $\tau = 0.05$.

### Complex Eigenvalues with |λ| >> 1

If the data contains growing modes ($|\lambda| > 1$), the system
is unstable. The frequencies are still valid but the amplitudes
are physically meaningless. Filter or detrend the data before DMD.

### Aliased Frequencies

If the true frequency exceeds $f_s / 2$ (Nyquist), it aliases to
a lower frequency. DMD cannot detect this — it reports the aliased
frequency. Ensure $f_s > 2 f_{\max}$.

---

## Troubleshooting

### Issue: All Frequencies Near Zero

**Diagnosis:** The data is dominated by low-frequency trends or DC.

**Solution:** Detrend the data: `data -= data.mean(axis=1, keepdims=True)`.
For slow drifts, use a high-pass filter.

### Issue: Too Many Modes

**Diagnosis:** The rank threshold is too low, retaining noisy modes.

**Solution:** Increase `rank_threshold` (e.g., from 0.01 to 0.05).
Or manually set `n_modes` to the expected number of oscillatory modes.

### Issue: Layer Assignment All Same Index

**Diagnosis:** All channels have similar dominant frequencies, mapping
to the same DMD mode. This is correct for globally synchronised data.

**Solution:** If finer discrimination is needed, use bandpass filtering
before DMD to separate overlapping frequency bands.

---

## Integration with Other SPO Modules

### With extract_phases

`identify_frequencies` uses `extract_phases` internally for
per-channel dominant frequency estimation:

```python
# Internal call in identify_frequencies:
for ch in range(n_ch):
    pr = extract_phases(data[ch], fs)
    channel_freqs.append(pr.dominant_freq)
```

### With CouplingBuilder

The layer assignment maps channels to SCPN layers. Channels assigned
to the same DMD mode can share coupling parameters:

```python
result = identify_frequencies(eeg_data, fs)
# Channels with same layer_assignment share a frequency band
# → group them in the coupling matrix
for mode_idx in set(result.layer_assignment):
    channels = [ch for ch, a in enumerate(result.layer_assignment) if a == mode_idx]
    # Set strong intra-group coupling, weak inter-group
```

### With OttAntonsenReduction

Each DMD mode can be independently analysed with the OA reduction:

```python
for k, freq in enumerate(result.frequencies):
    channels = [ch for ch, a in enumerate(result.layer_assignment) if a == k]
    omega_0 = freq * 2 * np.pi
    delta = np.std([omegas[ch] for ch in channels])
    oa = OttAntonsenReduction(omega_0, delta, K=coupling_strength)
    R_predicted = oa.steady_state_R()
    print(f"Mode {k} ({freq:.1f} Hz): R_predicted = {R_predicted:.4f}")
```

### With SplittingEngine

The identified frequencies initialise the natural frequencies:

```python
omegas = np.zeros(N)
for ch in range(N):
    mode_idx = result.layer_assignment[ch]
    omegas[ch] = result.frequencies[mode_idx] * 2 * np.pi

eng = SplittingEngine(N, dt=1.0/fs)
phases = eng.run(init_phases, omegas, knm, 0.0, 0.0, alpha, n_steps=1000)
```

### With SSGF Geometry Control

DMD mode structure can guide the SSGF initial geometry. Channels
in the same DMD mode should have stronger coupling:

```python
W_init = np.zeros((N, N))
for mode_idx in set(result.layer_assignment):
    channels = [ch for ch, a in enumerate(result.layer_assignment) if a == mode_idx]
    for i in channels:
        for j in channels:
            if i != j:
                W_init[i, j] = 0.5  # Strong intra-mode coupling
```

### With EVSMonitor

The identified dominant frequency informs the EVS target frequency:

```python
result = identify_frequencies(eeg_data, fs)
target_freq = result.frequencies[0]  # Most dominant mode
control_freq = result.frequencies[1] if len(result.frequencies) > 1 else target_freq * 1.5
evs_result = evs_monitor.evaluate(phases_trials, pause_idx, target_freq, control_freq)
```
