# Entrainment Verification Score (EVS)

## 1. Mathematical Formalism

### Inter-Trial Phase Coherence (ITPC)

ITPC measures the consistency of phase across repeated trials at
each time point:

$$\text{ITPC}(t) = \frac{1}{N_{\text{trials}}} \left| \sum_{k=1}^{N_{\text{trials}}} e^{i\theta_k(t)} \right|$$

where $\theta_k(t)$ is the phase of trial $k$ at time $t$. ITPC
ranges from 0 (uniform phase distribution) to 1 (identical phase
across all trials).

ITPC is the circular analogue of correlation: it measures how
consistently a stimulus evokes the same phase response across
repetitions.

### Mean ITPC

The overall ITPC is the mean across all time points:

$$\overline{\text{ITPC}} = \frac{1}{T} \sum_{t=1}^{T} \text{ITPC}(t)$$

### ITPC Persistence

Persistence measures whether phase coherence survives after the
stimulus stops:

$$\text{Persistence} = \frac{1}{|P|} \sum_{t \in P} \text{ITPC}(t)$$

where $P$ is the set of time-point indices during and after the
stimulus pause window. High persistence indicates genuine neural
entrainment rather than a stimulus-locked artefact.

### Frequency Specificity

The specificity ratio compares ITPC at the target (stimulus)
frequency versus a control (non-stimulus) frequency:

$$\text{Specificity} = \frac{\overline{\text{ITPC}}_{\text{target}}}{\overline{\text{ITPC}}_{\text{control}}}$$

Control phases are obtained by rescaling the target-frequency phases:

$$\theta_{\text{control}}(t) = \theta_{\text{target}}(t) \cdot \frac{f_{\text{control}}}{f_{\text{target}}}$$

This models the assumption that the raw signal was bandpass-filtered
at each frequency before phase extraction. If entrainment is
frequency-specific (genuine), $\text{Specificity} \gg 1$.
If entrainment is broadband (artefact), $\text{Specificity} \approx 1$.

### EVS Decision Rule

A system is classified as **entrained** if and only if all three
criteria pass:

$$\text{is\_entrained} = \left(\overline{\text{ITPC}} \geq \tau_{\text{itpc}}\right) \wedge \left(\text{Persistence} \geq \tau_{\text{pers}}\right) \wedge \left(\text{Specificity} \geq \tau_{\text{spec}}\right)$$

Default thresholds:
- $\tau_{\text{itpc}} = 0.6$ (moderate phase coherence)
- $\tau_{\text{pers}} = 0.4$ (survives brief pause)
- $\tau_{\text{spec}} = 1.5$ (50% stronger at target than control)

---

## 2. Theoretical Context

### Why Three Criteria?

Single measures of entrainment are insufficient:

1. **High ITPC alone** can be caused by stimulus-locked artefacts
   (volume conduction, electromagnetic pickup). The stimulus signal
   directly contaminates the recording without any neural response.

2. **High persistence alone** can arise from intrinsic oscillations
   that happen to be at the stimulus frequency. The oscillator is
   not entrained — it was already oscillating at that frequency.

3. **High specificity alone** can occur with weak entrainment that
   is frequency-specific but too weak to be physiologically meaningful.

The EVS battery requires all three to pass, dramatically reducing
false positives.

### Phase Entrainment in Neuroscience

Entrainment is the process by which an external periodic stimulus
(auditory, visual, or electrical) causes endogenous neural oscillations
to synchronise to the stimulus frequency. This is distinct from:

- **Evoked responses:** Transient, stimulus-locked, no sustained oscillation
- **Steady-state responses (SSR):** Driven by the stimulus, disappear
  immediately when stimulus stops
- **Entrainment:** Sustained phase-locking that persists briefly after
  stimulus cessation — indicating that endogenous oscillators have
  been "captured" by the stimulus

### Applications

- **Auditory entrainment:** Speech rhythm (Giraud & Poeppel, 2012),
  binaural beats, isochronous tone sequences
- **Visual entrainment:** Steady-state visually evoked potentials (SSVEP)
- **Transcranial alternating current stimulation (tACS):** Entraining
  cortical oscillations for cognitive enhancement
- **Brain-computer interfaces:** SSVEP-based BCIs use frequency-specific
  entrainment for user intent decoding

### Historical Context

- **Pikovsky, A., Rosenblum, M., & Kurths, J.** (2001):
  *Synchronization: A Universal Concept in Nonlinear Sciences.*
  Comprehensive theory of entrainment in oscillatory systems.
- **Giraud, A.-L. & Poeppel, D.** (2012): "Cortical oscillations and
  speech processing." Demonstrated that cortical theta entrains to
  speech rhythm.
- **Lakatos, P. et al.** (2008): "Entrainment of neuronal oscillations
  as a mechanism of attentional selection." Showed that attention
  modulates entrainment strength.
- **Lachaux, J.-P. et al.** (1999): "Measuring phase synchrony in
  brain signals." Introduced phase-locking value (PLV), closely related
  to ITPC.
- **Thut, G. et al.** (2011): "Entrainment of perceptually relevant
  brain oscillations by non-invasive rhythmic stimulation of the
  human brain." tACS entrainment evidence.
- **Nozaradan, S. et al.** (2011): "Tagging the neuronal entrainment
  to beat and meter." Frequency tagging approach to measuring
  entrainment.

### Relation to Fluctara

The EVS module is the scientific core of the Fluctara audio entrainment
engine. Fluctara generates binaural/monaural beats and isochronous
pulses; the EVS monitor verifies whether the target brain oscillation
has been successfully entrained by measuring ITPC, persistence, and
frequency specificity from the resulting phase dynamics.

---

## 3. Pipeline Position

```
 Stimulus generator (Fluctara / tACS / auditory)
                          │
                          ↓
 Phase extraction ──→ θ(t) per trial
                          │
                          ↓ (n_trials × n_timepoints)
 ┌── EVSMonitor.evaluate() ──────────────────────────┐
 │                                                     │
 │  1. compute_itpc(phases_trials) → ITPC(t)          │
 │     → mean ITPC                                     │
 │                                                     │
 │  2. itpc_persistence(phases_trials, pause_idx)     │
 │     → persistence score                             │
 │                                                     │
 │  3. _frequency_specificity(phases, f_target, f_ctrl)│
 │     → specificity ratio (Rust if available)         │
 │                                                     │
 │  4. Decision: all three ≥ thresholds?              │
 │     → EVSResult(itpc, persistence, specificity,     │
 │                  is_entrained)                       │
 │                                                     │
 └─────────────────────────────────────────────────────┘
                          │
                          ↓
 Supervisor / Fluctara controller adapts stimulus
```

### Input Contracts

| Parameter | Type | Shape | Range | Meaning |
|-----------|------|-------|-------|---------|
| `phases_trials` | `NDArray[float64]` | `(N_trials, T)` | $[0, 2\pi)$ | Phase per trial per timepoint |
| `pause_indices` | `list[int] \| NDArray` | `(P,)` | $[0, T)$ | Timepoint indices in pause window |
| `target_freq` | `float` | scalar | $> 0$ | Stimulus frequency (Hz) |
| `control_freq` | `float` | scalar | $> 0$ | Control frequency (Hz) |

### Output Contract

```python
@dataclass(frozen=True, slots=True)
class EVSResult:
    itpc_value: float       # Mean ITPC, ∈ [0, 1]
    persistence_score: float # ITPC during/after pause, ∈ [0, 1]
    specificity_ratio: float # target ITPC / control ITPC, ≥ 0
    is_entrained: bool       # All three criteria met
```

---

## 4. Features

- **Three-criterion entrainment verification** — ITPC + persistence +
  frequency specificity for robust false-positive rejection
- **ITPC computation** — circular mean resultant length across trials
- **Persistence measurement** — ITPC during stimulus pause window
- **Frequency specificity** — ratio of target/control ITPC via
  phase rescaling
- **Configurable thresholds** — all three thresholds adjustable
- **Frozen dataclass result** — immutable, hashable EVSResult
- **Rust FFI for specificity** — native mean_itpc computation,
  1.1-2.1x speedup
- **Inf handling** — returns infinity when control ITPC is zero but
  target is non-zero

---

## 5. Usage Examples

### Basic: Full EVS Battery

```python
import numpy as np
from scpn_phase_orchestrator.monitor.evs import EVSMonitor

n_trials, n_tp = 20, 100
rng = np.random.default_rng(42)

# Simulated entrained phases: consistent across trials at target freq
target_phase = np.linspace(0, 4 * np.pi, n_tp)
noise = rng.normal(0, 0.3, (n_trials, n_tp))
phases = target_phase[np.newaxis, :] + noise

pause_idx = list(range(80, 100))  # Last 20% is pause

monitor = EVSMonitor(
    itpc_threshold=0.5,
    persistence_threshold=0.3,
    specificity_threshold=1.2,
)
result = monitor.evaluate(phases, pause_idx, target_freq=10.0, control_freq=7.0)

print(f"ITPC: {result.itpc_value:.4f}")
print(f"Persistence: {result.persistence_score:.4f}")
print(f"Specificity: {result.specificity_ratio:.4f}")
print(f"Entrained: {result.is_entrained}")
```

### Not Entrained: Random Phases

```python
import numpy as np
from scpn_phase_orchestrator.monitor.evs import EVSMonitor

n_trials, n_tp = 20, 100
phases = np.random.default_rng(42).uniform(0, 2 * np.pi, (n_trials, n_tp))

monitor = EVSMonitor()
result = monitor.evaluate(phases, list(range(80, 100)), 10.0, 5.0)
assert not result.is_entrained  # Random phases → no entrainment
print(f"ITPC: {result.itpc_value:.4f} (should be low)")
```

### Comparing Entrainment Strengths

```python
import numpy as np
from scpn_phase_orchestrator.monitor.evs import EVSMonitor

monitor = EVSMonitor()
rng = np.random.default_rng(42)
target = np.linspace(0, 4 * np.pi, 100)
pause = list(range(80, 100))

for noise_std in [0.1, 0.5, 1.0, 2.0]:
    noise = rng.normal(0, noise_std, (20, 100))
    phases = target[np.newaxis, :] + noise
    result = monitor.evaluate(phases, pause, 10.0, 5.0)
    print(f"Noise σ={noise_std:.1f}: ITPC={result.itpc_value:.3f}, "
          f"Entrained={result.is_entrained}")
```

### Integration with UPDEEngine

```python
import numpy as np
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.monitor.evs import EVSMonitor

N = 16
eng = UPDEEngine(N, dt=0.01)
monitor = EVSMonitor()
rng = np.random.default_rng(42)
omegas = np.ones(N) * 2 * np.pi * 10.0  # 10 Hz target
knm = np.full((N, N), 0.5); np.fill_diagonal(knm, 0.0)
alpha = np.zeros((N, N))

# Collect trials
trials = []
for trial in range(20):
    phases = rng.uniform(0, 2 * np.pi, N)
    trial_phases = []
    for step in range(100):
        phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        trial_phases.append(phases[0])  # Track oscillator 0
    trials.append(trial_phases)

phases_trials = np.array(trials)
result = monitor.evaluate(phases_trials, list(range(80, 100)), 10.0, 5.0)
print(f"Engine-based ITPC: {result.itpc_value:.4f}")
```

---

## 6. Technical Reference

### Class: EVSMonitor

::: scpn_phase_orchestrator.monitor.evs

### Constructor Parameters

| Parameter | Type | Default | Meaning |
|-----------|------|---------|---------|
| `itpc_threshold` | `float` | 0.6 | Minimum mean ITPC |
| `persistence_threshold` | `float` | 0.4 | Minimum pause-window ITPC |
| `specificity_threshold` | `float` | 1.5 | Minimum target/control ratio |

### Methods

| Method | Input | Output |
|--------|-------|--------|
| `evaluate(phases_trials, pause_indices, target_freq, control_freq)` | Trial phases + params | `EVSResult` |
| `_frequency_specificity(phases_trials, target_freq, control_freq)` | Trial phases + freqs | `float` ratio |

### Rust Engine Function

```rust
pub fn frequency_specificity(
    phases_flat: &[f64],   // row-major (n_trials × n_tp)
    n_trials: usize,
    n_timepoints: usize,
    target_freq: f64,
    control_freq: f64,
) -> f64                   // target_itpc / control_itpc
```

Internal helper: `mean_itpc(phases_flat, n_trials, n_tp) -> f64`
computes the mean ITPC across timepoints using vectorised sin/cos
summation.

### Auto-Select Logic

```python
try:
    from spo_kernel import frequency_specificity_rust as _rust_freq_spec
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False
```

Only `_frequency_specificity` uses the Rust path. `compute_itpc` and
`itpc_persistence` are computed by the Python `itpc` module (which
may have its own Rust path).

---

## 7. Performance Benchmarks

Measured on Intel Core i5-11600K @ 3.90 GHz, 32 GB DDR4-2400.
Median of 100-200 iterations, random phase data.

### frequency_specificity

| Trials × Timepoints | Python (µs) | Rust (µs) | Speedup |
|---------------------|-------------|-----------|---------|
| 10 × 50 | 42.6 | 20.1 | **2.1x** |
| 20 × 100 | 121.0 | 85.4 | **1.4x** |
| 50 × 200 | 569.6 | 534.9 | **1.1x** |

### Why Decreasing Speedup?

The Python path uses NumPy vectorised operations (`np.exp(1j * phases)`,
`np.abs`, `np.mean`), which are BLAS-accelerated for large arrays.
The Rust path uses scalar sin/cos loops. At small sizes, Rust wins
via reduced Python overhead. At large sizes, NumPy's vectorisation
catches up.

### Full evaluate() Latency

The full EVS battery includes ITPC computation (Python), persistence
(Python), and specificity (Rust). For 20 × 100 phases:

| Component | Time (µs) | Fraction |
|-----------|-----------|----------|
| compute_itpc | ~80 | ~35% |
| itpc_persistence | ~40 | ~18% |
| frequency_specificity | ~85 | ~37% |
| Decision logic | ~1 | ~0.5% |
| **Total** | **~230** | **100%** |

### Memory Usage

- Phase arrays: $N_{\text{trials}} \times T$ floats
- Control phases: $N_{\text{trials}} \times T$ floats (temporary)
- ITPC values: $T$ floats
- Total for 20 × 100: ~35 KB

### Test Coverage

- **Rust tests:** 6 (evs module in spo-engine)
  - Perfect sync high specificity, random phases low specificity,
    zero frequencies, empty input, mean_itpc synchronised,
    mean_itpc uniform
- **Python tests:** 9 (`tests/test_evs.py`)
  - EVSResult creation, entrained detection, not entrained random,
    persistence measurement, specificity ratio, threshold sensitivity,
    edge cases, pipeline wiring
- **Source lines:** 130 (Rust) + 141 (Python) = 271 total

---

## 8. Citations

1. **Pikovsky, A., Rosenblum, M., & Kurths, J.** (2001).
   *Synchronization: A Universal Concept in Nonlinear Sciences.*
   Cambridge University Press. ISBN: 978-0-521-59285-7.

2. **Giraud, A.-L. & Poeppel, D.** (2012).
   "Cortical oscillations and speech processing: Emerging computational
   principles and operations."
   *Nature Neuroscience* 15(4):511-517.
   DOI: [10.1038/nn.3063](https://doi.org/10.1038/nn.3063)

3. **Lakatos, P., Karmos, G., Mehta, A. D., Ulbert, I., & Schroeder, C. E.**
   (2008).
   "Entrainment of neuronal oscillations as a mechanism of attentional
   selection."
   *Science* 320(5872):110-113.
   DOI: [10.1126/science.1154735](https://doi.org/10.1126/science.1154735)

4. **Lachaux, J.-P., Rodriguez, E., Martinerie, J., & Varela, F. J.**
   (1999).
   "Measuring phase synchrony in brain signals."
   *Human Brain Mapping* 8(4):194-208.
   DOI: [10.1002/(SICI)1097-0193(1999)8:4<194::AID-HBM4>3.0.CO;2-C](https://doi.org/10.1002/(SICI)1097-0193(1999)8:4<194::AID-HBM4>3.0.CO;2-C)

5. **Thut, G., Schyns, P. G., & Gross, J.** (2011).
   "Entrainment of perceptually relevant brain oscillations by
   non-invasive rhythmic stimulation of the human brain."
   *Frontiers in Psychology* 2:170.
   DOI: [10.3389/fpsyg.2011.00170](https://doi.org/10.3389/fpsyg.2011.00170)

6. **Nozaradan, S., Peretz, I., Missal, M., & Mouraux, A.** (2011).
   "Tagging the neuronal entrainment to beat and meter."
   *Journal of Neuroscience* 31(28):10234-10240.
   DOI: [10.1523/JNEUROSCI.0411-11.2011](https://doi.org/10.1523/JNEUROSCI.0411-11.2011)

7. **Tallon-Baudry, C., Bertrand, O., Delpuech, C., & Pernier, J.**
   (1996).
   "Stimulus specificity of phase-locked and non-phase-locked 40 Hz
   visual responses in human."
   *Journal of Neuroscience* 16(13):4240-4249.

8. **Obleser, J. & Kayser, C.** (2019).
   "Neural entrainment and attentional selection in the listening brain."
   *Trends in Cognitive Sciences* 23(11):913-926.
   DOI: [10.1016/j.tics.2019.08.004](https://doi.org/10.1016/j.tics.2019.08.004)

---

## Edge Cases and Limitations

### Single Trial

With $N_{\text{trials}} = 1$, ITPC = 1 at every time point
(trivially — the single trial always agrees with itself). This
produces a false positive. Minimum recommended: $N_{\text{trials}} \geq 10$.

### Very Short Pause Window

If `pause_indices` contains only 1-2 indices, the persistence
estimate has high variance. Recommend at least 10 pause indices.

### Target ≈ Control Frequency

When $f_{\text{target}} \approx f_{\text{control}}$, the phase
rescaling produces nearly identical phases → specificity ≈ 1.
Use control frequencies at least 30% different from target.

### Negative or Zero Frequencies

Both `target_freq` and `control_freq` must be positive. If either
is ≤ 0, `_frequency_specificity` returns 0.0.

---

## Integration with Other SPO Modules

### With Fluctara Audio Engine

The EVS monitor is the verification backend for Fluctara:

```python
# Fluctara generates stimulus → EEG recorded → phases extracted
result = evs_monitor.evaluate(phases_trials, pause_idx, stim_freq, ctrl_freq)
if result.is_entrained:
    fluctara.maintain_stimulus()
else:
    fluctara.increase_amplitude()
```

### With Sleep Staging

Entrainment efficacy varies by sleep stage:
- Wake/N1: Best entrainment response
- N2/N3: Reduced (endogenous slow oscillations compete)
- REM: Variable (depends on cortical activation)

### With SSGF Geometry Control

If entrainment verification fails despite strong coupling, the SSGF
engine may need to adjust the coupling topology to facilitate
frequency-specific synchronisation.

---

## Troubleshooting

### Issue: is_entrained Always False

**Diagnosis:** Check which criterion fails:
- Low ITPC → phases are inconsistent across trials (too much noise)
- Low persistence → entrainment is stimulus-locked, not endogenous
- Low specificity → entrainment is broadband, not frequency-specific

### Issue: Specificity is Infinity

**Diagnosis:** Control ITPC is near zero (< 1e-12) while target
ITPC is positive. This is valid — it means strong frequency-specific
entrainment with no broadband component.

### Issue: High ITPC but Specificity ≈ 1

**Diagnosis:** The phase coherence is broadband (affects all
frequencies equally). This suggests a stimulus artefact, not
genuine neural entrainment.

**Solution:** Verify the phase extraction pipeline. Ensure bandpass
filtering is applied before phase computation. Check for DC offset
or saturation in the recording.

### Issue: Persistence Higher Than Mean ITPC

**Diagnosis:** This can occur when the stimulus is disruptive
(reduces ITPC during stimulation) but the endogenous oscillator
resumes coherent oscillation during the pause. This is a valid
but unusual pattern — it indicates "rebound entrainment."

**Solution:** This is not an error. The system is genuinely entrained
but the measurement is capturing the post-stimulus recovery.

---

## Mathematical Appendix: ITPC Statistical Significance

For random (unentraining) phases uniformly distributed on $[0, 2\pi)$,
the expected ITPC follows a Rayleigh distribution:

$$P(\text{ITPC} > z) \approx e^{-N_{\text{trials}} \cdot z^2}$$

The critical ITPC for $p < 0.05$ significance with $N$ trials:

$$z_{\text{crit}} = \sqrt{\frac{-\ln(0.05)}{N_{\text{trials}}}} = \sqrt{\frac{2.996}{N_{\text{trials}}}}$$

| $N_{\text{trials}}$ | $z_{\text{crit}}$ ($p < 0.05$) |
|---------------------|-------------------------------|
| 10 | 0.547 |
| 20 | 0.387 |
| 50 | 0.245 |
| 100 | 0.173 |

The default threshold $\tau_{\text{itpc}} = 0.6$ exceeds the
$p < 0.05$ critical value for all $N \geq 10$, ensuring statistical
rigour. For stricter testing, use $p < 0.01$ thresholds or
Bonferroni correction for multiple frequency bins.
