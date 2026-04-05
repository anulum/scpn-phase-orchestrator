# Sleep Stage Classifier

## 1. Mathematical Formalism

### AASM Sleep Staging via Kuramoto Order Parameter

The American Academy of Sleep Medicine (AASM) defines five sleep
stages: Wake (W), N1, N2, N3, and REM. In the SCPN framework,
these stages are mapped to the Kuramoto order parameter $R$, which
measures cortical synchronisation.

### Stage Classification Function

$$\text{stage}(R, d) = \begin{cases} \text{N3} & R \geq 0.70 \\ \text{N2} & 0.40 \leq R < 0.70 \\ \text{REM} & 0.30 \leq R < 0.40 \text{ and } d = \text{true} \\ \text{N1} & 0.30 \leq R < 0.40 \text{ and } d = \text{false} \\ \text{REM} & 0.20 \leq R < 0.30 \text{ and } d = \text{true} \\ \text{Wake} & \text{otherwise} \end{cases}$$

where $d$ is the `functional_desync` flag, indicating whether the
desynchronisation pattern is characteristic of REM (low-voltage
mixed-frequency EEG) versus wakefulness.

### Physiological Basis of Thresholds

| Stage | $R$ Range | EEG Pattern | Kuramoto Interpretation |
|-------|----------|-------------|------------------------|
| N3 | $\geq 0.70$ | Delta waves (0.5-4 Hz), >75 µV | Strongly phase-locked cortical columns |
| N2 | $[0.40, 0.70)$ | Sleep spindles (12-14 Hz), K-complexes | Moderate synchrony with bursting |
| N1 | $[0.30, 0.40)$ | Theta waves (4-8 Hz), vertex sharp waves | Partial desynchronisation |
| REM | $[0.20, 0.40)$ + desync | Low-voltage mixed frequency | Desynchronised but not wakeful |
| Wake | $< 0.30$ | Alpha (8-12 Hz), beta (12-30 Hz) | Desynchronised cortex |

### Ultradian Sleep Cycle

Human sleep follows an approximately 90-minute ultradian rhythm
(Rechtschaffen & Kales, 1968):

$$\phi_{\text{ultradian}}(t) = \frac{(t - t_{\text{N3}}) \bmod T_{\text{ultradian}}}{T_{\text{ultradian}}}$$

where:
- $t_{\text{N3}}$ is the time of the most recent N3 epoch (cycle trough)
- $T_{\text{ultradian}} = 5400$ seconds (90 minutes)
- $\phi \in [0, 1)$ represents the position within the cycle

The cycle structure:
- $\phi \approx 0$: N3 onset (deepest sleep, highest $R$)
- $\phi \approx 0.3$: N2/N1 (ascending, decreasing $R$)
- $\phi \approx 0.5$: REM (lowest $R$ of sleep)
- $\phi \approx 0.7$: N1/N2 (descending, increasing $R$)
- $\phi \to 1$: approaching next N3

### Stage Code Mapping

For FFI transfer, stages are encoded as unsigned integers:

| Code | Stage | $R$ Range |
|------|-------|----------|
| 0 | Wake | $< 0.30$ (no desync) |
| 1 | N1 | $[0.30, 0.40)$ (no desync) |
| 2 | N2 | $[0.40, 0.70)$ |
| 3 | N3 | $\geq 0.70$ |
| 4 | REM | $[0.20, 0.40)$ + desync |

---

## 2. Theoretical Context

### Sleep as Synchronisation Dynamics

Sleep is fundamentally a phenomenon of cortical synchronisation
(Steriade et al. 1993). The transition from wakefulness to deep
sleep (N3) corresponds to a progressive increase in the coherence
of cortical oscillations:

- **Wakefulness:** Low-amplitude, high-frequency, desynchronised
  activity. Multiple cortical regions oscillate independently.
  $R \approx 0.1 - 0.3$.
- **NREM sleep:** Progressive synchronisation. Thalamocortical
  loops generate increasingly coherent slow oscillations.
  N1 → N2 → N3 corresponds to $R$ increasing from 0.3 to 0.9.
- **REM sleep:** Paradoxical desynchronisation. Cortex returns to
  wake-like low-$R$ activity, but subcortical structures maintain
  sleep-like patterns. The `functional_desync` flag captures this.

### The Kuramoto Model of Cortical Dynamics

The mapping from EEG to $R$ is grounded in the view of cortical
columns as coupled oscillators (Breakspear, 2017):
- Each cortical column oscillates at its natural frequency
- Thalamocortical input provides global coupling
- Sleep is the state where global coupling ($K$) exceeds the
  critical coupling ($K_c$), producing macroscopic synchronisation

The SCPN Phase Orchestrator simulates this directly: N oscillators
with coupling matrix $W$ evolve according to the Kuramoto equation,
and the order parameter $R$ naturally tracks the sleep stage.

### Ultradian Rhythm Modelling

The 90-minute NREM-REM cycle is one of the most robust biological
rhythms (Dement & Kleitman, 1957). The `ultradian_phase` function
provides a simple phase estimate based on N3 anchoring:

**Limitation:** This is a heuristic, not a model. The actual
ultradian rhythm involves the reciprocal interaction of aminergic
and cholinergic brainstem nuclei (McCarley & Hobson, 1975). A full
model would use the Limit Cycle Reciprocal Interaction (LCRI) model,
which is outside the scope of the phase classifier.

### Historical Context

- **Rechtschaffen, A. & Kales, A.** (1968): Original sleep staging
  manual. Defined the stages that AASM later revised.
- **Iber, C. et al.** (2007): AASM Manual — current standard. Merged
  stages 3 and 4 into N3, renamed stages.
- **Steriade, M., McCormick, D. A., & Sejnowski, T. J.** (1993):
  Thalamocortical oscillations in the sleeping and aroused brain.
  Explained the neurophysiology of sleep stages.
- **Dement, W. C. & Kleitman, N.** (1957): Discovery of the
  ~90-minute NREM-REM cycle.
- **McCarley, R. W. & Hobson, J. A.** (1975): Reciprocal interaction
  model of REM/NREM cycling.
- **Breakspear, M.** (2017): "Dynamic models of large-scale brain
  activity." Reviewed Kuramoto-based models of cortical dynamics.

### Clinical Relevance

Automated sleep staging is clinically important:
- **Sleep apnoea:** Stage distribution (reduced N3) is diagnostic
- **Insomnia:** Prolonged N1, reduced N3
- **Narcolepsy:** REM onset within 15 minutes (short $\phi$ to REM)
- **Parasomnias:** Occur in specific stages (N3 for sleepwalking,
  REM for REM behaviour disorder)

The SCPN classifier is not intended for clinical use but provides
a computational analogue for brain-inspired control systems.

---

## 3. Pipeline Position

```
 UPDEEngine.step() ──→ phases
                          │
                          ↓
 compute_order_parameter(phases) ──→ R, ψ
                          │
                          ↓
 ┌── classify_sleep_stage(R, functional_desync) ──┐
 │                                                 │
 │  Input: R ∈ [0, 1], functional_desync: bool    │
 │  Output: stage ∈ {"Wake", "N1", "N2", "N3", "REM"} │
 │                                                 │
 └─────────────────┬───────────────────────────────┘
                   │
                   ↓
 ┌── ultradian_phase(timestamps, stage_history) ──┐
 │                                                 │
 │  Input: epoch timestamps + stage labels         │
 │  Output: φ ∈ [0, 1) — position in 90-min cycle │
 │                                                 │
 └─────────────────┬───────────────────────────────┘
                   │
                   ↓
 RegimeManager / Supervisor decisions
 (e.g., reduce coupling during REM-like states)
```

### Input Contracts

**classify_sleep_stage:**

| Parameter | Type | Range | Meaning |
|-----------|------|-------|---------|
| `R` | `float` | $[0, 1]$ | Kuramoto order parameter |
| `functional_desync` | `bool` | — | REM-like desynchronisation pattern |

**ultradian_phase:**

| Parameter | Type | Shape | Meaning |
|-----------|------|-------|---------|
| `timestamps` | `NDArray[float64]` | `(T,)` | Epoch times in seconds |
| `stage_history` | `list[str]` | `(T,)` | Stage labels per epoch |

### Output Contracts

| Function | Returns | Type | Range |
|----------|---------|------|-------|
| `classify_sleep_stage` | Stage label | `str` | {"Wake", "N1", "N2", "N3", "REM"} |
| `ultradian_phase` | Cycle position | `float` | $[0, 1)$ |

---

## 4. Features

- **AASM-compliant staging** — maps $R$ to standard 5-stage classification
- **Functional desynchronisation flag** — distinguishes REM from wake
  in the low-$R$ overlap region
- **Ultradian phase estimation** — position within the 90-minute cycle
- **N3-anchored cycle tracking** — uses deepest sleep as cycle reference
- **Cyclic wrapping** — ultradian phase wraps modulo 90 minutes
- **Rust FFI for both functions** — classify and ultradian dispatch
  to native code
- **Stage code mapping** — integer codes (0-4) for FFI transfer
- **Configurable thresholds** — threshold constants in both backends
- **Zero-allocation classify** — pure arithmetic, no heap allocation

---

## 5. Usage Examples

### Basic: Classify a Single R Value

```python
from scpn_phase_orchestrator.monitor.sleep_staging import classify_sleep_stage

# Deep sleep
assert classify_sleep_stage(0.85) == "N3"

# Light sleep
assert classify_sleep_stage(0.50) == "N2"

# Drowsy
assert classify_sleep_stage(0.35) == "N1"

# REM (with functional desynchronisation)
assert classify_sleep_stage(0.25, functional_desync=True) == "REM"

# Wake
assert classify_sleep_stage(0.10) == "Wake"
```

### Track Sleep Through a Simulation

```python
import numpy as np
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter
from scpn_phase_orchestrator.monitor.sleep_staging import classify_sleep_stage

N = 32
eng = UPDEEngine(N, dt=0.01)
rng = np.random.default_rng(42)
phases = rng.uniform(0, 2 * np.pi, N)
omegas = np.ones(N)
knm = np.full((N, N), 0.5); np.fill_diagonal(knm, 0.0)
alpha = np.zeros((N, N))

for step in range(500):
    phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
    if step % 100 == 0:
        R, _ = compute_order_parameter(phases)
        stage = classify_sleep_stage(R)
        print(f"Step {step}: R = {R:.4f}, Stage = {stage}")
```

### Ultradian Phase Tracking

```python
import numpy as np
from scpn_phase_orchestrator.monitor.sleep_staging import ultradian_phase

# 100 epochs at 30-second intervals
timestamps = np.arange(100) * 30.0
stages = (["N3"] * 20 + ["N2"] * 20 + ["N1"] * 10 +
          ["REM"] * 20 + ["N2"] * 15 + ["N3"] * 15)

phi = ultradian_phase(timestamps, stages)
print(f"Ultradian phase: {phi:.4f}")
# Phase is measured from last N3 onset
```

### Coupling Adaptation Based on Stage

```python
import numpy as np
from scpn_phase_orchestrator.monitor.sleep_staging import classify_sleep_stage
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

def adapt_coupling(phases, knm_base, scale_map):
    R, _ = compute_order_parameter(phases)
    stage = classify_sleep_stage(R)
    scale = scale_map.get(stage, 1.0)
    return knm_base * scale

# Sleep-state-dependent coupling modulation
scale_map = {
    "Wake": 1.0,    # Full coupling
    "N1": 0.8,      # Slightly reduced
    "N2": 0.6,      # Moderate reduction
    "N3": 0.3,      # Minimal (slow-wave dominance)
    "REM": 0.9,     # Near-wake coupling
}
```

---

## 6. Technical Reference

### Function: classify_sleep_stage

::: scpn_phase_orchestrator.monitor.sleep_staging

### Threshold Constants

| Constant | Python | Rust | Value |
|----------|--------|------|-------|
| N3 threshold | `_STAGE_THRESHOLDS["N3"]` | `N3_THRESHOLD` | 0.70 |
| N2 threshold | `_STAGE_THRESHOLDS["N2"]` | `N2_THRESHOLD` | 0.40 |
| N1 threshold | `_STAGE_THRESHOLDS["N1"]` | `N1_THRESHOLD` | 0.30 |
| REM threshold | `_STAGE_THRESHOLDS["REM"]` | `REM_THRESHOLD` | 0.20 |
| Ultradian period | `_ULTRADIAN_PERIOD_S` | `ULTRADIAN_PERIOD_S` | 5400.0 s |

### Rust Engine Functions

```rust
// Returns stage code: 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM
pub fn classify_sleep_stage(r: f64, functional_desync: bool) -> u8

// Returns phase ∈ [0, 1) from last N3 epoch
pub fn ultradian_phase(timestamps: &[f64], stages: &[u8]) -> f64
```

### Auto-Select Logic

```python
try:
    from spo_kernel import classify_sleep_stage_rust as _rust_classify
    from spo_kernel import ultradian_phase_rust as _rust_ultradian
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False
```

### Stage Code Translation

The Python wrapper translates between string labels and integer
codes at the FFI boundary:

```python
_STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
_STAGE_CODES = {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
```

---

## 7. Performance Benchmarks

Measured on Intel Core i5-11600K @ 3.90 GHz, 32 GB DDR4-2400.

### classify_sleep_stage

| Backend | Time (µs) | Speedup |
|---------|-----------|---------|
| Python | 0.37 | — |
| Rust | 0.42 | **0.9x** |

The Rust path is marginally slower due to FFI call overhead
(~200 ns) exceeding the computation cost (~100 ns). The function
is 5 comparisons — even Python executes this in sub-microsecond.

### ultradian_phase (1000 epochs)

| Backend | Time (µs) | Speedup |
|---------|-----------|---------|
| Python | 1.9 | — |
| Rust | 81.8 | **0.02x** |

The Rust path is significantly slower because the Python→Rust
data marshalling dominates: converting a Python `list[str]` to
`NDArray[uint8]` via `[_STAGE_CODES.get(s, 0) for s in stage_history]`
requires iterating the list in Python, then passing the array
through PyO3. The actual Rust computation (reverse scan for N3)
takes nanoseconds.

**Recommendation:** The Rust path provides no benefit for either
function in the current design. The FFI overhead exceeds the
compute savings. These functions are best kept on the Python path.
The Rust implementations exist for correctness verification and
for use in pure-Rust pipelines (e.g., embedded systems).

### Memory Usage

- `classify_sleep_stage`: Zero allocation (pure comparisons)
- `ultradian_phase`: 1 temporary array for stage codes ($T$ bytes)

### Test Coverage

- **Rust tests:** 10 (sleep_staging module in spo-engine)
  - N3, N2, N1, REM with desync, Wake, Wake low desync,
    ultradian basic, ultradian no N3, ultradian empty, wrapping
- **Python tests:** 14 (`tests/test_sleep_staging.py`)
  - All stage classifications, boundary values, desync flag
    combinations, ultradian phase, cycle wrapping, pipeline
    wiring
- **Source lines:** 147 (Rust) + 122 (Python) = 269 total

---

## 8. Citations

1. **Iber, C., Ancoli-Israel, S., Chesson, A. L., & Quan, S. F.**
   (2007).
   *The AASM Manual for the Scoring of Sleep and Associated Events.*
   American Academy of Sleep Medicine.

2. **Rechtschaffen, A. & Kales, A.** (1968).
   *A Manual of Standardized Terminology, Techniques and Scoring
   System for Sleep Stages of Human Subjects.*
   U.S. Government Printing Office.

3. **Steriade, M., McCormick, D. A., & Sejnowski, T. J.** (1993).
   "Thalamocortical oscillations in the sleeping and aroused brain."
   *Science* 262(5134):679-685.
   DOI: [10.1126/science.8235588](https://doi.org/10.1126/science.8235588)

4. **Dement, W. C. & Kleitman, N.** (1957).
   "Cyclic variations in EEG during sleep and their relation to
   eye movements, body motility, and dreaming."
   *Electroencephalography and Clinical Neurophysiology* 9(4):673-690.
   DOI: [10.1016/0013-4694(57)90088-3](https://doi.org/10.1016/0013-4694(57)90088-3)

5. **McCarley, R. W. & Hobson, J. A.** (1975).
   "Neuronal excitability modulation over the sleep cycle: A
   structural and mathematical model."
   *Science* 189(4196):58-60.
   DOI: [10.1126/science.1135627](https://doi.org/10.1126/science.1135627)

6. **Breakspear, M.** (2017).
   "Dynamic models of large-scale brain activity."
   *Nature Neuroscience* 20(3):340-352.
   DOI: [10.1038/nn.4497](https://doi.org/10.1038/nn.4497)

7. **Borbély, A. A.** (1982).
   "A two process model of sleep regulation."
   *Human Neurobiology* 1(3):195-204.

8. **Tononi, G. & Cirelli, C.** (2006).
   "Sleep function and synaptic homeostasis."
   *Sleep Medicine Reviews* 10(1):49-62.
   DOI: [10.1016/j.smrv.2005.05.002](https://doi.org/10.1016/j.smrv.2005.05.002)

---

## Edge Cases and Limitations

### R Exactly at Threshold

At boundary values (e.g., $R = 0.40$), the stage is determined by
the $\geq$ comparison: $R = 0.40$ → N2 (not N1). This is consistent
across Python and Rust.

### functional_desync Without EEG

The `functional_desync` flag is a boolean input, not computed by
the classifier. In the SCPN pipeline, this flag should be provided
by a separate EEG feature extractor (not included in SPO). Without
EEG data, set `functional_desync=False` — the classifier then
cannot distinguish REM from N1/Wake.

### R > 1 or R < 0

The function does not clamp $R$. Values above 1.0 are classified
as N3; negative values as Wake. This is consistent with the
threshold logic but physically meaningless (R should be in [0, 1]).

### Ultradian Phase Accuracy

The `ultradian_phase` function assumes a fixed 90-minute period.
In reality, the cycle period varies:
- First cycle: ~70-100 minutes
- Later cycles: ~90-120 minutes
- N3 duration decreases through the night
- REM duration increases through the night

For accurate ultradian modelling, use the LCRI model (McCarley &
Hobson, 1975) or Borbély's two-process model.

---

## Troubleshooting

### Issue: All Epochs Classified as Wake

**Diagnosis:** The coupling strength is too low for $R$ to reach
0.30. The oscillators are not synchronising.

**Solution:** Increase coupling in the simulation. Check whether
$K > K_c$ (supercritical coupling).

### Issue: No REM Detected

**Diagnosis:** `functional_desync` is never set to `True`.

**Solution:** Implement a REM detection heuristic based on additional
features (e.g., EMG atonia, rapid eye movements in EOG, or spectral
power ratios).

### Issue: Ultradian Phase Always 0

**Diagnosis:** No N3 epochs in the stage history. The function
returns 0.0 when no N3 anchor is found.

**Solution:** Ensure the simulation reaches high enough $R$ (≥ 0.70)
for N3 classification.

---

## Integration with Other SPO Modules

### With RegimeManager

Sleep stages map to SCPN regime transitions:

| Stage | Regime | Supervisor Action |
|-------|--------|-------------------|
| Wake | NOMINAL | Normal operation |
| N1 | DEGRADED | Reduce monitoring frequency |
| N2 | SLEEP | Engage slow-wave mode |
| N3 | DEEP_SLEEP | Minimal activity |
| REM | DREAM | Engage creative exploration |

### With HCP Connectome

Sleep stages modulate effective connectivity:
- N3: Strong thalamocortical coupling → high effective $W$
- REM: Reduced long-range coupling → weakened inter-hemispheric $W$

```python
stage = classify_sleep_stage(R)
if stage == "N3":
    W_effective = W * 1.5  # Enhanced synchronisation
elif stage == "REM":
    W_effective = W * 0.7  # Reduced coupling
```

### With Ethical Cost

The ethical cost function can incorporate sleep-stage-dependent
safety thresholds:

```python
# In N3, allow lower R_min (system is meant to be deeply synchronised)
if stage == "N3":
    result = compute_ethical_cost(phases, W, R_min=0.6)
elif stage == "REM":
    result = compute_ethical_cost(phases, W, R_min=0.1)
else:
    result = compute_ethical_cost(phases, W, R_min=0.2)
```

### With SSGF Geometry Control

The ultradian phase can modulate the SSGF learning rate:
- Near N3 ($\phi \approx 0$): Low learning rate (stable geometry)
- Near REM ($\phi \approx 0.5$): High learning rate (exploratory)

This models the hypothesised role of sleep in synaptic homeostasis
(Tononi & Cirelli, 2006): N3 consolidates, REM explores.

```python
phi = ultradian_phase(timestamps, stages)
# Sinusoidal modulation: high lr at REM, low at N3
lr = 0.005 + 0.01 * np.sin(np.pi * phi)
carrier = GeometryCarrier(N, lr=lr)
```

### With Transfer Entropy

During N3, cortical synchronisation suppresses information transfer
($TE \approx 0$ when all phases are locked). TE-directed coupling
adaptation should be paused during N3 epochs to avoid falsely
reducing coupling based on uninformative zero-TE measurements.

```python
stage = classify_sleep_stage(R)
if stage != "N3":
    knm = te_adapt_coupling(knm, trajectory, lr=0.01)
# Skip TE update during N3 (TE is uninformative at R ≈ 1)
```
