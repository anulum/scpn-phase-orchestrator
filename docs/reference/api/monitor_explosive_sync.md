# Explosive-Synchronisation Early-Warning Monitor

The `monitor.explosive_sync` module raises an early warning before a
**first-order (explosive) synchronisation transition** — the abrupt,
hysteretic collapse to coherence behind power-grid blackouts and seizure
onset. It is the detection layer built on the ordinal-pattern transition
entropy primitive in [`monitor.opt_entropy`](monitor_opt_entropy.md).

The monitor is **passive**: it reads node observables and emits a warning
record. It never actuates — consistent with SPO's review-only posture.

---

## 1. Why ordinal transition entropy is the right signal

Classic critical-slowing-down early-warning signals (rising variance, rising
lag-1 autocorrelation; Scheffer et al. 2009) are tuned to *continuous*
(second-order) bifurcations. A first-order synchronisation transition is
abrupt and need not slow down before it fires, so variance/autocorrelation
react late or not at all.

Ahead of explosive locking, each oscillator's local dynamics instead become
more *predictable*: the diversity of its ordinal-pattern transitions
contracts. That contraction is visible in the transition entropy `H_T` (see
[`monitor.opt_entropy`](monitor_opt_entropy.md) §1.4) well before the
macroscopic order parameter jumps. The monitor watches `H_T` per node and
fires when it drops a robust margin below its leading baseline.

---

## 2. Algorithm

Given a node-by-time signal array `signals` of shape `(N, T)`:

1. **Slide.** Window starts at `0, step, 2·step, …` while a full `window`
   fits; each yields one analysis window.
2. **Local entropy field.** For every window and node, compute
   `transition_entropy(signals[node, start:start+window], dimension, delay)`,
   giving a `(W, N)` field `per_node_entropy`. The window must be long enough
   to admit two ordinal transitions: `window ≥ (D − 1)·τ + 3`.
3. **Aggregate index.** Average across nodes to the headline
   coherence-regularisation index `entropy_index` of shape `(W,)`.
4. **Baseline fit.** Use the leading
   `n_baseline = max(min_baseline_windows, ⌈baseline_fraction · W⌉)` windows.
   The baseline is summarised by its median `m_0` and robust scale
   `s_0 = 1.4826 · MAD` (a normal-consistent standard-deviation estimate). The
   scale is floored at `1e-12` so a perfectly flat baseline does not divide by
   zero.
5. **Robust drop score.** Per window,
   `robust_z = (entropy_index − m_0) / max(s_0, 1e-12)` and
   `relative_drop = (m_0 − entropy_index) / m_0` (zero when `m_0 = 0`).
6. **Alarm.** A window past the baseline *breaches* when both
   `robust_z ≤ −z_threshold` and `relative_drop ≥ drop_threshold`. The alarm
   fires at the first run of `persistence` consecutive breaching windows;
   `warning_window` is the start of that run and `warning_sample` its sample
   index. Requiring two gates (a robust z-score and an absolute fractional
   drop) and a sustained run suppresses single-window noise.

---

## 3. Python API

```python
from scpn_phase_orchestrator.monitor.explosive_sync import (
    explosive_sync_warning,
    ExplosiveSyncWarning,
)

result = explosive_sync_warning(
    signals,            # (N, T) float array; (T,) is treated as one node
    dimension=3,
    delay=1,
    window=128,
    step=16,
    baseline_fraction=0.25,
    min_baseline_windows=3,
    z_threshold=3.0,
    drop_threshold=0.1,
    persistence=2,
)

if result.warning_triggered:
    print("explosive-sync warning at sample", result.warning_sample)
print(result.summary())
```

`ExplosiveSyncWarning` is a frozen dataclass carrying the full diagnostic
record: `window_starts`, `entropy_index`, the `per_node_entropy` field,
`robust_z`, `relative_drop`, the baseline fit (`baseline_median`,
`baseline_scale`, `n_baseline_windows`), the alarm decision
(`warning_triggered`, `warning_window`, `warning_sample`), and the echoed
parameters. `summary()` returns a flat scalar dictionary suitable for logging
or metric export.

### 3.1 Input validation

`signals` must be a finite real one- or two-dimensional array (boolean,
complex, non-finite, and non-numeric inputs are rejected). `window`, `step`,
`min_baseline_windows`, and `persistence` must be positive integers;
`baseline_fraction` must lie in the open interval `(0, 1)`; `z_threshold` and
`drop_threshold` must be finite and non-negative; `window` must fit the series
and admit two ordinal transitions.

---

## 4. Choosing parameters

| Parameter | Effect | Guidance |
|---|---|---|
| `window` | Samples per entropy estimate | Long enough for a stable `H_T` (hundreds of samples for `D = 3`); shorter reacts faster but is noisier. |
| `step` | Hop between windows | Smaller `step` sharpens the lead time at more compute. |
| `baseline_fraction` | Reference span | Large enough to capture normal variability before any transition. |
| `z_threshold` | Robust-drop sensitivity | Higher rejects more noise; `3.0` is a conventional 3-σ-equivalent. |
| `drop_threshold` | Minimum fractional drop | Guards against tiny but statistically sharp drops on a quiet baseline. |
| `persistence` | Sustained-breach length | `≥ 2` rejects single-window flickers. |

The two gates are complementary: `z_threshold` catches drops that are large
*relative to baseline noise*, while `drop_threshold` requires the drop to be
*absolutely* meaningful — necessary because a very flat baseline makes the
robust z-score explode on negligible movement.

---

## 5. Behaviour (tested)

* **Fires on a noise→lock transition** in the neighbourhood of the switch and
  stays silent on stationary noise
  (`test_fires_on_regularisation_transition`, `test_silent_on_stationary_noise`).
* **`persistence` suppresses flickers** — a longer required run never fires
  earlier than a shorter one (`test_persistence_requires_sustained_breach`).
* **A high `drop_threshold` suppresses the warning** even on a real transition
  (`test_high_drop_threshold_suppresses_warning`).
* **The entropy index equals the per-node mean** and is bounded in `[0, 1]`
  (`test_entropy_index_is_node_mean`, `test_entropy_index_bounded`).
* **A flat baseline does not divide by zero** — the scale floor keeps
  `robust_z` finite (`test_zero_scale_does_not_divide_by_zero`).

---

## 6. Pipeline position

```
   signals (N, T) ──▶ explosive_sync_warning ──▶ ExplosiveSyncWarning
        │                      │                        │
        │                      ▼                        ▼
        │            transition_entropy           warning_triggered
        │            (per node, per window)        warning_sample
        ▼                                          entropy_index / per_node field
   any scalar node observable
   (phase velocity, sin θ, …)
```

The monitor consumes any per-node scalar observable (phase velocity, `sin θ`,
power-injection deviation, …) and emits a warning record for a supervisory or
alerting layer. It has no actuation path of its own.

---

## 7. Implementation cross-reference

| File | Role |
|---|---|
| `src/scpn_phase_orchestrator/monitor/explosive_sync.py` | Monitor + `ExplosiveSyncWarning` |
| `src/scpn_phase_orchestrator/monitor/opt_entropy.py` | Transition-entropy compute primitive |
| `tests/test_explosive_sync.py` | Detection, structure, validation, and guard tests |

---

## 8. References

* Scheffer, M. et al. 2009, *Nature* 461, 53 — "Early-warning signals for
  critical transitions" (the slowing-down framework this complements).
* Bandt, C. & Pompe, B. 2002, *Phys. Rev. Lett.* 88, 174102 — permutation
  entropy.
* Gómez-Gardeñes, J., Gómez, S., Arenas, A. & Moreno, Y. 2011, *Phys. Rev.
  Lett.* 106, 128701 — explosive synchronisation as a first-order transition.

---

## 9. API reference

::: scpn_phase_orchestrator.monitor.explosive_sync
