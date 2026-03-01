# Lock Metrics

## Kuramoto Order Parameter (R)

```
R = |mean(exp(i * theta))|
```

Global coherence measure. `R = 1` means perfect phase lock. `R ≈ 0` means incoherent.

Computed per layer by masking the phase vector to oscillators in that layer (`compute_layer_coherence`).

## Phase-Locking Value (PLV)

```
PLV = |mean(exp(i * (phi_a - phi_b)))|
```

Pairwise coherence between two phase time series over a window. `PLV = 1` means constant phase difference. `PLV ≈ 0` means no stable relationship.

Computed by `compute_plv(phases_a, phases_b)` in `upde.order_params`.

## Cross-Layer Alignment Matrix

`UPDEState.cross_layer_alignment` is an `(L x L)` matrix where entry `(i, j)` is the PLV between layers i and j. Symmetric. Diagonal is 1.0 by definition.

Used by the supervisor to detect unwanted cross-layer locking (R_bad objective).

## Lock Signatures

`LayerState.lock_signatures` maps string keys `"{i}_{j}"` to `LockSignature`:

| Field | Type | Meaning |
|-------|------|---------|
| `source_layer` | int | Layer index of source |
| `target_layer` | int | Layer index of target |
| `plv` | float | PLV between the two layers |
| `mean_lag` | float | Mean phase difference (radians) |

## Lock Detection

`CoherenceMonitor.detect_phase_lock(upde_state, threshold=0.9)` returns pairs `(i, j)` where PLV exceeds the threshold.

## Thresholds

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| R_good target | > 0.6 | NOMINAL regime |
| R_good minimum | > 0.3 | above CRITICAL |
| R_bad ceiling | < 0.3 | acceptable |
| PLV lock | > 0.9 | phase-locked pair |

## References

- **[kuramoto1975]** Y. Kuramoto (1975). Self-entrainment of a population of coupled non-linear oscillators. *Lecture Notes in Physics* 39, 420–422. — Order parameter R definition.
- **[lachaux1999]** J.-P. Lachaux et al. (1999). Measuring phase synchrony in brain signals. *Human Brain Mapping* 8, 194–208. — PLV definition and significance thresholds.
