# Cross-system external validation — eigenvalue ground truth

`grid_eigenvalue_external_validation.json` is a hash-sealed external validation of the grid
modal-growth detector on **independent** power systems, against a ground truth the detector
never sees: the true small-signal eigenvalue.

The PSML head-to-head certifies the detector on one 23-bus corpus. This asks a stronger,
non-circular question: does the quantity the detector measures — the dominant
electromechanical mode's growth rate σ — hold on other systems? For a sweep of operating
points on the **IEEE 39-bus New England** system and the **Kundur two-area** system (both in
the open-source [ANDES](https://github.com/CURENT/andes) simulator), the true dominant σ is
computed by ANDES **small-signal eigenvalue analysis** — a completely different method from
the detector's time-domain envelope slope — and the detector is run on a ringdown from a
small disturbance at each point. The two σ are then correlated.

## The result, stated honestly

| System | coherent (mean) ρ | coherent (spatial) ρ | focal ρ |
|--------|------------------:|---------------------:|--------:|
| IEEE-39 | **0.87** | 0.47 | −0.59 |
| Kundur | 0.60 | **0.66** | −0.27 |

(Spearman rank correlation of detector σ versus true eigenvalue σ, 14-point sweep each.)

The detector's **coherent** aggregation — the cross-bus mean, or the dominant spatial mode —
recovers the true σ trend on **both** independent systems, so the dominant-mode damping the
detector estimates **generalises across systems and simulators**. The **focal** aggregation
— the PSML winner — does **not** transfer (negative correlation): on a slow coherent
inter-area mode, the per-bus maximum locks onto spurious local excursions rather than the
network mode. So the growth-rate quantity is universal, but the best **aggregation is
regime-dependent**: focal for PSML's fast, localised oscillations, coherent for the slow
inter-area modes here.

## Scope and limits

- **Simulated ground truth**, not field PMU data — it validates the detector's core quantity
  against an independent, first-principles eigenvalue computation, not sim-to-real transfer.
- **Stable operating points only.** Well-damped standard benchmarks (IEEE-39, Kundur) either
  damp or lose synchronism and diverge; they do not enter a clean growing-oscillation regime
  without delicate, case-specific tuning that would weaken the test. So this is a
  **damping-ranking** validation, not a stable-versus-unstable classification.
- Fourteen operating points per system.

## Reproduce

Requires `andes` (`pip install andes`) and the detector on the path. A fresh run reproduces
the growth rate only to floating-point tolerance (nonlinear time-domain integration), so the
**seal is recomputed from the committed measurement rows**, never from a fresh simulation —
the integrity test `tests/test_grid_eigenvalue_external_validation_evidence.py` pins it.

```bash
python bench/grid_eigenvalue_external_validation.py OUT.json
```

## References

- Kundur, *Power System Stability and Control*, 1994 — small-signal (modal) stability and
  the two-area inter-area oscillation benchmark.
- Cui, Li & Tomsovic, *Hybrid Symbolic-Numeric Framework for Power System Modeling and
  Analysis* (ANDES), IEEE Trans. Power Systems, 2021.
