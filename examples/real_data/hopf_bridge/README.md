# The Hopf bridge — which detector family recovers the eigenvalue

`hopf_bridge_external_validation.json` is a hash-sealed external validation that unifies the
suite's two detector families against one analytic ground truth, on the bifurcation where
they meet.

The [grid ANDES test](../grid_external_validation/README.md) validates the *oscillatory*
envelope-growth family on real systems; the
[critical-slowing-down test](../csd_bifurcation/README.md) validates the *single-series*
autocorrelation family on *non-oscillatory* bifurcations. A **Hopf** bifurcation presents an
*oscillatory* critical slowing down (eigenvalue α ± iω with α → 0⁻), so both families point at
the same mode. We sweep the Hopf parameter α, integrate the stochastic Hopf normal form
(`dr = (α r − r³) dt` on the amplitude, `dθ = ω dt` on the phase), and read each family with
its shipped detector against the analytic α.

## The result, stated honestly

| Family | reads | Spearman ρ | mean magnitude gap |
|--------|-------|:----------:|:------------------:|
| **envelope-growth** | rectified ringdown | **0.97** | **\|σ − α\| = 0.04** |
| autocorrelation | stationary series | 0.97 | \|rate − α\| = 0.69 |

Both families *track* α, but only the **envelope-growth** family recovers it in **magnitude**;
the autocorrelation family's magnitude is **confounded by the oscillation** (pinned near
`cos(ω Δt)`, so `ln(AR1)/Δt` does not estimate α). The envelope recovery is also
**frequency-invariant** — identical across 0.2–1.2 Hz, because it fits the envelope, not the
oscillation.

This completes a **regime map** across the three eigenvalue external validations: the
eigenvalue's real part is the universal quantity, but the magnitude-correct estimator is
regime-dependent — **envelope-growth for an oscillatory mode** (here, and the grid),
**the autocorrelation for a non-oscillatory one** (fold / pitchfork).

## Scope and limits

- **Analytic ground truth** — the Hopf normal-form eigenvalue in closed form.
- **One honest sensitivity, recorded as a curve:** the envelope magnitude recovery **degrades
  with ringdown SNR** (ρ falls from 0.97 to 0.62 as the ringdown noise rises) — a physical
  floor, since a decay cannot be read below the noise it sinks into. The `snr_robustness` field
  seals this curve rather than hiding it.
- **Quasi-static per-α sweep**, additive noise, a scalar reduced normal form.

## Reproduce

Pure NumPy and the two shipped detectors — no external simulator. A fresh run reproduces the
estimates only to floating-point tolerance, so the **seal is recomputed from the committed
rows**, never from a fresh integration — the integrity test
`tests/test_hopf_bridge_external_validation_evidence.py` pins it.

```bash
python bench/hopf_bridge_external_validation.py OUT.json
```

## References

- Strogatz, *Nonlinear Dynamics and Chaos*, 1994 — the Hopf normal form and its α ± iω
  eigenvalues.
- Kundur, *Power System Stability and Control*, 1994 — the electromechanical oscillation whose
  damping the grid detector reads.
