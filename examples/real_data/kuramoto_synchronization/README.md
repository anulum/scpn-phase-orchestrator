# Kuramoto synchronisation — recovering the eigenvalue on the collective coordinate

`kuramoto_synchronization_external_validation.json` is a hash-sealed external validation
that carries the [critical-slowing-down eigenvalue test](../csd_bifurcation/README.md) from
*scalar* normal forms to the **emergent collective coordinate** of a high-dimensional
system: the mean field of the Kuramoto model SCPN is built around.

## The set-up

For the noisy Kuramoto model of `N = 512` phase oscillators with a Lorentzian frequency
law (half-width `γ = 0.5`) and phase diffusion `D = 0.5`, the incoherent state loses
stability at the mean-field critical coupling `K_c = 2(γ + D) = 2.0` — the `K_c = 2γ` of
the Ott–Antonsen reduction, generalised for noise. Just below it the fundamental
mean-field mode is **real and non-oscillatory** with the analytic eigenvalue
`λ(K) = (K − K_c)/2` (Sakaguchi 1988). We sweep `K` below `K_c`, integrate the model at each
operating point, and read the shipped critical-slowing-down detector on two observables of
the mean field `Z = (1/N) Σ exp(i θ)`.

## The result, stated honestly

| Observable | reads | Spearman ρ | fitted slope vs λ | mean \|gap\| |
|------------|-------|:----------:|:-----------------:|:-----------:|
| **`Re(Z)`** (signed mean field) | lag-1 autocorrelation | **0.97** | **1.19 (≈ 1)** | 0.15 |
| `|Z|` (order-parameter amplitude) | lag-1 autocorrelation | 0.98 | 2.66 | 0.80 |

**Both** observables track the eigenvalue in **rank** (ρ ≈ 0.98): the collective coordinate
undergoes textbook critical slowing down as `K → K_c`. But **only the signed `Re(Z)`
recovers λ in magnitude** — its lag-one autocorrelation is `exp(λ Δt)`, so `ln(AR1)/Δt`
fits λ with near-unit slope. The order-parameter amplitude `|Z|` that a practitioner
usually watches is **folded** (rectified), so its autocorrelation is not `exp(λ Δt)`: it
ranks the eigenvalue but sizes it at roughly twice the slope.

This extends the eigenvalue **regime map** from scalar normal forms to the order parameter
of a 512-oscillator system, and adds an actionable operational rule: to read the *distance
to the synchronisation threshold* (the eigenvalue's magnitude), monitor the **signed
mean-field component**, not the order-parameter magnitude, and use an autocorrelation
window long enough that the finite-window bias does not steepen the estimate.

## Scope and limits

- **Analytic ground truth** — the noisy-Kuramoto incoherent-state eigenvalue in closed form.
- **Finite-size correction** — the eigenvalue is the `N → ∞` mean-field result; the runs are
  finite-`N`, so `K_c` and `λ` carry an `O(1/√N)` correction, which is why the signed
  slope is near, not exactly, one.
- **Quasi-static per-coupling sweep**, a Lorentzian frequency law, additive phase noise,
  and coupling below `K_c` only — a damping-ranking test of the incoherent state, not a
  sync / no-sync classification.

## Reproduce

Pure NumPy and the shipped detector — no external simulator. A fresh run reproduces the
estimates only to floating-point tolerance, so the **seal is recomputed from the committed
rows**, never from a fresh integration — the integrity test
`tests/test_kuramoto_synchronization_external_validation_evidence.py` pins it.

```bash
python bench/kuramoto_synchronization_external_validation.py OUT.json
```

## References

- Kuramoto 1975; Strogatz 2000, *From Kuramoto to Crawford*, Physica D 143:1 — the
  mean-field synchronisation transition and `K_c = 2/(π g(0))`.
- Sakaguchi 1988, *Prog. Theor. Phys.* 79:39 — the noisy Kuramoto incoherent-state
  eigenvalue `λ = K/2 − (γ + D)`.
- Scheffer et al. 2009, *Nature* 461:53 — critical slowing down as the generic approach.
