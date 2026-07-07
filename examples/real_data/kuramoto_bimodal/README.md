# Bimodal Kuramoto — recovering the eigenvalue on an oscillatory collective transition

`kuramoto_bimodal_hopf_external_validation.json` is a hash-sealed external validation that
is the **oscillatory counterpart** of the [unimodal Kuramoto test](../kuramoto_synchronization/README.md),
and completes a two-regime map inside the Kuramoto model — the collective analogue of the
[Hopf bridge](../hopf_bridge/README.md).

## The set-up

For a symmetric **bimodal** Lorentzian frequency law (two peaks at `±ω₀`, half-width `Δ`),
the incoherent state loses stability at `K_c = 4Δ`, and for `ω₀ > K/4` the fundamental
eigenvalue is **complex** (Martens et al. 2009, via the Ott–Antonsen reduction):

```
λ_±(K) = K/4 − Δ ± √((K/4)² − ω₀²),   Re(λ) = K/4 − Δ,   Ω = √(ω₀² − (K/4)²).
```

So below `K_c` the order parameter is a **damped oscillation** — the regime where the Hopf
bridge found the *envelope* family, not the autocorrelation, sizes the eigenvalue. With
`N = 4000`, `ω₀ = 1.5`, `Δ = 0.5` (so `K_c = 2.0`), we ringdown from a partially-coherent
start and read two families.

## The result, stated honestly

| Family | reads | Spearman ρ | fitted slope vs Re(λ) | mean \|gap\| |
|--------|-------|:----------:|:---------------------:|:-----------:|
| **envelope-growth** | sub-population `\|Z₊\|` | 0.99 | **0.93 (≈ 1)** | 0.015 |
| autocorrelation | global `Re(Z)` tail | 0.96 | 15.8 | 5.2 |

Both families track `Re(λ)` in **rank**, but **only the envelope-growth family recovers it
in magnitude** (slope ≈ 1). The autocorrelation family is **confounded by the oscillation
`Ω`** — its lag-one autocorrelation is dominated by `cos(Ω Δt)`, so its fitted slope is far
from one: it ranks the eigenvalue but cannot size it.

The key subtlety is the observable. The *global* mean field `Z` is a **standing wave** of two
counter-rotating populations, so `|Z|` oscillates to zero and cannot be fit; the `+ω₀`
**sub-population order parameter `Z₊`** is a single complex mode, so its modulus decays
smoothly as `exp(Re(λ) t)` and the envelope fit recovers `Re(λ)` cleanly.

The seal also records the measured oscillation frequency. The analytic `Ω` is **nearly
constant** across the sweep (1.43–1.49), so it is checked in **value, not rank**: the measured
frequency matches `Ω` to a mean absolute error of 0.11 (~7 %), confirming the mode oscillates
at the predicted frequency — i.e. the eigenvalue is genuinely complex.

## The two-regime collective map

| Transition | eigenvalue | magnitude estimator |
|------------|-----------|---------------------|
| unimodal (non-oscillatory) | real `Re(λ)` | autocorrelation on signed `Re(Z)` |
| **bimodal (oscillatory)** | complex `Re(λ) ± iΩ` | **envelope-growth on `\|Z₊\|`** |

This mirrors the whole study's regime map (non-oscillatory → autocorrelation, oscillatory →
envelope) — now at the collective level of a high-dimensional system.

## Scope and limits

- **Analytic ground truth** — the bimodal incoherent-state eigenvalue in closed form.
- **Finite-`N` ringdown** — noiseless, with an `O(1/√N)` floor bounding how far the envelope
  can be read (the fit uses only the decaying window above the floor).
- **Oscillatory regime `ω₀ > Δ` only**, a quasi-static per-coupling sweep, coupling below
  `K_c`.

## Reproduce

Pure NumPy and the two shipped detectors. A fresh run reproduces the estimates only to
floating-point tolerance, so the **seal is recomputed from the committed rows** — the
integrity test `tests/test_kuramoto_bimodal_hopf_external_validation_evidence.py` pins it.

```bash
python bench/kuramoto_bimodal_hopf_external_validation.py OUT.json
```

## References

- Martens, Barreto, Strogatz, Ott, So, Antonsen 2009, *Exact results for the Kuramoto model
  with a bimodal frequency distribution*, Phys. Rev. E 79:026204.
- Strogatz 2000, *From Kuramoto to Crawford*, Physica D 143:1.
- Scheffer et al. 2009, *Nature* 461:53 — critical slowing down as the generic approach.
