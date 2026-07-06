# Critical-slowing-down external validation — analytic eigenvalue ground truth

`csd_bifurcation_external_validation.json` is a hash-sealed external validation of the
shipped critical-slowing-down (CSD) detector against a ground truth it never sees: the
**analytic** recovery rate at a bifurcation.

The palaeoclimate lead-time capstone certifies the CSD monitor on the Dakos et al. 2008
proxy records — but a proxy record carries no measured recovery rate to check the detector
against. This asks the stronger, non-circular question, and is the single-series analogue of
the [grid eigenvalue validation](../grid_external_validation/README.md) with a harder ground
truth: not a *simulated* eigenvalue but a *closed-form* one. For a quasi-static sweep of a
control parameter toward a codimension-one bifurcation, the scalar stochastic normal form is
integrated at each operating point and the shipped
[`critical_slowing_down_warning`](../../../src/scpn_phase_orchestrator/monitor/critical_slowing_down.py)
reads its two indicators — the lag-one autocorrelation and the variance. The true recovery
rate `λ` is the normal-form **Jacobian eigenvalue** in closed form. The two are correlated.

## The result, stated honestly

| Bifurcation class | normal form | true λ | autocorrelation ρ (rank / magnitude) | variance ρ |
|-------------------|-------------|--------|:------------------------------------:|:----------:|
| Fold (saddle-node) | `dx = (μ − x²) dt` | −2√μ | **0.98** / 0.98 | 0.98 |
| Supercritical pitchfork | `dx = (μx − x³) dt` | μ | **0.96** / 0.96 | 0.98 |

(Spearman rank / Pearson magnitude correlation of the detector indicator versus the true
eigenvalue λ, 14-point sweep each.)

The detector's **autocorrelation** channel recovers the true `λ` on both independent classes
— one with a quadratic nonlinearity, one with a globally confining cubic one. And because the
lag-one autocorrelation of a linear-response process is `exp(λ Δt)`, the implied rate
`ln(AR1)/Δt` tracks `λ` **in magnitude**, not merely in rank — a stronger statement than the
rank-only grid eigenvalue test can make. The **variance** channel rises in step as the
stationary variance `σ²/2|λ|` diverges. So the quantity the detector estimates on the Dakos
records — a falling recovery rate — is exactly what a bifurcation presents.

## Scope and limits

- **Analytic ground truth**, first-principles — the Jacobian eigenvalue of the normal form,
  not a numerical model. This is *stronger* than the grid's ANDES eigenvalue, which is itself
  a simulated quantity.
- **Quasi-static per-operating-point sweep** — an independent stationary run at each control
  value, not a single non-stationary approach to the bifurcation.
- **Additive noise, scalar reduced models** — the canonical low-dimensional pictures of these
  bifurcations, not the full high-dimensional systems.

## Reproduce

Pure NumPy and the shipped detector — no external simulator. A fresh run reproduces the
indicators only to floating-point tolerance (stochastic integration), so the **seal is
recomputed from the committed measurement rows**, never from a fresh integration — the
integrity test `tests/test_csd_bifurcation_external_validation_evidence.py` pins it.

```bash
python bench/csd_bifurcation_external_validation.py OUT.json
```

## References

- Strogatz, *Nonlinear Dynamics and Chaos*, 1994 — the saddle-node and pitchfork normal
  forms and their Jacobian eigenvalues.
- Scheffer et al., *Early-warning signals for critical transitions*, Nature 461:53, 2009 —
  critical slowing down (rising autocorrelation and variance) at a bifurcation.
- Dakos et al., *Slowing down as an early warning signal for abrupt climate change*, PNAS
  105:14308, 2008 — the palaeoclimate records the shipped monitor's lead-time capstone reads.
