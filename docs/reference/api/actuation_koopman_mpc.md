<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# Koopman MPC — Review-Only Convex Model Predictive Controller

`actuation.koopman_mpc` turns a fitted Koopman predictor into a convex
model-predictive controller. Because the Koopman model
(`monitor.koopman_edmd`) is linear, predictive control over it is a single
**convex quadratic programme** — no nonlinear optimisation, no local minima.
The controller is **review-only**: it returns a content-hashed proposal and
never actuates; the first proposed input is the action a safety envelope
(`actuation.foundation_model_governor` / `actuation.control_barrier`) admits,
constrains, or rejects.

It is not a selectable live `runtime.simulation.simulate()` mode. The generic
binding-spec simulator accepts only `control_mode="supervisor_policy"` and fails
closed for `koopman_mpc`; the Koopman MPC remains in the offline dVOC damping and
FMI co-simulation surfaces where the fitted predictor and plant boundary are
explicit.

## 1. The condensed quadratic programme

Over a horizon `H` the lifted states are eliminated so the only decision
variable is the input sequence `U` (Korda & Mezić 2018, eq. 24); the online cost
is independent of the lift dimension `N`. The predicted outputs stack as

    Y = Ψ ψ(x_k) + Θ U,        Ψ_i = C Aⁱ,        Θ_{i,j} = C A^{i-1-j} B (j < i),

and the controller minimises the tracking-and-effort cost

    Σ_{i=1}^{H} (y_i − r)ᵀ Q (y_i − r) + u_{i-1}ᵀ R u_{i-1} + (y_H − r)ᵀ Q_f (y_H − r)

subject to actuator bounds `u_min ≤ u_i ≤ u_max` and optional move limits
`|u_i − u_{i-1}| ≤ Δ`. Condensing gives `min ½UᵀPU + qᵀU` with
`P = 2(Θᵀ Q̄ Θ + R̄)` and `q = 2 Θᵀ Q̄ (Ψ ψ(x_k) − r̄)`.

> The basic formulation penalises `u` (not `u − u_eq`), so it regulates to an
> equilibrium (oscillation damping) with no offset but tracks a
> non-equilibrium-input set-point with a small steady-state offset.

## 2. The QP layer

The quadratic programme is solved by `actuation._qp`, whose **canonical** path is
a deterministic operator-splitting (ADMM) solver — the OSQP algorithm of Stellato
et al. (2020), including the **adaptive-ρ** re-scaling that lets it converge on
ill-conditioned predictive-control programmes. A review-only controller must
produce a reproducible, content-hashable decision, so the deterministic floor is
the default; the optional `osqp` C solver (the `mpc` extra) is held to the ADMM
result by the parity gate (`1e-5`) and is never the silent default.

## 3. Python API

```python
from scpn_phase_orchestrator.monitor.koopman_edmd import (
    KoopmanDictionary, fit_koopman_predictor,
)
from scpn_phase_orchestrator.actuation.koopman_mpc import (
    KoopmanMPCConfig, KoopmanMPCController,
)

predictor = fit_koopman_predictor(states, next_states, inputs, dictionary=...)
controller = KoopmanMPCController(
    predictor=predictor,
    config=KoopmanMPCConfig(horizon=20, input_lower=-1.0, input_upper=1.0),
)
decision = controller.solve(current_state)        # review-only proposal
action = decision.proposed_input                  # hand to the safety governor
```

`KoopmanMPCController.solve` returns a frozen `KoopmanMPCDecision` carrying the
first proposed input, the full input plan, the predicted output trajectory, the
objective, an `OPTIMAL`/`MAX_ITER` status, an active-bound flag, and the SHA-256
`content_hash` of the rounded payload.

## 4. Tested behaviour

- **Oscillation damping** — closed-loop regulation drives a lightly damped
  oscillatory plant from `‖x‖≈3.8` (uncontrolled) to `≈0`.
- **Set-point tracking** — drives the state substantially toward a reachable
  equilibrium.
- **Constraints** — actuator bounds and move limits are satisfied; the QP
  reports `OPTIMAL`.
- **Reproducibility** — the same inputs yield the same content hash.
- **QP parity** — the ADMM floor matches `osqp` to `1e-5` on random programmes.
- **Composition** — the proposed input flows into the foundation-model governor.

## 5. Pipeline position

`oscillation_modes` / `modal_participation` (monitor) → `koopman_edmd` (model) →
**`koopman_mpc` (control, review-only)** → `foundation_model_governor` /
`control_barrier` (safety envelope) → `prc_oscillation` (assurance). The
controller proposes; the envelope gates; nothing actuates without that review.

## 6. References

- Korda & Mezić 2018, *Automatica* 93, 149-160 (arXiv:1611.03537) — Koopman
  operator meets MPC.
- Stellato, Banjac, Goulart, Bemporad & Boyd 2020, *Math. Program. Comput.* 12,
  637-672 (arXiv:1711.08013) — OSQP: an operator splitting solver for QPs.

## 7. API reference

::: scpn_phase_orchestrator.actuation.koopman_mpc

## 8. Closed-loop oscillation-damping pipeline

`runtime.dvoc_oscillation_damping` closes the dVOC loop end to end: an
underdamped oscillator rings down and the matrix-pencil estimator plus the NERC
PRC screener flag its poorly-damped mode; an EDMD-with-control Koopman predictor
is fitted and driven in closed loop by the Koopman MPC; the controlled ringdown
is re-screened and the weakest mode is now better damped. The result carries both
hash-sealed `PRCOscillationEvidence` records, so the damping improvement is
auditable. The `spo koopman-mpc` command runs this pipeline on a default grid
oscillator. The pipeline is review-only and offline — it performs no live
actuation.

::: scpn_phase_orchestrator.runtime.dvoc_oscillation_damping
