# Control Systems

SPO adds a supervision layer over coupled-oscillator dynamics that emits
review-only control *proposals*; TVB, neurolib, Brian2, and NEST are
simulate-and-observe libraries. SPO does not close a control loop on hardware.

## Model-Predictive Controller (MPC)

Predicts R trajectory 10 steps ahead using the Ott-Antonsen mean-field
reduction as a fast forward model. Acts BEFORE degradation, not after.
Detects divergence and reverts to reactive control as fallback.

```python
from scpn_phase_orchestrator.supervisor.predictive import PredictiveSupervisor

supervisor = PredictiveSupervisor(engine, horizon=10)
# supervisor.step() predicts future R, triggers actions preemptively
```

::: scpn_phase_orchestrator.supervisor.predictive

## Regime Manager

Finite state machine for synchronization regimes with hysteresis.
States: NOMINAL, DEGRADED, CRITICAL. Transitions based on R thresholds
with configurable hysteresis bands to prevent oscillation between states.

::: scpn_phase_orchestrator.supervisor.regimes

## Petri Net State Machine

Formal Petri net FSM with guard conditions, token counts, and
priority-based transition firing. Enables formal verification of
safety properties (deadlock freedom, liveness).

```python
from scpn_phase_orchestrator.supervisor.petri_net import PetriNet

net = PetriNet()
net.add_place("nominal", tokens=1)
net.add_place("critical", tokens=0)
net.add_transition("degrade", inputs=["nominal"], outputs=["critical"],
                    guard=lambda ctx: ctx["R"] < 0.3)
```

::: scpn_phase_orchestrator.supervisor.petri_net

## Policy Engine

Rule-based policy evaluation for supervisor actions. Rules define
conditions (R thresholds, boundary violations) and actions (coupling
boost, frequency adjustment, external drive).

::: scpn_phase_orchestrator.supervisor.policy

## Three-Factor Hebbian Plasticity

Coupling adaptation rule: K_ij += lr × eligibility × modulator × gate.

1. **Eligibility**: cos(θ_j - θ_i) — pairwise Hebbian trace
2. **Modulator**: scalar neuromodulatory signal from L16 director
3. **Phase gate**: Boolean from TCBO consciousness boundary

Grounded in Friston 2005 on free energy and synaptic plasticity.

::: scpn_phase_orchestrator.coupling.plasticity

## Transfer Entropy Adaptive Coupling

K_ij(t+1) = (1-decay)·K_ij(t) + lr·TE(i→j)

Unlike symmetric Hebbian learning, transfer entropy breaks symmetry
to detect causal direction. Coupling adapts based on directed
information flow (Lizier 2012).

::: scpn_phase_orchestrator.coupling.te_adaptive

## Audit Trail with Deterministic Replay

SHA256-chained JSONL audit log with per-step regime, R values,
actions, and coupling state. Enables deterministic replay and
cryptographic verification of simulation reproducibility.

::: scpn_phase_orchestrator.audit

## What makes this closed-loop in production

Most oscillator libraries expose observability but stop before actuation.
SPO’s control surface pushes into action selection with three constraints in the
same loop:

- **Prediction:** expected coherence trend via MPC proxy.
- **Safety:** regime and evidence checks before promotion.
- **Governance:** audit-ready proposal records for every non-trivial action.

This changes operational posture from “watch and decide” to
“predict-then-verify-then-act” under explicit constraints.

## Operator operating model

For day-to-day deployment, teams typically configure:

1. A baseline monitor that maps `R` and `chimera_index` to supervisory inputs.
2. A policy layer with conservative defaults.
3. A Petri-NET-safe transition set for state changes.
4. An audit sink that captures state transitions and proposal rationale.

That model keeps tuning sessions reviewable and supports post-incident replay
without manual reconstruction of transient state.

## Stability and rollback behavior

Because state transitions are explicit and regime-gated, operators can define clear
rollback boundaries: if coherence drops or monitor evidence regresses after an action,
policy proposals can be bounded and reverted before the next control cycle.

## Operational placement in closed-loop projects

Use this page when the target outcome is a stable control loop, not just a
monitoring dashboard. The control stack in this repository is built in layers:

- prediction: MPC and monitors produce a near-term risk estimate,
- policy and regime selection: Petri-Net and policy DSL select candidate actions,
- constraint application: projector, boundaries, and imprint constraints limit
  per-cycle movement,
- governance: audit logging and replay preserve every non-trivial change.

A common rollout order is:

1. choose monitor set (`R`, `PLV`, boundary metrics),
2. define objectives and regime thresholds,
3. dry-run policy and projection limits,
4. replay the same sequence with fixed seeds,
5. promote only when both expected and observed lock metrics match the decision
   gate.

Keep this page as a production entry for teams that need a predictable control
path after data onboarding and domain calibration.

## Control posture and evidence contract

Every production setup using this page should keep three artifacts versioned:

- policy definition files (rules, cooldowns, caps),
- latest audit configuration and lockfile choice,
- baseline replay traces from the last accepted tuning cycle.

That set is what allows an operator to compare one control action against
history and answer whether the action improved or degraded the system trajectory.

The supervisor surfaces are intentionally split into prediction, policy, and
governance layers. If one layer is changed, replay the full sequence before
promotion so you can isolate the effect of that change without mixing it with
backend or extractor drift.
