# Control Systems

SPO is the only oscillator library with closed-loop supervisory control.
TVB, neurolib, Brian2, and NEST are all open-loop simulators.

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
