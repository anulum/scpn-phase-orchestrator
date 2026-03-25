# Supervisor

The supervisor subsystem provides closed-loop control of oscillator
dynamics — the feature that distinguishes SPO from all other oscillator
libraries. TVB, neurolib, Brian2, and NEST are open-loop simulators.
SPO's supervisor predicts, detects, and corrects synchronization problems
in real time.

## Regime Manager

Finite state machine for synchronization regimes with hysteresis.
States: NOMINAL → DEGRADED → CRITICAL. Transitions based on R thresholds
with configurable hysteresis bands to prevent oscillation between states.

Optional Rust acceleration via `spo_kernel.PyRegimeManager`.

::: scpn_phase_orchestrator.supervisor.regimes

## Policy Engine

Rule-based evaluation of supervisor actions. Rules define conditions
(R thresholds, boundary violations, regime state) and actions (coupling
boost, frequency adjustment, external drive modification). Rules are
evaluated in priority order; first match fires.

::: scpn_phase_orchestrator.supervisor.policy

## Policy Rules

Individual rule definitions for the policy engine. Each rule has a
condition (Python callable or declarative predicate) and an action
(parameter modification).

::: scpn_phase_orchestrator.supervisor.policy_rules

## Petri Net FSM

Formal Petri net state machine with places (token containers),
transitions (guarded actions), and arcs (token flow). Enables formal
verification of safety properties: deadlock freedom, liveness, and
bounded token counts.

Guards are evaluated from a context dictionary containing current R,
boundary violations, regime state, and step count. First-match priority.

::: scpn_phase_orchestrator.supervisor.petri_net

## Petri Net Adapter

Bridge between the UPDE engine state and the Petri net FSM. Converts
oscillator metrics (R, frequency spread, boundary flags) into the
context dictionary consumed by Petri net guards.

::: scpn_phase_orchestrator.supervisor.petri_adapter

## Event Bus

Publish-subscribe event system for supervisor actions. Monitors,
regime manager, and policy engine communicate via typed events
(RegimeChanged, BoundaryViolation, ActionFired).

::: scpn_phase_orchestrator.supervisor.events

## Model-Predictive Controller (MPC)

Predicts the order parameter R trajectory 10 steps ahead using the
Ott-Antonsen mean-field reduction as a fast forward model. Acts
BEFORE degradation occurs (anticipatory control), not after.

Falls back to reactive control when the OA prediction diverges from
measured R (divergence_threshold configurable). This makes the MPC
safe — it never acts on a prediction it can't trust.

The MPC is O(1) per prediction step (via OA reduction) versus O(N)
for running the full Kuramoto forward model, enabling real-time
control even for large networks.

::: scpn_phase_orchestrator.supervisor.predictive
