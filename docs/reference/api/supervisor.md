# Supervisor

The supervisor subsystem provides closed-loop control of oscillator
dynamics — the feature that distinguishes SPO from all other oscillator
libraries. TVB, neurolib, Brian2, and NEST are open-loop simulators.
SPO's supervisor predicts, detects, and corrects synchronisation problems
in real time.

## Pipeline position

```
UPDEEngine.step() ──→ phases ──→ compute_order_parameter()
                                          │
                                          ↓
                                   UPDEState (R, ψ, locks)
                                          │
                        ┌─────────────────┼─────────────────┐
                        ↓                 ↓                 ↓
                 RegimeManager     PetriNetAdapter   PredictiveSupervisor
                        │                 │                 │
                        └─────────┬───────┘                 │
                                  ↓                         ↓
                          SupervisorPolicy.decide()  ←──────┘
                                  │
                                  ├──→ CausalInterventionEngine
                                  │        (baseline vs intervention rollout)
                                  │
                                  ↓
                        list[ControlAction]
                                  │
                                  ↓
                         ActionProjector.project()
                                  │
                                  ↓
                         ActuationMapper.map_actions()
```

The supervisor sits between the engine output and the next engine step.
It consumes `UPDEState` and `BoundaryState`, produces `ControlAction`
instructions that modify `K_nm`, `ζ`, `Ψ`, or `ω` for the next step.

---

## Regime Manager

Finite state machine for synchronisation regimes with hysteresis,
cooldown, and event logging.

### Regime enum

| Value | Meaning | R range (default thresholds) |
|-------|---------|-----|
| `NOMINAL` | Healthy synchronisation | R ≥ 0.6 |
| `DEGRADED` | Partial desynchronisation | 0.3 ≤ R < 0.6 |
| `CRITICAL` | Synchronisation failure | R < 0.3 or hard violation |
| `RECOVERY` | Transitioning from CRITICAL | CRITICAL → R improving |

### Safety requirement SR-3

CRITICAL **must** pass through RECOVERY before reaching NOMINAL.
Direct CRITICAL → NOMINAL is forbidden. This prevents premature
resumption of normal operation after a synchronisation failure.

### Constructor

```python
RegimeManager(
    hysteresis: float = 0.05,      # band around thresholds
    cooldown_steps: int = 10,      # steps before next transition
    event_bus: EventBus | None = None,
    hysteresis_hold_steps: int = 0,  # consecutive proposals needed
)
```

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `evaluate` | `(UPDEState, BoundaryState) → Regime` | Proposes regime from metrics |
| `transition` | `(Regime) → Regime` | Applies FSM rules, returns actual |
| `force_transition` | `(Regime) → Regime` | Bypasses cooldown |

### Hysteresis

To prevent oscillation between regimes when R is near a threshold,
the manager applies a hysteresis band:

```
NOMINAL → DEGRADED: requires R < threshold - hysteresis
DEGRADED → NOMINAL: requires R > threshold + hysteresis
```

`hysteresis_hold_steps` adds an additional guard: the proposed regime
must be proposed for N consecutive steps before the transition fires.
CRITICAL always bypasses this hold (safety override).

### Cooldown

After a transition, subsequent non-CRITICAL transitions are blocked
for `cooldown_steps` evaluations. CRITICAL always bypasses cooldown.

### Transition history

`transition_history: deque[tuple[int, Regime, Regime]]` stores the
last 100 transitions as (step_number, old_regime, new_regime).

**Performance:** `evaluate()` < 10 μs.

::: scpn_phase_orchestrator.supervisor.regimes

---

## Higher-Order Topology Adaptation

`HigherOrderTopologySupervisor` is the first supervisor-side topology editor.
It consumes live phases plus the current pairwise `K_nm` matrix and returns a
next-step topology:

- bounded pairwise coupling updates from local phase alignment
- optional triadic `Hyperedge` proposals when global coherence is below target
- pruning of stale or incoherent higher-order edges
- serialisable audit metadata for added/pruned simplices and pairwise delta norm

The core control knob is `TopologyMutationPolicy.mutation_rate`. A value of
`0.0` freezes topology; larger values increase the maximum per-step pairwise
and triadic changes while preserving non-negative couplings and a zero
diagonal. `TopologyMutationPolicy.simplex_pairwise_support_floor` is the
policy-hardening gate for deployment reviews: a candidate 2-simplex is only
created when every pairwise edge inside that triad is already at or above the
configured support floor.

```python
import numpy as np

from scpn_phase_orchestrator.supervisor import (
    HigherOrderTopologySupervisor,
    TopologyMutationPolicy,
)
from scpn_phase_orchestrator.upde.hypergraph import HypergraphEngine

policy = TopologyMutationPolicy(mutation_rate=0.2, coherence_floor=0.8)
topology = HigherOrderTopologySupervisor(policy)
result = topology.mutate(phases, knm)

engine = HypergraphEngine(len(phases), dt=0.01, hyperedges=list(result.hyperedges))
next_phases = engine.step(phases, omegas, pairwise_knm=result.knm)
audit_payload = result.to_audit_record()
```

This slice does not claim autonomous online structural control. It provides the
auditable mutation primitive that existing policy, causal, STL, simplicial, and
hypergraph paths can gate before applying a topology change.

Domainpack demos:

- `domainpacks/plasma_control/topology_adaptation_demo.py` runs one guarded
  mutation against the plasma-control binding and prints the audit payload as
  JSON.
- `domainpacks/traffic_flow/topology_adaptation_demo.py` builds pairwise
  support from transfer-entropy evidence before proposing traffic-corridor
  simplices, then records Lyapunov before/after energy and basin evidence for
  the proposed mutation.

::: scpn_phase_orchestrator.supervisor.topology

---

## Strange-Loop Supervisor Monitor

`StrangeLoopSupervisor` is the first self-referential supervisor slice. It
treats the supervisor's own action stream as a four-dimensional control
channel over `K`, `alpha`, `zeta`, and `Psi`. The monitor records recent
action bundles, computes a control phase, control coherence, drift score,
oscillation score, and over-control score, then returns conservative damping
recommendations for a normal policy or safety gate to approve.

```python
from scpn_phase_orchestrator.supervisor import StrangeLoopSupervisor

loop = StrangeLoopSupervisor(overcontrol_threshold=0.2)
assessment = loop.observe(actions_from_supervisor_policy)

if assessment.recommended_actions:
    audit_payload = assessment.to_audit_record()
```

This slice does not hot-patch the supervisor or claim autonomous
self-awareness. It provides an auditable meta-control signal that can detect
policy drift, control-loop oscillation, and excessive actuation before those
dynamics are fed back into the plant.

::: scpn_phase_orchestrator.supervisor.strange_loop

---

## Morphogenetic Topology Field

`MorphogeneticTopologySupervisor` evolves a persistent field over the pairwise
coupling topology. Each step combines:

- pairwise phase-alignment reaction terms
- incident-edge diffusion over the current topology field
- bounded growth and shrink rates
- a hard maximum per-step coupling delta

The result is a next-step `K_nm`, a carried `MorphogeneticFieldState`, grown and
shrunk edge lists, and compact field statistics for audit logs.

```python
from scpn_phase_orchestrator.supervisor import MorphogeneticTopologySupervisor

supervisor = MorphogeneticTopologySupervisor()
result = supervisor.step(phases, knm)

next_knm = result.knm
field_state = result.field_state
audit_payload = result.to_audit_record()
```

This slice provides a reviewable grow/shrink primitive for topology shaping. It
does not bypass the existing policy, causal, STL, or action-projection gates.

::: scpn_phase_orchestrator.supervisor.morphogenetic

---

## Sheaf Coherence Supervisor

`SheafCoherenceSupervisor` evaluates N-channel node states against directed
restriction maps. It builds the block sheaf Laplacian, computes edge residuals,
and reports obstruction metrics for audit logs.

This is the first supervisor-facing sheaf-cohomology slice: it exposes
obstruction score, consistency energy, approximate kernel dimension, and
obstruction dimension. It does not claim a complete formal proof system or
autonomous sheaf-control loop.

```python
from scpn_phase_orchestrator.supervisor import SheafCoherenceSupervisor

supervisor = SheafCoherenceSupervisor(tolerance=1e-8)
result = supervisor.assess(node_states, restriction_maps)

if result.obstruction_score > 0.1:
    audit_payload = result.to_audit_record()
```

::: scpn_phase_orchestrator.supervisor.sheaf

---

## Value-Alignment Guard

`ValueAlignmentGuard` is a hard safety wrapper around proposed
`ControlAction` lists. It evaluates explicit objective constraints, blocks
violating actions, and returns a forced fallback action set when the proposal
does not satisfy the configured score threshold.

The guard is intentionally simple and auditable: no hidden reward model is
loaded at runtime. Domainpacks can translate their safety or objective priors
into `ValueConstraint` entries and attach the resulting decision record to the
normal audit trace.

Binding specs may carry the same policy as a reviewable `value_alignment`
template:

```yaml
value_alignment:
  minimum_score: 0.8
  constraints:
    - name: limit-coupling
      knob: K
      scope: global
      max_abs_value: 0.1
      weight: 2.0
  fallback_actions:
    - knob: zeta
      scope: global
      value: 0.0
      ttl_s: 1.0
      justification: value guard safe hold
```

Use `value_alignment_policy_from_binding_spec(spec)` to convert that template
into a `ValueAlignmentPolicy`. Audit records include hard bound violations and
score-threshold counterfactuals so reviewers can distinguish a blocked unsafe
action from a fallback forced by the policy's minimum alignment score.

```python
from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.supervisor import (
    ValueAlignmentGuard,
    ValueAlignmentPolicy,
    ValueConstraint,
    value_alignment_policy_from_binding_spec,
)

policy = ValueAlignmentPolicy(
    constraints=(ValueConstraint("limit-coupling", knob="K", max_abs_value=0.1),),
    fallback_actions=(
        ControlAction("zeta", "global", 0.0, 1.0, "alignment fallback: hold"),
    ),
)
decision = ValueAlignmentGuard(policy).evaluate(proposed_actions)
actions_to_apply = decision.actions_to_apply
audit_payload = decision.to_audit_record()

templated_policy = value_alignment_policy_from_binding_spec(binding_spec)
```

::: scpn_phase_orchestrator.supervisor.alignment

---

## Policy Engine

Rule-based evaluation of supervisor actions.

### SupervisorPolicy

```python
SupervisorPolicy(
    regime_manager: RegimeManager,
    petri_adapter: PetriNetAdapter | None = None,
)
```

### decide()

```python
def decide(
    upde_state: UPDEState,
    boundary_state: BoundaryState,
    petri_ctx: dict[str, float] | None = None,
) -> list[ControlAction]
```

Returns a list of `ControlAction` instructions. Each action specifies:

| Field | Type | Example |
|-------|------|---------|
| `knob` | `str` | `"K"`, `"zeta"`, `"psi"` |
| `scope` | `str` | `"global"`, `"layer_0"` |
| `value` | `float` | `0.05` (K boost), `0.1` (zeta damp) |
| `ttl_s` | `float` | `5.0` (action expires after 5s) |
| `justification` | `str` | `"degraded: K boost"` |

### Regime-action mapping

| Regime | Actions |
|--------|---------|
| NOMINAL | None (no intervention) |
| DEGRADED | K boost +0.05 (global) |
| CRITICAL | ζ damping +0.1 + K reduce -0.03 (worst layer) |
| RECOVERY | K restore +0.025 (half boost, global) |

### Hard violation override

Hard boundary violations (`BoundaryState.hard_violations`) force
CRITICAL regardless of R values.

**Performance:** `decide()` < 50 μs.

::: scpn_phase_orchestrator.supervisor.policy

---

## Causal Counterfactual Rollouts

`CausalInterventionEngine` evaluates proposed supervisor actions by running
paired UPDE trajectories from the same state:

- baseline: no action
- intervention: action-adjusted `K`, `alpha`, `zeta`, or `Psi`

The result is a `CounterfactualRollout` with `R` and `Psi` trajectories,
final and mean `R` deltas, signed final phase delta, and a serialisable audit
payload.

```python
from scpn_phase_orchestrator.supervisor import CausalInterventionEngine

engine = CausalInterventionEngine(n_oscillators=8, dt=0.01, horizon=20)
rollout = engine.evaluate_actions(phases, omegas, knm, alpha, 0.0, 0.0, actions)
record = rollout.to_audit_record()
attribution = rollout.attribute(threshold=1e-3).to_audit_record()
```

This is the first causal-supervision slice: it does not claim formal
do-calculus yet, but it makes every proposed actuation comparable against a
no-action counterfactual under the same UPDE dynamics.

`CounterfactualRollout.attribute()` compresses the final and mean `R` deltas
into an audit-ready effect label: `stabilising`, `neutral`, or `destabilising`.

`learn_causal_graph()` adds a lightweight live causal-model learner. It
estimates signed directed edges from lagged monitor traces and appends explicit
`do(knob:scope) -> R` edges from paired counterfactual rollouts. The output is
a `CausalGraphEstimate` with JSON-safe nodes, edge weights, confidence scores,
lags, and evidence labels for the audit trail.

```python
from scpn_phase_orchestrator.supervisor import learn_causal_graph

graph = learn_causal_graph(
    {"R_good": good_trace, "R_bad": bad_trace},
    [rollout],
    lag=1,
    min_abs_weight=1e-4,
)
audit_graph = graph.to_audit_record()
```

Domainpack demos:

- `domainpacks/cardiac_rhythm/causal_attribution_demo.py` evaluates a
  pacing-drive candidate against a ventricular-disturbance baseline.
- `domainpacks/power_grid/causal_attribution_demo.py` evaluates a governor
  droop coupling candidate against a no-action load-step baseline.

**Backend and cost:** each evaluation performs two UPDE rollouts over the
configured horizon, so work scales with `2 * horizon` engine steps. It uses
the existing `UPDEEngine` backend dispatcher; Rust acceleration is used when
available, otherwise the NumPy path is used.

::: scpn_phase_orchestrator.supervisor.causal

---

## Policy Rules (Declarative)

Declarative rules loaded from YAML/JSON configuration.

### Data model

```python
PolicyCondition(metric: str, layer: int | None, op: str, threshold: float)
CompoundCondition(conditions: list[PolicyCondition], logic: str = "AND")
PolicyAction(knob: str, scope: str, value: float, ttl_s: float)
PolicyRule(
    name: str,
    regimes: list[str],       # active in these regimes
    condition: PolicyCondition | CompoundCondition,
    actions: list[PolicyAction],
    cooldown_s: float = 0.0,  # min seconds between firings
    max_fires: int = 0,       # 0 = unlimited
)
```

### STL Monitors In Policy YAML

Policy files may also declare reviewable Signal Temporal Logic monitors under
top-level `stl_monitors`. These monitors do not emit control actions directly;
they evaluate scalar traces and return audit records that can be used by the
runtime gate or safety review job.

```yaml
rules: []
stl_monitors:
  - name: keep_sync
    spec: always (R >= 0.3)
    severity: hard
  - name: eventual_recovery
    spec: eventually (R >= 0.8)
```

```python
from scpn_phase_orchestrator.supervisor.policy_rules import (
    evaluate_policy_stl_specs,
    load_policy_stl_specs,
)

specs = load_policy_stl_specs("policy.yaml")
results = evaluate_policy_stl_specs(specs, {"R": [0.2, 0.4, 0.9]})
audit_payloads = [result.to_audit_record() for result in results]
```

### PolicyEngine

```python
engine = PolicyEngine(rules)
engine.advance_clock(dt)
actions = engine.evaluate(regime, upde_state, good_layers, bad_layers)
```

Rules are evaluated in list order. Each rule fires if:
1. Current regime is in `rule.regimes`
2. Condition evaluates True against UPDEState metrics
3. Cooldown has expired since last firing
4. `max_fires` not exceeded

`load_policy_rules(path)` loads rules from YAML/JSON file.

::: scpn_phase_orchestrator.supervisor.policy_rules

## Policy Diagnostics

Dry-run helpers for validating policy reachability, overlap, cooldown, and
action output before a rule set is allowed into a live supervisor path.

::: scpn_phase_orchestrator.supervisor.policy_diagnostics

## Formal Export

Export helpers translate Petri-net, policy-rule, and policy-declared STL
surfaces into model-checker artefacts for independent safety analysis. PRISM
exports remain the default; TLA+ modules are available for protocol and policy
transition-system checks.

The CLI supports:

```bash
spo formal-export domainpacks/my_domain/binding_spec.yaml --export protocol
spo formal-export domainpacks/my_domain/binding_spec.yaml --export policy
spo formal-export domainpacks/my_domain/binding_spec.yaml --export stl
spo formal-export domainpacks/my_domain/binding_spec.yaml --export protocol-tla
spo formal-export domainpacks/my_domain/binding_spec.yaml --export policy-tla
```

`--export stl` reads `stl_monitors` from the sibling `policy.yaml` by default
and emits signal constants plus satisfied/violated labels for the builtin STL
subset. This is a model-checker linkage surface; full temporal automata
synthesis remains future work. `--export protocol-tla` emits a bounded TLA+
module with Petri places as variables, transition guards as constants, `Init`,
`Next`, `Spec`, and `Safety == TypeOK`. `--export policy-tla` emits bounded
rule-fire counters plus reachability predicates for fired rules and emitted
actions.

::: scpn_phase_orchestrator.supervisor.formal_export

---

## Petri Net FSM

Formal Petri net state machine enabling formal verification of
safety properties: deadlock freedom, liveness, bounded token counts.

### Components

| Class | Fields | Description |
|-------|--------|-------------|
| `Place` | `name: str` | Token container (regime state) |
| `Arc` | `place: str, weight: int` | Token flow edge |
| `Guard` | `metric: str, op: str, threshold: float` | Firing condition |
| `Transition` | `name, inputs, outputs, guard` | Guarded state change |
| `Marking` | `tokens: dict[str, int]` | Current token distribution |

### Guard operators

Guards support five comparison operators: `>`, `>=`, `<`, `<=`, `==`.
`Guard.evaluate(ctx)` checks the condition against a context dictionary.

### PetriNet methods

| Method | Description |
|--------|-------------|
| `enabled(marking, ctx)` | Returns transitions whose guards pass |
| `fire(marking, transition)` | Moves tokens and returns new marking |
| `step(marking, ctx)` | Fires first enabled transition |

`parse_guard("R < 0.3")` parses a string into a Guard object.

**Performance:** `enabled_transitions()` < 10 μs.

::: scpn_phase_orchestrator.supervisor.petri_net

---

## Petri Net Adapter

Bridge between UPDEState and the Petri net FSM.

```python
PetriNetAdapter(
    net: PetriNet,
    initial_marking: Marking,
    place_to_regime: dict[str, str],  # maps place names to regime names
    event_bus: EventBus | None = None,
)
```

`adapter.step(ctx)` evaluates the Petri net with the given context and
returns the current `Regime` based on which place holds the token.

::: scpn_phase_orchestrator.supervisor.petri_adapter

---

## Event Bus

Publish-subscribe system for supervisor events.

### RegimeEvent (frozen dataclass)

| Field | Type | Description |
|-------|------|-------------|
| `kind` | `str` | `"regime_transition"` or `"boundary_violation"` |
| `step` | `int` | Step number when event occurred |
| `detail` | `str` | Human-readable description |

### EventBus

```python
bus = EventBus(maxlen=200)
bus.subscribe(callback)
bus.post(RegimeEvent(kind="regime_transition", step=42, detail="nominal->degraded"))
bus.history  # list of all events
bus.count    # total events posted
```

Events are stored in a bounded deque (default 200). Subscribers are
called synchronously on `post()`.

::: scpn_phase_orchestrator.supervisor.events

---

## Model-Predictive Controller (MPC)

Anticipatory control using Ott-Antonsen mean-field reduction.

### Prediction (dataclass)

| Field | Type | Description |
|-------|------|-------------|
| `R_predicted` | `list[float]` | Predicted R trajectory (horizon steps) |
| `will_degrade` | `bool` | R predicted to cross DEGRADED threshold |
| `will_critical` | `bool` | R predicted to cross CRITICAL threshold |
| `steps_to_degradation` | `int` | Steps until predicted degradation |

### PredictiveSupervisor

```python
PredictiveSupervisor(
    n_oscillators: int,
    dt: float,
    horizon: int = 10,              # prediction steps ahead
    divergence_threshold: float = 0.3,  # OA model trust threshold
)
```

**Methods:**

- `predict(phases, omegas, knm, alpha) → Prediction` — runs OA forward
  model for `horizon` steps, returns trajectory
- `decide(phases, omegas, knm, alpha, upde_state, boundary_state)
  → list[ControlAction]` — predicts then acts if degradation imminent

### Safety fallback

When |R_predicted - R_measured| > `divergence_threshold`, the MPC
**discards** its prediction and falls back to reactive control. This
prevents acting on a forward model that has lost accuracy.

### Computational advantage

The OA reduction is O(1) per step (single complex ODE) versus O(N)
for the full Kuramoto model. For N=1000 oscillators with horizon=10,
MPC prediction costs ~10 ODE steps versus 10000 Euler steps.

::: scpn_phase_orchestrator.supervisor.predictive

---

## FEP Predictive Supervisor

`FEPPredictiveSupervisor` is the first Python supervisor mode that uses the
existing `VariationalPredictor` as an auditable free-energy signal. It observes
the current phase vector, updates the variational predictor, and emits bounded
`zeta` / `Psi` actions only when free energy, prediction error, or stability
proxy thresholds indicate a pre-emptive correction is needed.

```python
from scpn_phase_orchestrator.supervisor import (
    FEPPredictiveSupervisor,
    assess_fep_hierarchy,
)

fep = FEPPredictiveSupervisor(
    n_oscillators=len(phases),
    dt=0.01,
    target_R=0.8,
    free_energy_threshold=1.0,
)
assessment = fep.assess(phases, omegas)
actions = fep.decide(phases, omegas, upde_state, boundary_state)
audit_payload = assessment.to_audit_record()
```

`FEPPredictionAssessment` records free energy, complexity, mean absolute
prediction error, precision statistics, observed and predicted order
parameters, target `R`, and a scalar surprise proxy. This keeps the FEP path
reviewable in the same audit trail as policy, causal, STL, and topology
decisions.

This slice is intentionally conservative: it is a FEP-Kuramoto correspondence
controller over the existing variational predictor, not a claim of a complete
biological active-inference agent.

`assess_fep_hierarchy()` is the reusable hierarchy primitive. It runs one child
`FEPPredictiveSupervisor` per named child observation, reduces each child's
observed coherence into a parent phase vector, then runs a parent
`FEPPredictiveSupervisor` over the reduced child state. The returned
`FEPHierarchyAssessment` records child assessments, child actions, parent
assessment, parent actions, child `R` values, and parent phase encoding.

```python
hierarchy = assess_fep_hierarchy(
    {
        "generation_area": (generation_phases, generation_omegas),
        "demand_area": (demand_phases, demand_omegas),
    },
    dt=0.01,
    parent_dt=0.1,
)
audit_hierarchy = hierarchy.to_audit_record()
```

Domainpack hierarchy proofs:

- `domainpacks/power_grid/fep_hierarchy_demo.py` runs generation and
  demand/renewable child regions into a parent grid supervisor.
- `domainpacks/cardiac_rhythm/fep_hierarchy_demo.py` runs pacemaker/atrial and
  ventricular/recovery child axes into a parent cardiac supervisor.

---

## Performance summary

| Operation | Budget | Notes |
|-----------|--------|-------|
| `RegimeManager.evaluate()` | < 10 μs | Pure Python comparison |
| `SupervisorPolicy.decide()` | < 50 μs | Rule evaluation + action construction |
| `PetriNet.enabled()` | < 10 μs | Guard evaluation |
| `PredictiveSupervisor.predict()` | < 1 ms | OA mean-field (10 complex ODE steps) |
| `EventBus.post()` | < 5 μs | Synchronous dispatch |

---

## Active Inference Agent

The `ActiveInferenceAgent` provides a **predictive control framework** based on
Karl Friston's Variational Free Energy Principle. It represents the next-generation
of SPO controllers, moving beyond static YAML rules into self-adaptive state-space
models.

### Mathematical Model

The agent maintains a low-dimensional internal state $ and minimizes the
**Variational Free Energy** $ between its prediction $\hat{R}$ and the
observed coherence {obs}$:

936875 F \approx \int q(x) \ln \frac{q(x)}{p(R_{obs}, x)} dx 936875

The controller outputs the optimal forcing strength $\zeta$ and reference phase
$\Psi$ to drive the network toward a target coherence level {target}$
(often set to the **metastability threshold**  \approx 0.6$).

### Features
- **Adaptive Suppression:** Spontaneously discovers anti-phase driving ($\Psi = \psi + \pi$) to break harmful phase-locking.
- **Sub-microsecond Control:** Fully implemented in the `spo-kernel` Rust backend for real-time high-frequency response.
- **Emergent Resilientness:** Naturally handles non-stationary frequency drifts by integrating prediction errors into the internal state.

!!! note "Rust-only module"
    `ActiveInferenceAgent` is implemented in `spo-kernel` (Rust crate `spo-supervisor::active_inference`).
    Python access via `spo_kernel.PyActiveInferenceAgent` when the FFI is installed.
