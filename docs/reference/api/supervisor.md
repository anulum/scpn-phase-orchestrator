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

::: scpn_phase_orchestrator.supervisor.active_inference_agent
