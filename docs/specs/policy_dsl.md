# Policy DSL

YAML-based rule engine for declarative supervisor behaviour.

## Status

**Implemented** in `supervisor/policy_rules.py` (`PolicyEngine` + `load_policy_rules`).
The hardcoded `SupervisorPolicy` provides default regime-driven actions;
`PolicyEngine` extends it with domainpack-specific YAML rules evaluated
per step.  All 33 domainpacks ship with `policy.yaml` files.

## Syntax (v0.1 — single condition, single action)

```yaml
rules:
  - name: suppress_retry_storm
    regime: [NOMINAL, DEGRADED]
    condition:
      metric: R_bad
      layer: 1
      op: ">"
      threshold: 0.7
    action:
      knob: alpha
      scope: layer_1
      value: 0.3
      ttl_s: 5.0
```

## Syntax (v0.2 — compound conditions, action chains)

```yaml
rules:
  - name: emergency_boost
    regime: [DEGRADED, CRITICAL]
    logic: AND
    conditions:
      - metric: R
        layer: 0
        op: "<"
        threshold: 0.3
      - metric: stability_proxy
        op: "<"
        threshold: 0.5
    actions:
      - knob: K
        scope: global
        value: 0.5
        ttl_s: 5.0
      - knob: alpha
        scope: layer_0
        value: 0.1
        ttl_s: 3.0
    cooldown_s: 10.0
    max_fires: 5

  - name: either_layer_low
    regime: [NOMINAL]
    logic: OR
    conditions:
      - metric: R
        layer: 0
        op: "<"
        threshold: 0.3
      - metric: R
        layer: 1
        op: "<"
        threshold: 0.3
    action:
      knob: K
      scope: global
      value: 0.2
      ttl_s: 5.0
```

Both v0.1 and v0.2 syntax are supported in the same file.  The parser
auto-detects:

- `condition` (singular) → single `PolicyCondition`
- `conditions` (plural) + `logic` → `CompoundCondition`
- `action` (singular) → single-element action list
- `actions` (plural) → multi-action chain

## Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Rule identifier |
| `regime` | list[str] | Active only in listed regimes (`NOMINAL`, `DEGRADED`, `CRITICAL`, `RECOVERY`) |
| `condition.metric` | str | `R`, `R_good`, `R_bad`, `stability_proxy`, `pac_max`, `mean_amplitude`, `subcritical_fraction`, `amplitude_spread`, or `mean_amplitude_layer` |
| `condition.layer` | int \| null | Layer index (null for global metrics like `stability_proxy`) |
| `condition.op` | str | `">"`, `"<"`, `">="`, `"<="`, `"=="` |
| `condition.threshold` | float | Comparison value |
| `logic` | str | `"AND"` (default) or `"OR"` — combinator for `conditions` list |
| `action.knob` | str | Control knob: `K`, `alpha`, `zeta`, or `Psi` |
| `action.scope` | str | `global` or `layer_N` |
| `action.value` | float | Adjustment magnitude |
| `action.ttl_s` | float | Time-to-live in seconds |
| `cooldown_s` | float | Minimum seconds between consecutive firings of this rule (0 = no cooldown) |
| `max_fires` | int | Maximum total firings of this rule (0 = unlimited) |

## Evaluation

Rules are evaluated in order.  All matching rules fire (not first-match-only).

For compound conditions:
- **AND**: all sub-conditions must pass
- **OR**: at least one sub-condition must pass

Each matching rule produces one `ControlAction` per entry in its `actions`
list. The `justification` field of each action references the rule name.

Rate-limiting applies per rule:
- `cooldown_s` prevents re-firing within the specified interval
- `max_fires` caps the total number of times the rule can fire

## Regime Scoping

A rule only fires if the current regime (from `RegimeManager`) is in its
`regime` list.  This prevents conflicting actions across regimes.

## Integration

The binding spec references a policy file:

```yaml
policy: policy.yaml
```

Resolved relative to the binding spec directory.  In `run.py`:

```python
from scpn_phase_orchestrator.supervisor.policy_rules import PolicyEngine, load_policy_rules

rules = load_policy_rules(policy_path)
policy_engine = PolicyEngine(rules)
actions = policy_engine.evaluate(regime, upde_state, good_layers, bad_layers)
```

## Protocol Net (v0.3)

The binding spec supports an optional `protocol_net:` key for Petri net
regime sequencing.  When present, `SupervisorPolicy.decide()` delegates
regime evaluation to `PetriNetAdapter` instead of `RegimeManager.evaluate()`.

```yaml
protocol_net:
  places: [warmup, nominal, cooldown, done]
  initial: {warmup: 1}
  place_regime: {warmup: NOMINAL, nominal: NOMINAL, cooldown: RECOVERY, done: NOMINAL}
  transitions:
    - name: start
      inputs: [{place: warmup, weight: 1}]
      outputs: [{place: nominal, weight: 1}]
      guard: "stability_proxy > 0.6"
    - name: wind_down
      inputs: [{place: nominal, weight: 1}]
      outputs: [{place: cooldown, weight: 1}]
      guard: "R_0 < 0.3"
    - name: finish
      inputs: [{place: cooldown, weight: 1}]
      outputs: [{place: done, weight: 1}]
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `places` | list[str] | Place names in the net |
| `initial` | dict[str, int] | Initial token marking |
| `place_regime` | dict[str, str] | Maps each place to a Regime |
| `transitions[].name` | str | Transition identifier |
| `transitions[].inputs` | list[{place, weight}] | Input arcs |
| `transitions[].outputs` | list[{place, weight}] | Output arcs |
| `transitions[].guard` | str \| null | Guard expression: `"metric op threshold"` |

Guard syntax uses the same operators as policy conditions: `>`, `>=`, `<`,
`<=`, `==`.  Metrics are the same flat dict keys passed to policy rules
(`R`, `R_0`..`R_n`, `stability_proxy`).

When multiple places are marked, the highest-severity regime wins
(CRITICAL > RECOVERY > DEGRADED > NOMINAL).

## References

Implementation: `src/scpn_phase_orchestrator/supervisor/policy_rules.py`.
The hardcoded default policy: `src/scpn_phase_orchestrator/supervisor/policy.py`.
Petri net: `src/scpn_phase_orchestrator/supervisor/petri_net.py`.
Petri adapter: `src/scpn_phase_orchestrator/supervisor/petri_adapter.py`.
Threshold values used in example rules are empirical — see [ASSUMPTIONS.md](../ASSUMPTIONS.md).
