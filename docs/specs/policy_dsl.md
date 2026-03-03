# Policy DSL

YAML-based rule engine for declarative supervisor behaviour.

## Status

**Implemented** in `supervisor/policy_rules.py` (`PolicyEngine` + `load_policy_rules`).
The hardcoded `SupervisorPolicy` provides default regime-driven actions;
`PolicyEngine` extends it with domainpack-specific YAML rules evaluated
per step.  12 of 17 domainpacks ship with `policy.yaml` files.

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
| `condition.metric` | str | `R`, `R_good`, `R_bad`, or `stability_proxy` |
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

## References

Implementation: `src/scpn_phase_orchestrator/supervisor/policy_rules.py`.
The hardcoded default policy: `src/scpn_phase_orchestrator/supervisor/policy.py`.
Threshold values used in example rules are empirical — see [ASSUMPTIONS.md](../ASSUMPTIONS.md).
