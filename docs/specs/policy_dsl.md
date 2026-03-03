# Policy DSL

YAML-based rule engine for declarative supervisor behaviour.

## Status

**Implemented** in `supervisor/policy_rules.py` (`PolicyEngine` + `load_policy_rules`).
The hardcoded `SupervisorPolicy` provides default regime-driven actions;
`PolicyEngine` extends it with domainpack-specific YAML rules evaluated
per step.  12 of 17 domainpacks ship with `policy.yaml` files.

## Syntax

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

  - name: restore_coordination
    regime: [DEGRADED, RECOVERY]
    condition:
      metric: R_good
      layer: 3
      op: "<"
      threshold: 0.3
    action:
      knob: K
      scope: global
      value: 0.1
      ttl_s: 10.0
```

## Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Rule identifier |
| `regime` | list[str] | Active only in listed regimes (`NOMINAL`, `DEGRADED`, `CRITICAL`, `RECOVERY`) |
| `condition.metric` | str | `R_good`, `R_bad`, or `R` (raw layer order parameter) |
| `condition.layer` | int | Index into the good/bad layer list (for `R_good`/`R_bad`) or absolute layer index (for `R`) |
| `condition.op` | str | `">"`, `"<"`, `">="`, `"<="`, `"=="` |
| `condition.threshold` | float | Comparison value |
| `action.knob` | str | Control knob: `K`, `alpha`, `zeta`, or `Psi` |
| `action.scope` | str | `global` or `layer_N` |
| `action.value` | float | Adjustment magnitude |
| `action.ttl_s` | float | Time-to-live in seconds |

## Evaluation

Rules are evaluated in order.  All matching rules fire (not first-match-only).
Each produces a `ControlAction` that the domainpack's main loop applies to
coupling, lag, drive, or target phase.

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
