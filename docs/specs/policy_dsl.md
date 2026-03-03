# Policy DSL (Planned v0.2)

YAML-based rule engine for declarative supervisor behaviour.

## Status

Not yet implemented. The current supervisor uses hardcoded `SupervisorPolicy`. This spec describes the planned replacement.

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
| `regime` | list[str] | Active only in listed regimes |
| `condition.metric` | str | `R_good`, `R_bad`, or any boundary variable |
| `condition.layer` | int | Layer index for R metrics |
| `condition.op` | str | `">"`, `"<"`, `">="`, `"<="`, `"=="` |
| `condition.threshold` | float | Comparison value |
| `action` | object | `ControlAction` fields |

## Evaluation

Rules are evaluated in order. First matching rule fires. No chaining within a single step -- at most one rule fires per step per knob.

## Regime Scoping

A rule only fires if the current regime is in its `regime` list. This prevents conflicting actions across regimes.

## Integration

The binding spec references a policy file:

```yaml
policy: policy.yaml
```

Resolved relative to the binding spec directory.

## References

Planned for v0.2 — see [ROADMAP.md](https://github.com/anulum/scpn-phase-orchestrator/blob/main/ROADMAP.md). The current hardcoded policy is documented in [regime_manager.md](regime_manager.md) § Supervisor Integration. Threshold values used in example rules are empirical — see [ASSUMPTIONS.md](../ASSUMPTIONS.md).
