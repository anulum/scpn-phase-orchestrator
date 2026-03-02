# Boundary Contract

## Boundary Types

| Severity | Meaning | Consequence |
|----------|---------|-------------|
| `soft` | Warning. System operating outside preferred range. | Logged. May trigger DEGRADED regime. |
| `hard` | Violation. System in unsafe state. | Forces CRITICAL regime immediately. |

## BoundaryDef

Defined in the binding spec `boundaries` list:

```yaml
boundaries:
  - name: queue_depth_limit
    variable: queue_depth
    lower: null
    upper: 10000
    severity: hard
  - name: latency_warning
    variable: p99_latency_ms
    lower: null
    upper: 500
    severity: soft
```

Fields:

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Human-readable identifier |
| `variable` | str | Key in the values dict passed to `observe()` |
| `lower` | float or null | Lower bound. null = no lower check. |
| `upper` | float or null | Upper bound. null = no upper check. |
| `severity` | str | `"soft"` or `"hard"` |

## BoundaryObserver

`BoundaryObserver(boundary_defs)` checks a dict of measured values against all definitions.

```python
state = observer.observe({"queue_depth": 12000, "p99_latency_ms": 300})
# state.hard_violations = ["queue_depth_limit: queue_depth=1.2e+04 outside [None, 10000]"]
# state.soft_violations = []
```

## BoundaryState

| Field | Type | Content |
|-------|------|---------|
| `violations` | list[str] | All violated boundaries (soft + hard) |
| `soft_violations` | list[str] | Soft-severity violations only |
| `hard_violations` | list[str] | Hard-severity violations only |

## Integration with RegimeManager

`RegimeManager.evaluate()` checks `boundary_state.hard_violations` first. Any hard violation forces CRITICAL regardless of R values.

Soft warnings do not directly trigger regime changes but are logged in the audit trace for post-hoc analysis.

## References

Regime escalation logic is defined in [regime_manager.md](regime_manager.md). Boundary severity levels are configured per-domain in the binding spec — see [binding_spec.schema.json](binding_spec.schema.json).
