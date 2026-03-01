# Regime Manager

## Regimes

| Regime | Description |
|--------|-------------|
| `NOMINAL` | Healthy operation. R_good above threshold, no violations. |
| `DEGRADED` | R_good below target but above critical. Soft warnings possible. |
| `CRITICAL` | Hard boundary violations or R below critical threshold. |
| `RECOVERY` | Transitioning from CRITICAL back toward NOMINAL. |

## Entry Conditions

| Regime | Condition |
|--------|-----------|
| NOMINAL | `avg_R >= 0.6` and no hard violations |
| DEGRADED | `0.3 <= avg_R < 0.6` and no hard violations |
| CRITICAL | `avg_R < 0.3` OR hard boundary violations |
| RECOVERY | Previously CRITICAL, now `avg_R >= 0.3` |

`avg_R` is the mean R across all layers in `UPDEState`.

## Thresholds

See [ASSUMPTIONS.md](../ASSUMPTIONS.md) § Regime Thresholds for provenance.

```python
_R_CRITICAL = 0.3
_R_DEGRADED = 0.6
```

## Hysteresis

`RegimeManager(hysteresis=0.05, cooldown_steps=10)`.

- **hysteresis:** Not currently used for threshold offset; reserved for future refinement.
- **cooldown_steps:** Minimum steps between regime transitions. Prevents oscillation between adjacent regimes.

Exception: escalation to CRITICAL always bypasses cooldown.

## Transition Logic

```
proposed = evaluate(upde_state, boundary_state)
regime   = transition(current, proposed)
```

`transition()` applies cooldown gating:

1. If `proposed == current`, no change.
2. If `steps_since_last_transition < cooldown_steps` AND `proposed != CRITICAL`, reject transition.
3. Otherwise, accept transition and reset counter.

## Supervisor Integration

`SupervisorPolicy.decide()` calls `RegimeManager.evaluate()` and `transition()` internally, then selects control actions based on the resulting regime:

| Regime | Action |
|--------|--------|
| NOMINAL | No-op |
| DEGRADED | Boost global K |
| CRITICAL | Increase zeta, reduce K on worst layer |
| RECOVERY | Gradual K restore |

## References

- **[acebron2005]** J. A. Acebrón et al. (2005). The Kuramoto model: a simple paradigm for synchronization phenomena. *Rev. Mod. Phys.* 77, 137–185. — Incoherence / partial-sync boundaries motivating R thresholds.
- **[dorfler2014]** F. Dörfler & F. Bullo (2014). Synchronization in complex networks of phase oscillators: a survey. *Automatica* 50, 1539–1564. — Finite-N synchronisation conditions and hysteresis.
