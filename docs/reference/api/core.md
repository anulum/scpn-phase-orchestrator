# Core & Exceptions

Foundation types shared across all SPO subsystems.

## Exception Hierarchy

SPO defines a hierarchy of domain-specific exceptions rooted at
`SPOError`. Each subsystem raises its own subclass so that callers
can catch errors at the appropriate granularity:

```
SPOError
├── BindingError          # Invalid binding specification or missing fields
├── ValidationError       # Schema or constraint violation
├── ExtractorError        # Phase extraction failure (bad signal, wrong sample rate)
├── EngineError           # UPDE integration failure (NaN, divergence, dt violation)
├── PolicyError           # Policy rule evaluation failure
└── AuditError            # Audit chain integrity violation
```

All exceptions carry a descriptive message and preserve the original
traceback when wrapping lower-level errors.

### Catching by Granularity

```python
from scpn_phase_orchestrator.exceptions import SPOError, EngineError

try:
    engine.step(phases, omegas, knm, zeta, psi, alpha)
except EngineError as e:
    # Handle integration failure specifically
    logger.warning(f"Engine diverged: {e}")
    supervisor.force_transition(Regime.CRITICAL)
except SPOError as e:
    # Catch-all for any SPO error
    logger.error(f"SPO error: {e}")
```

::: scpn_phase_orchestrator.exceptions

## Compatibility

Internal compatibility module providing shared constants and
conditional imports for optional dependencies:

- `TWO_PI` — $2\pi$ as `float64` (avoids repeated computation)
- Conditional imports for `jax`, `equinox`, `redis`, `opentelemetry`
  that fall back gracefully when the dependency is not installed

::: scpn_phase_orchestrator._compat
