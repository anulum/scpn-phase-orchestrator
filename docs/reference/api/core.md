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

## Optional dependency detection

`_compat.py` exports two values:

- `TWO_PI` — 2π as float64
- `HAS_RUST` — `True` when `spo_kernel` Rust extension is importable

Individual modules handle their own optional imports locally:

| Module | Guard | Package |
|--------|-------|---------|
| `nn/` | `pytest.importorskip("jax")` | jax, equinox, optax |
| `reporting/plots.py` | `_HAS_MPL` | matplotlib |
| `ssgf/tcbo.py` | `_HAS_RIPSER` | ripser |
| `upde/engine.py` | `_HAS_RUST` (from _compat) | spo_kernel |
| `coupling/knm.py` | `_HAS_RUST` (from _compat) | spo_kernel |
| `oscillators/physical.py` | `_HAS_RUST` (from _compat) | spo_kernel |

When an optional dependency is missing, the corresponding subsystem
either skips the optimised path (Rust → Python fallback) or raises
`ImportError` with install instructions.

## Error handling philosophy

SPO follows a fail-fast strategy at system boundaries:

- **Input validation:** `ValueError` for invalid shapes, NaN, etc.
- **Engine divergence:** `EngineError` with step number and divergence magnitude
- **Binding errors:** `BindingError` with field path and expected type
- **Audit tampering:** `AuditError` with chain break location

Internal code trusts validated data — no redundant checks on hot paths.
This keeps engine step latency under 1 ms for N ≤ 64.
