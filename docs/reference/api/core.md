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

## CLI entry point

The public Click command tree lives in `scpn_phase_orchestrator.runtime.cli`. Use the
[CLI Reference](../cli.md) for command-oriented examples; this API section
keeps the callable entry points visible to mkdocstrings.

::: scpn_phase_orchestrator.runtime.cli

## Network security helpers

Shared helpers for production-mode detection, environment integer parsing, and
per-identity fixed-window rate limiting.

::: scpn_phase_orchestrator.runtime.network_security

## Runtime observability

Prometheus text metrics are a default Runtime/Serving surface, not an optional
external adapter. OpenTelemetry remains an optional backend; absent OTel packages
produce validated no-op spans while `/api/metrics` and local Prometheus text
export stay active.

::: scpn_phase_orchestrator.runtime.observability

## Web and gRPC services

The service modules expose the FastAPI dashboard state container and the gRPC
servicer. Optional web or gRPC dependencies are handled at import time so
documentation builds can inspect the public surface without launching servers.

::: scpn_phase_orchestrator.runtime.server

::: scpn_phase_orchestrator.runtime.server_grpc

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

## Role in the production boundary

This module is the project-wide seam between raw exceptions and deterministic
control outcomes. Every subsystem raises typed SPO errors into the same hierarchy,
so operational handlers can route failures consistently across CLI, services, and
embedded drivers.

In practice this gives operators a bounded failure vocabulary:

- **Validation faults** are recoverable through corrected input or fallback
  profiles.
- **Engine faults** can trigger deterministic supervisor actions.
- **Audit faults** stop replay promotion by default to protect chain integrity.

## Why optional dependency handling matters

The documented optional import and `HAS_*` flags are part of reliability policy, not
just packaging convenience. They prevent import-time crashes when one runtime path is
unavailable, while keeping production-facing Python behavior explicit:

- Required behaviour remains available on the baseline path.
- Optional accelerators are additive when present.
- Audit and CLI surfaces can still start on minimal environments.

This pattern prevents “one package drift takes down all controls” incidents by
making capability and observability dependencies explicit in the import graph.

## Enterprise usage note

When teams evaluate SPO for regulated or multi-region deployment, this contract is a
critical documentation checkpoint because it documents which failures are expected to
halt, which failures are reviewed, and which can be auto-recovered without risking
audit continuity.
