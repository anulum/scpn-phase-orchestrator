# Core & Exceptions

Foundation types shared across all SPO subsystems.

## Purpose for production teams

This page is the reliability seam for the entire runtime. The shared error family and compatibility controls are what allow subsystems to fail with bounded, actionable behaviour instead of cascading into undefined control actions.

When reading this page:

- use the exception section to align operational alerting and fallback policy,
- use the compatibility section to understand optional dependency fallbacks,
- use the observability entry points to verify what metrics and telemetry remain
  available when optional stacks are missing.

The intent is to make runtime limits and optional paths explicit before deployment planning begins.

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

## Simulation core

`runtime.simulation` is the single non-actuating closed/open-loop core that
backs both ``spo run`` and the public ``evaluate_binding_spec`` facade, so a
binding spec is advanced by exactly one implementation. ``policy_enabled`` is the
open/closed-loop switch: with it on, the supervisor and domainpack policy feed
bounded actions back into the dynamics; with it off, the same drivers and
intrinsic plasticity run without control feedback, giving the baseline for
measuring orchestration uplift on a fixed seed.

`simulate()` also accepts an optional `scenario_hook` for deterministic
non-actuating perturbation schedules. The hook receives a
`SimulationScenarioContext` before each integration step and may adjust phases,
natural frequencies, coupling state, `zeta`, or `psi_target`; the core validates
finite vector shapes, scalar types, and `CouplingState` identity immediately
after the hook returns. This is the supported path for benchmark and case-study
scenario evidence, not a hardware or live-actuation path.

::: scpn_phase_orchestrator.runtime.simulation

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

## Environment readiness

`runtime.doctor` aggregates the per-module detection above into one report
behind the `spo doctor` command. It probes the interpreter version against
`requires-python`, the required runtime dependencies, the optional native
backends (Rust/Julia/Go/Mojo), and the optional feature extras, returning a
`DoctorReport` whose status is `pass` only when the interpreter is in range and
every required dependency is importable. See the
[CLI reference](../cli.md#spo-doctor) for the command and exit codes.

::: scpn_phase_orchestrator.runtime.doctor

### Chaos-engineering resilience injection

`runtime.chaos` injects realistic, non-actuating faults — coupling drops,
frequency drift, sensor noise, and drive dropout — into a controlled simulation
and measures how the orchestrator recovers. A `ChaosSchedule` of `ChaosFault`
windows is applied through the simulation's `scenario_hook` boundary, so the
heavy compute stays in the existing multi-language UPDE engine and this module is
the orchestration and scoring layer. `run_resilience_experiment` runs the spec
once nominally and once perturbed under the same seed, then `compute_resilience`
derives recovery time, peak coherence drop, stability-margin erosion, and final
deviation. The `spo chaos` command exposes this from the CLI; all runs are
review-only.

::: scpn_phase_orchestrator.runtime.chaos

### Deterministic (bounded-jitter) execution mode

`runtime.deterministic` runs an arbitrary per-step callable against a fixed
period with the timing guarantees a plain loop cannot give: every step is
scheduled at `t0 + i·period` on the monotonic clock (sleep, with an optional
final spin, then measured jitter), each step is timed against a worst-case
execution-time budget so an overrun is a fatal *deadline miss* by default
(`miss_policy='abort'`). Callers must explicitly choose `miss_policy='observe'`
for diagnostic runs that record misses and continue. The cyclic garbage
collector is frozen and disabled for the hot path so GC pauses leave the jitter
budget.
`run_deterministic_loop` returns an `ExecutionTimingReport` with per-step
latencies and jitters plus aggregate statistics (mean / max / p99 latency, max
absolute jitter, deadline-miss count). The loop is timing-only and
non-actuating: it never inspects or mutates the step's state, so it drives the
simulation step, a controller tick, or any periodic task without coupling to a
specific engine.

::: scpn_phase_orchestrator.runtime.deterministic

### Post-quantum audit-chain seal

`runtime.audit_pqc` adds an additive, post-quantum seal over the audit hash
chain. The audit logger already chains each record with SHA-256 and signs it
with HMAC; HMAC is symmetric and not post-quantum. `seal_audit_log` signs the
chain tip (the `_hash` of the last record — the SHA-256 commitment to the whole
log) with **ML-DSA** (FIPS 204, available natively in `cryptography`), producing
an `AuditChainSeal` that anyone holding the trusted public key can verify, long
after the run and against a future quantum adversary. `verify_audit_log_seal`
re-reads the chain tip and record count and rejects the seal if either changed,
so any post-hoc edit is detected. ML-DSA-65 is the default; ML-DSA-44/87 are
selectable, and the seal records its algorithm so SLH-DSA (FIPS 205) can be
added later without breaking existing seals. The seal is additive — it does not
touch the HMAC flow — so it carries no regression risk.

::: scpn_phase_orchestrator.runtime.audit_pqc

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
