<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Install profiles and preflight commands -->

# Install Profiles

This guide defines supported install profiles and a preflight command for each.
Use it to confirm toolchains, optional dependencies, and expected fallback
behaviour before production or benchmark runs.

## Profile Matrix

| Profile | Install command | Preflight command | Expected fallback |
| --- | --- | --- | --- |
| Python-only | `pip install scpn-phase-orchestrator` | `spo --help` | Supported numerical consumers can use NumPy. |
| Rust FFI, source checkout | `uv sync --locked --extra rust` | `.venv/bin/python tools/install_spo_kernel.py --check-only --json` | The check requires actual native phase and amplitude results. |
| Simulation server, source checkout | `uv sync --locked --extra server --extra rust` | `.venv/bin/spo serve domainpacks/minimal_domain/binding_spec.yaml` | Startup refuses missing or unusable native computation. |
| JAX | `pip install scpn-phase-orchestrator[nn]` or `pip install scpn-phase-orchestrator[jax]` | `python -c "from scpn_phase_orchestrator.nn import HAS_JAX; print(HAS_JAX)"` | JAX-specific modules require JAX; they do not silently switch semantics. |
| QueueWaves | `pip install scpn-phase-orchestrator[queuewaves]` | `python -c "import httpx; print('httpx OK')"` | Collector/server components requiring `httpx` stay unavailable if missing. |
| Full extras | `pip install scpn-phase-orchestrator[full]` | `python -m pytest -q tests/test_backend_module_imports.py` | Missing optional toolchains demote affected backends to fallback chain. |

## PyPI availability boundary

The base package and most extras install from public PyPI. The source checkout
resolves `spo-kernel` from `spo-kernel/crates/spo-ffi` through maturin. If an
index does not supply the required kernel release, use that source-backed path.
Separate product extras retain their own distribution requirements:

| Extra | Requires | Availability |
| --- | --- | --- |
| `rust` | `spo-kernel>=0.5.11` | `uv sync --locked --extra rust` builds the in-repo Rust workspace into this project's `.venv`. An already prepared environment can use `make bridge PYTHON=.venv/bin/python`. |
| `fusion` | `scpn-fusion-core` | Not on public PyPI. A separate product, obtained from its own distribution channel. |
| `scpn-all` | the two above, plus `scpn-quantum-control` and `scpn-control` | Kernel installation does not establish the availability or namespace compatibility of the other products. |

Every other extra (`server`, `quantum`, `plasma`, `nn`, `mpc`, `eeg`, `cardiac`,
`queuewaves`, `studio`, `otel`, `opcua`, `mqtt`, `pqc`, `plot`, `julia`,
`notebook`, `full`) installs from public PyPI.

The Rust acceleration is a **performance** extra, not a correctness dependency:
when `spo_kernel` is absent the runtime uses the pure-Python path with the same
public API, so a bare `pip install scpn-phase-orchestrator` runs the full
analysis → review → audit pipeline end to end. `spo doctor` reports the Rust
backend as an honest `[warn]` in that case, not a failure.

## Preflight Notes

### 1. Python-only

- Confirms base CLI imports and package integrity.
- Suitable for baseline simulation and documentation workflows.

### 2. Rust FFI

- The verification executes public phase and amplitude integration against
  analytical solutions and reports the loaded native library's version and digest.
- A required-native server refuses startup if those checks or native dispatch
  fail. For an intentionally Python-only server, install `[server]` and pass
  `--allow-python` explicitly; this permits fallback rather than forcing it.

### Local server lifecycle

On the host, the kernel is installed in the SPO project environment. It is an
in-process library: the consuming process loads it and releases it on exit.
`spo serve` runs in the foreground on `127.0.0.1:8000` by default; Ctrl-C stops
it. Container packaging carries its own native library in the image.
See [Runtime API](../reference/api/runtime.md) and
[Production Deployment](production.md) for startup and shutdown commands.

### 3. JAX

- `HAS_JAX` must be `True` for differentiable `nn/` workflows.
- If `False`, install JAX-compatible wheels for your platform and Python ABI.

### 4. QueueWaves

- `httpx` availability is required for Prometheus collector runtime.
- Missing dependency affects QueueWaves service components only.

### 5. Full extras

- Use backend import tests as a broad capability smoke check.
- Auxiliary backends (Go/Julia/Mojo) remain experimental and may be absent
  without blocking primary Rust/Python/JAX workflows.

## Recommended Verification Order

1. Confirm profile install command completed without resolver conflicts.
2. Run the profile preflight command.
3. Run focused tests for your active subsystem.
4. For release candidates, run project CI-equivalent checks.

## Profile selection overview

The profile matrix is designed to align infrastructure constraints with deployment
risk posture:

- `Python-only` keeps the operational surface minimal for analysis, docs, and
  benchmark reproduction.
- `Rust FFI` buys runtime throughput on supported hosts and is appropriate when
  deterministic parity evidence already covers the pure-Python path.
- `JAX` is selected when differentiable training or gradient-based
  investigations are part of normal operation.
- `QueueWaves` adds the telemetry and alerting surface needed for live
  microservice resilience use-cases.
- `Full extras` is the least ambiguous option for release prep because it
  exercises the documented capability surface in one run.

### Preflight as release control

The profile preflight should be viewed as a contract check, not a convenience.
When a team moves from lab to production, these checks give a reproducible
evidence artifact that says, *which capabilities are active and which are
explicitly unavailable* under that deployment profile.

## Related

- [Backend Fallback Chain](backend_fallbacks.md)
- [Multi-Language Backend Review Gate](backend_review_gate.md)
- [Dependency Locks](dependency_locks.md)

## Why this section is kept in release preparation

Profiles are the first operational control point for reproducibility. Each row
in this table gives both what is enabled and what is not, which is essential when
incident reports require capability evidence.

Recommended use:

- capture profile result in release notes,
- keep the preflight command output with benchmark inputs,
- and keep dependency overrides explicit when environment constraints prevent a
  profile from being fully enabled.
