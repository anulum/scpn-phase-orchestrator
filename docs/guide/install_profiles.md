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
| Python-only | `pip install scpn-phase-orchestrator` | `python -m scpn_phase_orchestrator.cli --help` | Rust/JAX unavailable paths fall back to Python where implemented. |
| Rust FFI | `pip install scpn-phase-orchestrator` + built `spo_kernel` | `python -c "import spo_kernel; print('spo_kernel OK')"` | If `spo_kernel` missing, modules use Python fallback (same public API). |
| JAX | `pip install scpn-phase-orchestrator[nn]` | `python -c "from scpn_phase_orchestrator.nn import HAS_JAX; print(HAS_JAX)"` | JAX-specific modules require JAX; they do not silently switch semantics. |
| QueueWaves | `pip install scpn-phase-orchestrator[queuewaves]` | `python -c "import httpx; print('httpx OK')"` | Collector/server components requiring `httpx` stay unavailable if missing. |
| Full extras | `pip install scpn-phase-orchestrator[full]` | `python -m pytest -q tests/test_backend_module_imports.py` | Missing optional toolchains demote affected backends to fallback chain. |

## Preflight Notes

### 1. Python-only

- Confirms base CLI imports and package integrity.
- Suitable for baseline simulation and documentation workflows.

### 2. Rust FFI

- `spo_kernel` import success confirms accelerated path is usable.
- If import fails, runtime remains functional on Python fallback for supported
  modules, at lower performance.

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

## Related

- [Backend Fallback Chain](backend_fallbacks.md)
- [Multi-Language Backend Review Gate](backend_review_gate.md)
- [Dependency Locks](dependency_locks.md)
