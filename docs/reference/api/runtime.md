<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Simulation server and native kernel API -->

# Runtime server and kernel

The simulation server keeps one binding and its mutable simulation state in
one process. `POST /api/step` integrates one configured timestep; reset restores
the binding's initial state. Observing state, health or the WebSocket stream
does not advance the simulation.

## Foreground startup

From a source checkout:

```bash
uv sync --locked --extra server --extra rust
.venv/bin/python tools/install_spo_kernel.py --check-only --json
.venv/bin/spo serve domainpacks/minimal_domain/binding_spec.yaml
```

The default address is `127.0.0.1:8000`. Ctrl-C stops the process. The command
starts no persistent service and configures no automatic restart. On Windows,
use the corresponding executables under `.venv/Scripts/`.

The default `--require-kernel` admission first requires the selected binding's
simulation engine to dispatch to Rust. A NumPy selection raises `RuntimeError`
before the listener opens. Admission then executes actual native phase and
amplitude integration with analytical solutions. A missing or unusable kernel
refuses startup.

Install or update the kernel before starting the consumer process. A process
that loaded SPO without the kernel can retain its earlier NumPy selection;
required-native verification refuses it until the process is restarted. The
verification report also requires an identifiable native library file so its
hash can be recorded; an embedded module without that file is refused clearly.

An intentionally Python-only environment can install `[server]` and pass
`--allow-python`. This allows NumPy fallback; an installed usable kernel can
still be selected. The base Python package retains its documented numerical
fallbacks.

## HTTP contracts

| Endpoint | Behaviour |
| --- | --- |
| `GET /` | Simulation dashboard. |
| `GET /api/state` | Current step, layer/global coherence, regime and amplitude summary. |
| `POST /api/step` | Advance one timestep and return the resulting snapshot. |
| `POST /api/reset` | Restore the initial simulation state. |
| `GET /api/config` | Binding name/dimensions/periods, amplitude mode, selected backend and kernel admission. |
| `GET /api/health` | Check simulation snapshot/coherence/regime without advancing it. |
| `GET /api/metrics` | Current Prometheus metrics. |
| `GET /api/studio-feed` | Current STUDIO feed envelope. |
| `WS /ws/stream` | Read-only snapshot stream. |

Configuration includes `backend` (`"rust"` or `"numpy"`), `kernel_required`,
and `kernel`. Required-native admission fills `kernel` with the verified
distribution `version` and native library `sha256`; optional admission leaves
that verification report `null`.

With `SPO_ENV=production`, `SPO_API_KEY` is mandatory. Mutable endpoints require
the matching `X-API-Key` header; production rate limits apply. Use one worker
for the shared in-process simulation. See
[Production Deployment](../../guide/production.md).

## Public Python API

`create_app(spec_path, require_kernel=False)` returns a FastAPI application.
Direct Python callers explicitly select `require_kernel=True` when native
computation is part of their deployment contract. The CLI selects it by default.

`verify_kernel()` runs public `UPDEEngine.step` and
`StuartLandauEngine.step` against uncoupled analytical trajectories and returns
`KernelVerification(version, extension, sha256)`. It creates its own engines
and does not advance an existing simulation. The report binds successful
numerical verification to the loaded native library.

::: scpn_phase_orchestrator.runtime.kernel

::: scpn_phase_orchestrator.runtime.server.create_app
