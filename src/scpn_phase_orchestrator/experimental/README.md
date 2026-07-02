<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator - Accelerator namespace boundary -->

# Accelerator Namespace Boundary

`scpn_phase_orchestrator.experimental` is a historical package name. It is not a
research sandbox and it is not an invitation to import backend files directly.
The package contains load-bearing polyglot backend implementations used by the
validated dispatchers in `coupling`, `monitor`, and `upde`.

## Usage

Use the owning production API:

- `scpn_phase_orchestrator.coupling.*` for coupling construction, analysis, and
  adaptation;
- `scpn_phase_orchestrator.monitor.*` for observer and metric surfaces;
- `scpn_phase_orchestrator.upde.*` for phase-dynamics integration.

Those modules own validation, backend selection, audit fields, fallback
semantics, and public documentation. Files under `experimental/accelerators/`
are implementation backends for those modules.

## Backend Policy

- Python remains the reference floor.
- Rust, WebGPU, Mojo, Julia, and Go paths are selected by the owning dispatcher
  when the backend exists for that kernel.
- Missing optional toolchains may demote to the next backend.
- Julia paths require a complete `juliacall.Main` runtime before side-file
  loading; partially initialised `juliacall` modules are treated as unavailable.
- Validation failures, non-finite outputs, ABI mismatches, and physics-contract
  violations must not demote silently.
- New backend work needs production-surface tests, parity evidence, docs, and
  benchmark metadata before a dispatcher can rely on it.

The public architecture map for this namespace is
`docs/architecture/subsystems/experimental-accelerators.md`.
