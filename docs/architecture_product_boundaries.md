# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - Product boundaries

# Product boundaries

SCPN Phase Orchestrator is organised around four product boundaries. The
boundaries are architectural ownership lines, not an immediate promise that all
legacy import paths have already moved into matching directories.

## Core Engine

Core Engine contains the deterministic mathematical and phase-control substrate:
binding, oscillators, coupling, UPDE engines, monitors, core actuation models,
imprint state, SSGF primitives, and the minimal supervisor types required to run
phase-control decisions.

Core Engine must remain free of serving, tenancy, external system, notebook,
bridge, and experimental runtime dependencies. Runtime, integrations, and
research surfaces may depend on Core Engine; Core Engine must not depend on
those surfaces.

## Runtime and Serving

Runtime and Serving contains process orchestration and operator surfaces: CLI,
HTTP, gRPC, replay, audit streams, state backends, authentication, tenancy,
observability glue, Studio, QueueWaves serving, and generated transport code.

Runtime may depend on Core Engine and integrations, but Runtime owns process
lifetime and request/response behaviour. Core Engine must not import Runtime.

## Integrations

Integrations contains external system adapters: Redis, Modbus, OpenTelemetry,
hardware IO, BCI/LSL, Prometheus, Remanentia, NeuroCore, Fusion Core, Plasma
Control, and related bridge adapters.

Integrations must stay optional where their dependencies are optional. Missing
external dependencies must fail with explicit, local errors instead of leaking
into Core Engine import time. Integrations may depend on Core Engine contracts,
but must not import Runtime/Serving or Research/Experimental modules.

## Research and Experimental

Research and Experimental contains neural-network research modules, notebooks,
experiments, Go/Julia/Mojo/WebGPU accelerator implementations, visualisation
helpers, and special domain packs that are not required for the production
Runtime surface. It also contains autotuning pipelines and legacy public
neural-network compatibility aliases until those surfaces are split behind
explicit optional package extras.

Experimental modules may depend on Core Engine for parity and validation, but
Core Engine must not import arbitrary Experimental modules. Accelerator
implementations live under
`scpn_phase_orchestrator.experimental.accelerators.{coupling,monitor,upde}`.
Core dispatch modules may import only the explicit accelerator-port modules
listed in `tools/check_product_boundaries.py`; legacy module paths under
`coupling`, `monitor`, and `upde` are compatibility wrappers only. The
accelerator-port allowlist is self-auditing: full-tree runs fail if an entry
becomes stale, forcing migrated dispatch ports to be removed from the exception
set instead of leaving dead architecture debt behind.

## Enforcement

`tools/check_product_boundaries.py` is the first enforcement rail. It parses
Python imports and fails when Core Engine imports Runtime, Integrations, or
Research/Experimental modules. It also fails when Integrations import
Runtime/Serving or Research/Experimental modules, preserving the adapter layer
as an optional boundary around external systems.

Every first-party top-level package must be assigned to one of the four
boundaries. The checker fails on unclassified source modules and unclassified
first-party imports, so new surfaces cannot bypass the architecture contract by
landing outside the boundary map.

This guard deliberately starts with the highest-value invariant: Core Engine is
the stable lower layer. Later migration batches should add stricter rules for
Runtime-to-Experimental coupling and optional integration dependency isolation
after those imports are inventoried and compatibility re-exports are in place.
