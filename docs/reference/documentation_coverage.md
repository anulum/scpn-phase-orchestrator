<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Documentation Coverage Matrix -->

# Documentation Coverage Matrix

This page records the public documentation surface for the repository.
It is intentionally practical: users should be able to install, learn,
run notebooks, run demos, inspect APIs, and move to production without
searching the source tree.

This is the quality gate document for documentation discoverability. If a user can
operate the product, onboard a new domainpack, and audit decisions without
consulting source code, then the public documentation surface is complete for that
user path.

The counts and mappings below should therefore be treated as a release input, not
as static inventory. When implementation coverage expands (new adapters, modules,
frontier surfaces), the matrix must be updated before release and before any
roadmap milestone is announced.

## Owner model

- **Maintainers**: keep this page aligned with `mkdocs.yml` and API source exports.
- **Release owners**: verify that any feature or benchmark launch includes matching
  documentation entries here.
- **Contributors**: when touching a public surface, add/update the corresponding
  documentation row in the same change to avoid “code-first” drift.

## How to use this as a release gate

This matrix is a production-readiness artifact, not a documentation convenience
index. It is used to answer three operational questions before a milestone:

- Can a user reproduce the same path from install to first successful run without
  reading source?
- Is every advertised runtime path represented in API pages, guides, and walkthroughs?
- Do docs indicate where implementation evidence stops and claim scope begins?

Before release, the page is validated by:

1. `mkdocs build --strict --clean` (navigation completeness),
2. the API navigation coverage test (`tests/test_api_docs_navigation.py`),
3. roadmap and release note linkage for each externally announced capability.

If any major path lacks onboarding/tutorial coverage, that path is treated as
`documentation incomplete` even if code exists and tests pass.

## Current Inventory

| Surface | Current coverage | Entry point |
|---------|------------------|-------------|
| Onboarding | Role route, first-hour checklist, repository map | [Onboarding Handbook](../getting-started/onboarding.md) |
| Installation | Package extras, local development install, quick checks | [Installation](../getting-started/installation.md) |
| Quickstart | Minimal Python and CLI run path | [Quickstart](../getting-started/quickstart.md) |
| Tutorials | Seven task-oriented tutorials | [New Domain Checklist](../tutorials/01_new_domain_checklist.md) |
| Guides | Runtime, production, testing, adapters, backend, notebooks | [Guides](../guide/production.md) |
| Contributor onboarding | setup, repository map, checks, documentation duties | [Contributor Onboarding](../guide/contributor_onboarding.md) |
| API reference | 67 MkDocs API pages, all wired into navigation | [API Reference](api/index.md) |
| Domainpacks | 36 domainpacks documented in the gallery | [Domainpack Gallery](../galleries/domainpack_gallery.md) |
| Notebooks | 19 notebook workflows | [Notebooks & Demos](../galleries/notebooks_and_demos.md) |
| Notebook CI matrix | per-notebook extras and execution expectations | [Notebook Execution Matrix](../galleries/notebook_execution_matrix.md) |
| Examples | 28 terminal-first Python examples | [Notebooks & Demos](../galleries/notebooks_and_demos.md) |
| Interactive demos | Streamlit tools, browser WASM demo, CLI demo | [Interactive Tools](../guide/interactive_tools.md) |
| Validation | V&V report, study protocol, testing guide | [V&V Report](../VALIDATION_REPORT.md) |
| Roadmap | public stable, active, deferred, and research tracks | [Public Roadmap](../roadmap.md) |

Counts above are derived from the repository tree at the time this page
was updated: `36` domainpack directories, `28` `examples/*.py` scripts,
`19` notebooks, and `67` files under `docs/reference/api/`.
`tests/test_api_docs_navigation.py` guards that every API reference page is
listed in `mkdocs.yml` and that every maintained public source module has a
mkdocstrings directive in the API reference. Generated protobuf stubs are
covered through the gRPC facade rather than direct autodoc.

The current source inventory contains `204` maintained public Python modules
under `src/scpn_phase_orchestrator/`. API-reference coverage is complete for
those modules through package-level or detailed mkdocstrings pages, excluding
generated protobuf stubs. Module-level docstrings are still being tightened:
the root package now documents the frozen top-level import surface, and new or
touched public modules should carry a module docstring that states the stable
contract, backend assumptions, and failure-mode policy.

## API Documentation Policy

Public APIs are documented at one of two levels:

- **Package-level pages** document the public module surface through
  `mkdocstrings` and a short pipeline explanation.
- **Detailed pages** document numerically or operationally important
  modules with theory, usage examples, backend notes, benchmarks, and
  validation commands.

Private backend probes and experimental auxiliary-backend shim modules
are not listed as user-facing APIs unless they expose a stable public
contract. They are covered through the owning package page and backend
strategy guides.

Module docstrings are required for new public modules. Existing public modules
without module-level docstrings are treated as documentation debt, not as a
reason to weaken the mkdocstrings coverage gate. When touching such a module,
add the module docstring in the same change instead of spreading generic
one-line docstrings across unrelated code.

## v0.6.0 Code-to-Documentation Reconciliation

The v0.6.0 release preparation reconciled the changed source tree against the
public documentation surface using `git diff --name-only v0.5.11..HEAD`.

| Reconciliation item | Result |
| --- | ---: |
| Changed Python modules under `src/scpn_phase_orchestrator/` | 396 |
| Changed public modules matched by reference, guide, tutorial, example, README, or changelog text | 232 |
| Changed modules intentionally excluded as package initialisers, generated gRPC/protobuf stubs, or experimental accelerator implementation mirrors | 125 |
| Residual unmatched changed modules after exclusions | 39 |

The `39` residual modules are private auxiliary backend shims:

- coupling shims: `_hodge_go`, `_hodge_julia`, `_hodge_mojo`,
  `_spectral_go`, `_spectral_julia`, `_spectral_mojo`;
- monitor shims: `_psychedelic_go`, `_psychedelic_julia`,
  `_psychedelic_mojo`;
- UPDE shims: `_basin_stability_*`, `_envelope_*`, `_geometric_*`,
  `_hypergraph_*`, `_inertial_*`, `_market_*`, `_reduction_*`,
  `_simplicial_*`, `_splitting_*`, and `_swarmalator_*` for Go, Julia, and
  Mojo.

These files are not standalone public APIs. They are implementation mirrors
for the documented public dispatchers and are covered by:

- [Backend Fallback Chain](../guide/backend_fallbacks.md), which defines the
  fallback and demotion policy;
- [Backend Strategy](../guide/backend_strategy.md), which defines support
  tiers and promotion criteria;
- the owning API pages for coupling, monitor, and UPDE functions.

No release-blocking public documentation gaps remain from this reconciliation.

## API Coverage Summary

| Package or surface | Public page | Detailed pages |
|--------------------|-------------|----------------|
| Core package, exceptions, compatibility | [Core](api/core.md) | CLI reference |
| Binding specs and resolution | [Binding](api/binding.md) | schema, defaults, tutorials |
| UPDE engines | [UPDE](api/upde.md) | basin stability, bifurcation, delay, engine, envelope, geometric, hypergraph, inertial, market, order parameters, PAC, reduction, simplicial, splitting, stochastic, Stuart-Landau, swarmalator |
| Coupling | [Coupling](api/coupling.md) | attention residuals, connectome, prior, transfer entropy adaptive |
| Monitors | [Monitor](api/monitor.md) | chimera, dimension, entropy production, EVS, ITPC, Lyapunov, NPE, recurrence, sleep staging, transfer entropy, winding |
| Supervisor | [Supervisor](api/supervisor.md) | policy DSL, regime manager, boundary contract |
| Adapters | [Adapters](api/adapters.md) | production, observability, hardware deployment |
| SSGF | [SSGF](api/ssgf.md) | carrier, ethical cost |
| Autotune | [Autotune](api/autotune.md) | SINDy, coupling estimation, phase extraction, frequency ID |
| Neural network layers | [nn API](api/nn.md) | full nn reference and physics validation plan |
| Visualisation and reporting | [Visualisation](api/visualization.md), [Reporting](api/reporting.md) | interactive tools |
| QueueWaves application | [QueueWaves](api/queuewaves.md) | QueueWaves guide and production guide |

## Notebook And Demo Policy

Every notebook or demo should state:

- the domainpack or API surface it exercises;
- the command or UI path to run it;
- whether it is CI-executed or local-only;
- optional dependencies, if any;
- a nearby production path when applicable.

The current notebook/demo inventory is listed in
[Notebooks & Demos](../galleries/notebooks_and_demos.md).

## Maintenance Checklist

When adding a module, domainpack, notebook, or demo:

- Add or update the relevant API reference page.
- Add or update the module docstring for public modules.
- Add a guide, tutorial, notebook, or example that shows the user path.
- Add the page to `mkdocs.yml`.
- Update this matrix if counts or coverage categories change.
- Run `mkdocs build --strict --clean`.

## Reader-path coverage map

Different teams read the repository for different outcomes. The table below maps the
same surface to the most common operational intent.

| Reader goal | Primary source path | Supporting evidence pages |
|-------------|---------------------|-------------------------|
| first-run success | `getting-started/quickstart.md` | `getting-started/installation.md`, `getting-started/hello_world.md` |
| evidence-first adoption | `getting-started/executive_overview.md` | `VALIDATION_REPORT.md`, `guide/production.md`, `guide/notebook_to_production.md` |
| API integration | `reference/api/index.md` | `guide/studio.md`, `reference/api/core.md`, `reference/api/queuewaves.md` |
| operations and runtime rollout | `guide/production.md` | `guide/queuewaves.md`, `guide/hardware_deployment.md`, `guide/rust_ffi.md` |
| formal or safety review | `reference/api/upde_pha_c_formal_obligation.md` | `formal/kinematic_safety.md`, `guide/backend_review_gate.md`, `RELEASE_HYGIENE.md` |

This map is used to avoid documentation dead zones where one role has all required
facts and another role receives only implementation hints.

## Documentation debt policy for new work

For any merge request that ships a new capability in production scope:

- add or update at least one getting-started page for first-time operators,
- ensure the API reference links to the touched module,
- add or update a test or benchmark reference if performance, safety, or numerics change,
- update this matrix before merge.

Missing one of those checkpoints blocks release-signoff unless the change is
explicitly tagged as review-only.
