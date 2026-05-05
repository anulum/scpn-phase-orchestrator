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

## Current Inventory

| Surface | Current coverage | Entry point |
|---------|------------------|-------------|
| Onboarding | Role route, first-hour checklist, repository map | [Onboarding Handbook](../getting-started/onboarding.md) |
| Installation | Package extras, local development install, quick checks | [Installation](../getting-started/installation.md) |
| Quickstart | Minimal Python and CLI run path | [Quickstart](../getting-started/quickstart.md) |
| Tutorials | Six task-oriented tutorials | [New Domain Checklist](../tutorials/01_new_domain_checklist.md) |
| Guides | Runtime, production, testing, adapters, backend, notebooks | [Guides](../guide/production.md) |
| API reference | 58 MkDocs API pages | [API Reference](api/index.md) |
| Domainpacks | 36 domainpacks documented in the gallery | [Domainpack Gallery](../galleries/domainpack_gallery.md) |
| Notebooks | 19 notebook workflows | [Notebooks & Demos](../galleries/notebooks_and_demos.md) |
| Examples | 27 terminal-first Python examples | [Notebooks & Demos](../galleries/notebooks_and_demos.md) |
| Interactive demos | Streamlit tools, browser WASM demo, CLI demo | [Interactive Tools](../guide/interactive_tools.md) |
| Validation | V&V report, study protocol, testing guide | [V&V Report](../VALIDATION_REPORT.md) |

Counts above are derived from the repository tree at the time this page
was updated: `36` domainpack directories, `27` `examples/*.py` scripts,
`19` notebooks, and `58` files under `docs/reference/api/`.

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
- Add a guide, tutorial, notebook, or example that shows the user path.
- Add the page to `mkdocs.yml`.
- Update this matrix if counts or coverage categories change.
- Run `mkdocs build --strict --clean`.
