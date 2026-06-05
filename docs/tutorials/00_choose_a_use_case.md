<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Tutorial Use-Case Router -->

# 00 — Choose a Use Case

## Purpose of this index page

This page is the starting point for teams that need a fast route into the
repository. It exists to reduce ambiguity between domain onboarding and execution
path selection.

The table below is a routing layer only: it maps intent to the most relevant
tutorial or reference first, then to supporting pages.

Start here when the repository feels too broad. Pick the row closest to your
problem, then follow the linked tutorial path.

| You have | You want | Start with | Then read |
|----------|----------|------------|-----------|
| A system with repeated measurements | Find useful or harmful synchrony | [Oscillator Hunt Sheet](02_oscillator_hunt_sheet.md) | [Build K_nm Templates](03_build_knm_templates.md) |
| CSV files or logs | Build a complete local run | [From Raw Sources to Run](05_from_raw_sources_to_run.md) | [Deterministic Replay](06_deterministic_replay_for_debugging.md) |
| A new domain idea | Create a domainpack | [New Domain Checklist](01_new_domain_checklist.md) | [Minimal Domainpack](../getting-started/minimal_domainpack_5min.md) |
| A coupling matrix problem | Optimise or infer K | [Differentiable Kuramoto](04_differentiable_kuramoto.md) | [Coupling API](../reference/api/coupling.md) |
| An operations incident | Reproduce and explain decisions | [Deterministic Replay](06_deterministic_replay_for_debugging.md) | [Audit API](../reference/api/audit.md) |
| A notebook prototype | Move toward maintainable use | [Notebook to Production](../guide/notebook_to_production.md) | [Production Guide](../guide/production.md) |

## Decision Checklist

Before writing code, answer these questions:

1. What cycles in the system?
2. Which cycles should synchronise, and which should not?
3. What data source can observe each cycle?
4. Which channel is each signal: physical, informational, or symbolic?
5. What coupling assumptions are defensible?
6. What safety boundary prevents unsafe actuation?
7. What audit evidence would let someone replay or reject the result?

If you cannot answer one of those questions yet, treat the work as discovery,
not deployment.

## Fastest Runnable Path

Use this when you want a bounded first experiment and a reproducible trail:

Use this sequence for a first concrete result:

```bash
pip install scpn-phase-orchestrator
spo validate domainpacks/minimal_domain/binding_spec.yaml
spo run domainpacks/minimal_domain/binding_spec.yaml --steps 100 --seed 7
```

Then move to the raw-source tutorial when you need a domain-specific binding.

## When to Use Python Instead of the CLI

Use the Python facade when you need a deterministic local simulation inside a
notebook, service, or validation script:

```python
from scpn import Orchestrator

orch = Orchestrator.from_yaml("domainpacks/minimal_domain/binding_spec.yaml")
state = orch.run(steps=100, seed=7)
print(state.to_record())
```

Use the CLI when you need operator-facing validation, audit files, reports, or
review workflow output.

## Suggested follow-up once this is clear

After you have run the first cycle, capture the resolved runtime defaults and
review audit output before changing knobs. This keeps your second cycle focused on
the right axis: execution shape, coupling assumptions, and safety boundaries.

## Readiness rule before moving to implementation

Treat the row in this page as a hypothesis, not a design decision. Before writing
engine glue or custom automation, answer all routing questions with artifacts:

1. one selected tutorial or reference page,
2. one binding and one validation command,
3. one replay or report artifact for evidence.

Skipping this step is the common source of false starts in complex domains: teams
build custom code while the fundamental routing decision was still unresolved.

## Why this index is structured this way

Each row maps business problem → executable path → governance path, because the
repository contains both research surfaces and production surfaces. This page is
therefore meant to reduce mode confusion by forcing the first choice between
learning and operationalisation paths.
