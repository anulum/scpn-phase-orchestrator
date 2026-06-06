<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Onboarding Handbook -->

# Onboarding Handbook

This handbook is the first page to use when joining the project or
evaluating it for a new domain. It links installation, first run,
domainpack authoring, notebooks, demos, production deployment, and API
reference in one route.

## Why this handbook exists

It is structured as a single evidence ladder for teams with mixed roles:

- people choosing a domain pack get a direct validation path,
- operators get a review path with auditable replay,
- integrators get explicit API and adapter entry points.

The page is not a feature list. It is a sequence that reduces early confusion
between exploratory simulation and review-ready execution.

If the product purpose is still unclear, read the
[Use Cases and Value Map](use_cases.md) first; it maps domains, market value,
user roles, evidence boundaries, and API routes.

## First 20 Minutes

| Goal | Page | Outcome |
|------|------|---------|
| Understand the product | [Use Cases and Value Map](use_cases.md) | know the domain and user routes |
| Understand product value | [Executive Overview](executive_overview.md) | explain the market, operator, and evidence posture |
| Run a local simulation | [Quickstart](quickstart.md) | validate and run a minimal domainpack |
| Choose a tutorial path | [Choose a Use Case](../tutorials/00_choose_a_use_case.md) | select the right workflow |
| Inspect the Python entry point | [Python Facade API](../reference/api/api.md) | embed a reviewed binding in code |
| Review maturity | [Roadmap](../roadmap.md) | distinguish shipped, open, and research surfaces |

## Outcome Map

| Goal | Start here | Verify with |
|------|------------|-------------|
| Install the package | [Installation](installation.md) | `python -c "import scpn_phase_orchestrator"` |
| Run one simulation | [Quickstart](quickstart.md) | `spo demo --domain minimal_domain --steps 20` |
| Build a new domainpack | [Minimal Domainpack in 5 Minutes](minimal_domainpack_5min.md) | `spo validate domainpacks/<name>/binding_spec.yaml` |
| Understand the runtime pipeline | [Pipeline Execution](../concepts/pipeline_execution.md) | `spo run ... --audit audit.jsonl` |
| Move from notebook to service | [Notebook to Production](../guide/notebook_to_production.md) | `spo replay audit.jsonl --verify` |
| Explore APIs | [API Reference](../reference/api/index.md) | import the documented class/function |
| Present or teach the system | [Notebooks & Demos](../galleries/notebooks_and_demos.md) | run the listed notebook or demo command |
| Explain adoption value | [Executive Overview](executive_overview.md) | connect use cases, evidence, and buyer-facing value |
| Fix a local failure | [Troubleshooting](troubleshooting.md) | reproduce with exact command, seed, and environment |

## What to Explain First

When introducing SPO to a new reader, use this sequence:

1. The software is for repeated behaviour: waves, cycles, events, states, and
   coupled timing.
2. It converts those sources into phase variables so different telemetry can be
   compared with one mathematical contract.
3. It distinguishes useful coherence from harmful coherence.
4. It produces bounded proposals and replay records rather than hidden
   controller changes.
5. It supports research, simulation, and operator review without claiming that
   example domainpacks are calibrated for every live system.

## Explain the Value in One Page

When introducing SPO to a new stakeholder, keep this sequence:

1. **Phase is the shared language.** Different signals are converted into phase-aligned state variables.
2. **Coherence is context-dependent.** One system may need coupling, another may need de-synchronisation.
3. **Interventions are bounded by design.** Limits and rate caps prevent abrupt changes.
4. **Every recommendation is reviewable.** Audit records and replay commands create operational traceability.
5. **Deployment is staged by evidence.** Simulation-first, policy-first, and hardware-gated roll-outs keep risk visible.

If this sequence is not clear in the first meeting, route the audience back to
[Use Cases and Value Map](use_cases.md) before discussing any control surfaces.

## First Hour

1. Install the package:

   ```bash
   pip install scpn-phase-orchestrator
   ```

2. Run a domainpack demo:

   ```bash
   spo demo --domain minimal_domain --steps 20
   ```

3. Open the learning path for your role:

   ```text
   docs/getting-started/start_here.md
   ```

4. Read the contract files before authoring a new domain:

   - [Phase Contract](../specs/phase_contract.md)
   - [Binding Spec Schema](../specs/binding_spec.schema.json)
   - [Boundary Contract](../specs/boundary_contract.md)
   - [Policy DSL](../specs/policy_dsl.md)

5. Run the minimal authoring loop:

   ```bash
   spo validate domainpacks/minimal_domain/binding_spec.yaml
   spo run domainpacks/minimal_domain/binding_spec.yaml --steps 100 --seed 42
   ```

## Role Routes

| Role | Read | Run | Extend |
|------|------|-----|--------|
| Domain author | [New Domain Checklist](../tutorials/01_new_domain_checklist.md) | `spo validate` | `domainpacks/<name>/binding_spec.yaml` |
| Research user | [Kuramoto Theory](../concepts/kuramoto_theory.md) | notebooks `02`, `06`, `17`, `18`, `19` | engine parameters and monitors |
| Platform operator | [Production Deployment](../guide/production.md) | `spo serve ...` | Prometheus, OpenTelemetry, gRPC |
| API integrator | [Python Facade API](../reference/api/api.md) | Python imports | adapters, server, CLI |
| Demo presenter | [Interactive Tools](../guide/interactive_tools.md) | Streamlit, WASM, `spo demo` | domainpack gallery |
| Contributor | [Contributor Onboarding](../guide/contributor_onboarding.md) | scoped checks | docs, tests, examples |

## Repository Map

| Path | Purpose |
|------|---------|
| `src/scpn_phase_orchestrator/` | Python package |
| `spo-kernel/` | Rust acceleration crates and WASM package |
| `domainpacks/` | Reproducible domain mappings |
| `examples/` | Terminal-first demonstrations |
| `notebooks/` | Notebook demonstrations and analysis workflows |
| `docs/` | MkDocs public documentation |
| `tests/` | Unit, integration, property, and regression tests |
| `benchmarks/` | Benchmark harnesses and measured reference scripts |

## Quality Gates For New Work

Before considering a new module, domainpack, or guide complete:

- The public API is documented or linked from [API Reference](../reference/api/index.md).
- The user path is documented in a guide, tutorial, notebook, or example.
- The validation command is listed next to the instructions.
- Any benchmark number is reproducible from a command in the same page.
- The backend fallback path is clear when optional Rust, JAX, or external
  dependencies are unavailable.

## Documentation Coverage

See [Documentation Coverage](../reference/documentation_coverage.md) for the
current public documentation inventory and the policy for API, guide,
notebook, and demo coverage.

For planned work, see the [Public Roadmap](../roadmap.md).

## Suggested onboarding milestones

Recommended order for an engineering team:

1. **Understand**: read use cases and run quickstart.
2. **Validate**: execute the full install validation and a minimal domainpack run.
3. **Review**: inspect generated audit output before any control action.
4. **Harden**: set backend selection, lock policy, and dependency posture.
5. **Extend**: move through tutorial and domainpack authoring only after each
   milestone passes.

This sequence minimizes drift between documentation claims and what a team can
reproduce on day one.

## Operator handoff format

When passing the workspace to another operator, include four artifacts in one place:

- environment details (Python/Rust versions and backend lane),
- last successful simulation command and seed,
- latest audit report and replay command,
- validation pass result and lockfile choice.

The same four artifacts are the minimum evidence set for safe incident follow-up.

## Use this page when alignment is unclear

If a team is uncertain about where to start, use this order:

1. read this onboarding page with one agreed role,
2. open one domainpack example (`minimal_domain`) and run the listed validation
   commands,
3. compare expected outputs with `docs/reference/documentation_coverage.md`,
4. only then open tool-specific pages (`studio`, `api`, `production`).

This avoids switching between guides before the team shares a common baseline
for what counts as verified evidence.

## What to send after onboarding

After completing this page, pass three artifacts to operations:

- a pinned environment spec (`Python`, optional extras, backend lane),
- the latest validation command set and outcomes,
- the first replay reference file for review.

That package is usually enough to decide whether the team is ready for domain
authoring or should stay in evaluation mode for one more cycle.
