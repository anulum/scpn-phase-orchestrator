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
| API integrator | [Core API](../reference/api/core.md) | Python imports | adapters, server, CLI |
| Demo presenter | [Interactive Tools](../guide/interactive_tools.md) | Streamlit, WASM, `spo demo` | domainpack gallery |

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
