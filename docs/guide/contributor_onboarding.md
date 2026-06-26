<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Contributor Onboarding -->

# Contributor Onboarding

This page is the contributor path for repository work. It complements the
user-facing [Onboarding Handbook](../getting-started/onboarding.md) with
setup commands, file placement, verification gates, and documentation duties.

## First Local Setup

```bash
git clone https://github.com/anulum/scpn-phase-orchestrator.git
cd scpn-phase-orchestrator
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,queuewaves,plot,notebook]"
```

Run the smoke path:

```bash
spo validate domainpacks/minimal_domain/binding_spec.yaml
spo run domainpacks/minimal_domain/binding_spec.yaml --steps 50 --seed 42 --audit /tmp/spo-smoke.jsonl
spo replay /tmp/spo-smoke.jsonl --verify
```

Equivalent make target:

```bash
make quickstart
```

## Repository Map

| Path | Contributor responsibility |
|------|-----------------------------|
| `src/scpn_phase_orchestrator/` | public package code and typed API surfaces |
| `tests/` | unit, integration, property, parity, regression, and app tests |
| `spo-kernel/` | Rust acceleration crates, FFI, and WASM package |
| `domainpacks/` | reusable binding specs, policies, run scripts, and README files |
| `examples/` | terminal-first demonstrations |
| `notebooks/` | notebook demonstrations and analysis workflows |
| `docs/` | public MkDocs documentation |
| `requirements/` | hash-pinned dependency lockfiles |
| `tools/` | repository tools and interactive demos |
| `benchmarks/` | benchmark harnesses and reference scripts |

## Common Workflows

| Task | Start with | Verify with |
|------|------------|-------------|
| Change public Python API | owning `src/` module and API page | focused `pytest`, `ruff`, `mypy`, docs build |
| Add a domainpack | `domainpacks/<name>/binding_spec.yaml` | `spo validate`, short `spo run`, README |
| Add a notebook | closest existing notebook | notebook matrix row and `nbconvert` when practical |
| Add a CLI or demo path | `examples/`, `tools/`, or `cli.py` | documented command and smoke run |
| Change Rust acceleration | `spo-kernel/` crate and Python fallback | Rust tests plus Python parity test |
| Change docs only | owning page and `mkdocs.yml` | strict MkDocs build |

## Local Verification

Use scoped checks for the files you touched:

```bash
ruff check src/ tests/
ruff format --check src/ tests/
pytest tests/ -v --tb=short
bandit -r src/ -c pyproject.toml
PYTHONPATH=src mkdocs build --strict --clean
```

Rust paths:

```bash
cd spo-kernel
cargo fmt --check
cargo test --workspace --exclude spo-ffi
```

Dependency lock checks:

```bash
make lock-check
```

Full local preflight exists, but it is heavier than the normal scoped commit
gate:

```bash
python tools/preflight.py
```

## Documentation Duties

Every user-visible change needs documentation at the level where users will
look for it:

| Change | Documentation target |
|--------|----------------------|
| public class, function, or module | module docstring plus API reference page under `docs/reference/api/` |
| new domainpack | `domainpacks/<name>/README.md` and gallery entry when appropriate |
| new notebook | [Notebook Execution Matrix](../galleries/notebook_execution_matrix.md) |
| new demo command | [Notebooks & Demos](../galleries/notebooks_and_demos.md) or the owning guide |
| new runtime default | resolved defaults spec or the owning guide |
| new deployment or service path | production guide, interactive tools guide, or troubleshooting |

Keep benchmark numbers tied to a command and environment. If a value is a
historical snapshot, label it as a snapshot rather than current validation.
For public modules, prefer a precise module docstring over a generic label:
state what contract the module owns, which backend or transport assumptions it
makes, and whether invalid inputs fail closed or propagate to an owning
validator.
The CI lint job runs an `interrogate` 100% docstring ratchet over
`src/scpn_phase_orchestrator`; generated gRPC stubs and protocol-defined
`__init__`/dunder methods are the only documented source-level exemptions.

## Domainpack Contribution Checklist

1. Add `domainpacks/<name>/binding_spec.yaml`.
2. Add `README.md` with the seven-line public header.
3. Add `policy.yaml` only when the domain has declarative supervisor rules.
4. Validate with `spo validate domainpacks/<name>/binding_spec.yaml`.
5. Run a short deterministic simulation with `--seed`.
6. Document the expected channels, layers, objectives, and run command.
7. Add a notebook or terminal example if the domain introduces a new pattern.

## Notebook Contribution Checklist

1. Keep default cell execution inside the CI timeout unless the notebook is
   explicitly local-only.
2. Use deterministic seeds for simulations.
3. Avoid hidden external services or private data paths.
4. Add the notebook to [Notebook Execution Matrix](../galleries/notebook_execution_matrix.md).
5. Link a terminal-first equivalent when possible.
6. If optional dependencies are required, list the exact package extra.

## Pull Request Readiness

Before opening a PR:

- run the scoped checks that match the touched surface
- update docs and examples alongside code
- ensure `git diff --check` is clean
- ensure no credentials, private paths, generated caches, or local artefacts are staged
- keep the change focused enough to review as one topic
- record any skipped heavy check or local-only limitation in the PR body

## Where To Go Next

- [Testing Guide](testing.md)
- [Troubleshooting](../getting-started/troubleshooting.md)
- [Documentation Coverage](../reference/documentation_coverage.md)
- [Public Roadmap](../roadmap.md)
