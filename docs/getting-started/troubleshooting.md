<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Troubleshooting Guide -->

# Troubleshooting

Use this page when installation, local docs, notebooks, domainpack validation,
or optional acceleration does not behave as expected.

## First Checks

Run these from the repository root before debugging a specific subsystem:

```bash
python --version
python -m pip --version
python -m pip install -e ".[dev,notebook,plot]"
PYTHONPATH=src python -c "import scpn_phase_orchestrator as spo; print(spo.__version__)"
spo --help
spo doctor
```

`spo doctor` is the fastest single-command triage: it reports the interpreter
version, the required dependencies, the optional native backends
(Rust/Julia/Go/Mojo), and the optional feature extras, marking each as available
or missing. Start here when acceleration or an optional feature is not behaving
as expected.

Expected baseline:

| Check | Expected result |
|-------|-----------------|
| `python --version` | Python 3.10 or newer |
| editable import | `scpn_phase_orchestrator` imports without `ModuleNotFoundError` |
| `spo --help` | CLI command list prints |
| `PYTHONPATH=src` | required when running from a source checkout without editable install |

## Install Problems

### `ModuleNotFoundError: scpn_phase_orchestrator`

The package is not installed in the active environment, or the source tree is
not on `PYTHONPATH`.

```bash
python -m pip install -e ".[dev]"
PYTHONPATH=src python -c "import scpn_phase_orchestrator"
```

For docs builds, use:

```bash
PYTHONPATH=src mkdocs build --strict --clean
```

### Optional extras are missing

Install the smallest extra that matches the workflow:

| Workflow | Command |
|----------|---------|
| Notebooks | `python -m pip install -e ".[notebook,plot]"` |
| JAX/nn guides | `python -m pip install -e ".[nn]"` |
| QueueWaves service | `python -m pip install -e ".[queuewaves]"` |
| OpenTelemetry export | `python -m pip install -e ".[otel]"` |
| Full local development | `python -m pip install -e ".[dev,queuewaves,plot,notebook]"` |

If a lockfile workflow is required, use the repository lockfiles in
`requirements/` rather than mixing ad-hoc dependency versions.

## Rust FFI Build Problems

Rust FFI is optional. The Python package works without it; Rust provides the
accelerated path for supported kernels.

### `maturin` or Rust toolchain missing

Install the Rust toolchain and maturin, then build from the repository root:

```bash
rustc --version
cargo --version
python -m pip install maturin
maturin develop --manifest-path spo-kernel/Cargo.toml
```

### Windows MSVC errors

Use a Python version supported by the repository lockfiles and run from a
Developer PowerShell with the MSVC toolchain available. If Rust cannot build,
continue with the Python fallback and run the affected guide without FFI-only
claims.

### FFI import falls back to Python

Check whether the Python module is importable first, then inspect the Rust
package separately:

```bash
PYTHONPATH=src python -c "import scpn_phase_orchestrator"
python -c "import spo_kernel"
```

If `spo_kernel` is absent but the Python import succeeds, the fallback path is
active.

## Domainpack Validation Problems

Run validation before simulation:

```bash
spo validate domainpacks/minimal_domain/binding_spec.yaml
spo inspect domainpacks/minimal_domain/binding_spec.yaml
```

Common causes:

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| missing layer or oscillator ID | `binding_spec.yaml` references an undeclared oscillator | align layer `oscillator_ids`, coupling scopes, and objectives |
| boundary validation fails | threshold names or metric paths do not match runtime metrics | compare with `domainpacks/minimal_domain/binding_spec.yaml` |
| policy rule does not fire | rule condition never becomes true or scope is wrong | dry-run with a short audited run and inspect metrics |
| run succeeds but `R` is low | initial phases, weak coupling, or short run length | increase steps, inspect `K`, and compare with the domain README |

When creating a new domainpack, start from
[Minimal Domainpack in 5 Minutes](minimal_domainpack_5min.md), then compare
against an existing README in `domainpacks/`.

## Audit Replay Problems

Always run audited simulations with an explicit seed when reproducibility
matters:

```bash
spo run domainpacks/minimal_domain/binding_spec.yaml --steps 500 --seed 42 --audit audit.jsonl
spo replay audit.jsonl --verify
```

If replay fails:

| Failure | Check |
|---------|-------|
| hash-chain mismatch | audit file was edited, truncated, or concatenated |
| numerical mismatch | dependency versions, backend path, or tolerance changed |
| missing audit fields | audit file came from an older run format |

Keep the original audit file unchanged and create a new run for comparison.

## Notebook Problems

Install notebook extras and run from the repository root:

```bash
python -m pip install -e ".[dev,notebook,plot]"
jupyter lab notebooks/
```

CI executes the notebook suite on Python 3.12 with:

```bash
jupyter nbconvert --execute --to notebook notebooks/*.ipynb --ExecutePreprocessor.timeout=120
```

If a notebook fails locally:

| Symptom | Fix |
|---------|-----|
| import error | install editable mode or set `PYTHONPATH=src` |
| optional dependency missing | install the extra listed in the notebook matrix |
| timeout | run the notebook interactively, reduce demo steps, or use the matching terminal example |
| stale output | restart kernel and run all cells from a clean state |

See [Notebook Execution Matrix](../galleries/notebook_execution_matrix.md)
for notebook-specific extras and CI expectations.

## MkDocs Problems

Use the same source-path pattern as the local validation gate:

```bash
PYTHONPATH=src mkdocs build --strict --clean
```

Common failures:

| Symptom | Fix |
|---------|-----|
| `Expected nav to be a list` | check indentation in `mkdocs.yml` |
| mkdocstrings cannot import package | set `PYTHONPATH=src` or install editable mode |
| broken relative link | link to a file under `docs/`, or write root-file references as literal paths |
| page not in nav | add public pages to `mkdocs.yml`; internal notes can remain unnav'd |

## Demo Problems

For source checkouts, prefix terminal examples with `PYTHONPATH=src`:

```bash
PYTHONPATH=src python examples/supervisor_advantage.py
PYTHONPATH=src python examples/failure_recovery.py
spo demo --domain minimal_domain --steps 20
```

For Streamlit tools:

```bash
python -m pip install -e ".[dev,plot]"
streamlit run tools/spo_studio.py
```

If a browser or GUI dependency is not available, use the matching CLI or
terminal example first.

## Reporting a Reproducible Issue

Capture:

- operating system and Python version
- install command or lockfile used
- exact command that failed
- full traceback
- domainpack path and seed if simulation-related
- whether Rust FFI was installed or Python fallback was used

Do not include credentials, tokens, `.env` contents, or private data paths.

## Standard triage flow for repeatable failures

For failures that reproduce on two runs, apply this sequence:

1. run `git status --short` to confirm a clean baseline,
2. run the documented baseline command for the failing area,
3. include `--seed` and lockfile in the issue packet,
4. rerun with `spo replay --verify` in the same environment.

If a failure appears only after a backend swap (Python ↔ Rust), capture both
runs and compare recorded command outputs before changing test cases.
