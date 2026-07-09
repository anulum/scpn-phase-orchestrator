# Installation

## Install lane model

This page is split by operational lane, not by preference:

- **Base lane:** Python-only path for first-pass evaluation and smoke validation.
- **FFI lane:** add `rust` when deterministic performance checks are needed.
- **Domain lane:** add scoped extras (`queuewaves`, `quantum`, `fusion`, `plasma`)
  once base and smoke checks are stable.

The lane model is intended to avoid unnecessary dependency drift in early stages.
Every optional dependency adds operational risk if added before evidence gates pass.

## From PyPI

```bash
pip install scpn-phase-orchestrator
```

Requires Python 3.10+. Core dependencies: `numpy`, `scipy`, `pyyaml`, `click`.

After installing, confirm the environment is ready:

```bash
spo doctor
```

It reports the interpreter version, required dependencies, optional native
backends (Rust/Julia/Go/Mojo), and feature extras, and exits non-zero only when
a required component is missing. See [CLI Reference](../reference/cli.md#spo-doctor).

## One-command smoke test

After `spo doctor` reports the required dependencies as present, run the bundled
power-grid quickstart to exercise the full validate → run → replay → report path:

```bash
spo quickstart power
```

Expected output ends with a regime summary and an audit hash-chain verification
line. If this passes, the base install is sufficient for simulation, audit, and
report workflows. Optional extras (`rust`, `nn`, `queuewaves`, `plot`, etc.) are
only needed for their specific backends.

## Development Install

```bash
git clone https://github.com/anulum/scpn-phase-orchestrator.git
cd scpn-phase-orchestrator
pip install -e ".[dev]"
```

The `dev` extra includes pytest, hypothesis, ruff, mypy, bandit, coverage, mkdocs-material, pre-commit, and twine.

For a full local setup plus a minimal audited smoke run:

```bash
make quickstart
```

The target creates `.venv`, installs development extras, validates the
minimal domainpack, runs `spo run`, and reports the generated audit log.

## Optional Extras

| Extra | Installs | Purpose |
|-------|----------|---------|
| `rust` | `spo-kernel` | Rust FFI acceleration (7.3 us/step for N=16) |
| `quantum` | `scpn-quantum-control` | Quantum simulation adapter |
| `plasma` | `scpn-control` | Plasma/tokamak control adapter |
| `fusion` | `scpn-fusion-core` | Fusion equilibrium adapter |
| `queuewaves` | `fastapi`, `uvicorn`, `httpx`, `websockets` | QueueWaves cascade detector server |
| `nengo` | *(none -- pure-numpy LIF engine)* | Nengo-style SNN integration |
| `lava` | *(none -- external runtime managed outside SPO)* | Lava schedule handoff and optional operator-managed runtime integration |
| `eeg` | `pyedflib` | EDF ingestion for scalp-EEG case studies |
| `cardiac` | `wfdb` | WFDB ingestion for cardiac-ECG case studies |
| `otel` | `opentelemetry-api`, `opentelemetry-sdk` | OpenTelemetry tracing adapter |
| `plot` | `matplotlib` | Coherence plots and reporting |
| `notebook` | `jupyter`, `nbconvert`, `matplotlib` | Notebook demos |
| `scpn-all` | `spo-kernel`, `scpn-quantum-control`, `scpn-control`, `scpn-fusion-core` | All SCPN ecosystem packages |

Install multiple extras:

```bash
pip install "scpn-phase-orchestrator[queuewaves,plot,otel]"
```

## Rust FFI Build (Optional)

The Rust kernel `spo-kernel` provides a 10--50x speedup over the pure-Python engine. Building from source requires:

- Rust 1.75+ (`rustup update stable`)
- maturin (`pip install maturin`)

```bash
cd spo-kernel
maturin develop --release -m crates/spo-ffi/Cargo.toml
```

The Python engine auto-detects the Rust FFI at import time. If `spo-kernel` is not installed, the pure-NumPy fallback is used transparently.

## Platform Support

| Platform | Python | Rust FFI | Status |
|----------|--------|----------|--------|
| Linux x86_64 | 3.10--3.13 | Yes | Primary CI target |
| macOS arm64 | 3.10--3.13 | Yes | Tested |
| macOS x86_64 | 3.10--3.13 | Yes | Tested |
| Windows x86_64 | 3.10--3.13 | Yes | Tested |

## Verify Installation

```bash
python -c "from scpn_phase_orchestrator import UPDEEngine; print('OK')"
```

Check version:

```bash
python -c "import scpn_phase_orchestrator; print(scpn_phase_orchestrator.__version__)"
```

Verify CLI:

```bash
spo --help
```

Expected output (truncated to the first commands; run `spo --help` for the
full current list):

```
Usage: spo [OPTIONS] COMMAND [ARGS]...

  SCPN Phase Orchestrator CLI.

Options:
  --help  Show this message and exit.

Commands:
  audit-detector    Audit a detector's event-vs-null skill.
  demo              Run a self-contained demo.
  doctor            Check environment readiness.
  inspect           Inspect resolved runtime defaults.
  quickstart        Run the validate → run → replay → report path.
  replay            Replay an audit log and print summary.
  report            Generate coherence report from audit log.
  run               Run simulation from a binding spec.
  scaffold          Create a domainpack directory structure with template files.
  validate          Validate a binding specification file.
```

## Installation strategy by lane

Use this sequence when onboarding a new environment:

1. Start with a base install.
2. Verify CLI import and CLI command surface.
3. Add one optional dependency group for the intended lane.
4. Re-run `spo validate` and `spo report` smoke checks.

This keeps optional runtime stacks out of environments that do not need them,
reducing support complexity before CI-equivalent verification starts.

## Why this page stays explicit

The matrix of extras is long because it reflects real integration touchpoints,
not marketing breadth. Each explicit extra maps to a concrete optional stack.

For production, the practical guidance is to install only what is needed for the
deployment target, then treat any additional integrations as opt-in operational
steps after replay and stability checks pass.

## Environment readiness check (post-install)

At the end of installation, confirm:

- importability of the runtime core,
- basic CLI function,
- one bounded simulation path,
- and one persisted audit output path if policy control is in scope.

That check is what converts an install into an operationally ready workspace.

## Post-install decision flow

Use this flow whenever you are preparing a new environment:

1. **Install only required extras**
   - Start with the base package and add integrations (`queuewaves`, `quantum`,
     `fusion`, `plasma`) only after the base command path works.
2. **Run validation gates**
   - Confirm `python -c "from scpn_phase_orchestrator import UPDEEngine; ..."`
   - Confirm `spo validate` and `spo report` can execute on a known domainpack.
3. **Lock dependency surface**
   - Record selected lockfile and optional extras in deployment notes before
     first policy run.
4. **Run an audit-backed smoke path**
   - Execute one deterministic run and one `spo replay --verify` to prove
     reproducibility.

Each environment should document its lane choice (base, queuewaves, quantum,
WASM, or production Rust) so incident response can map a behavior change to either
code or dependency drift.

## Real-data corpora for early-warning studies

The case studies under `docs/studies/` use citation-only public corpora.  A
helper fetches the raw files into `data/` so the capstones can read them without
redistributing protected recordings:

```bash
python tools/fetch_real_corpora.py dakos        # Early-Warning-Signals climate records
python tools/fetch_real_corpora.py psml         # Zenodo power-system oscillation dataset (5.2 GB)
```

The EEG and cardiac corpora come from PhysioNet and require a free
credentialed account (CITI training).  Provide credentials via environment
variables or command-line flags:

```bash
export PHYSIONET_USER=your_user
export PHYSIONET_PASSWORD=your_password
python tools/fetch_real_corpora.py physionet-chbmit physionet-afdb
```

Run `python tools/fetch_real_corpora.py verify` to check which corpora are
complete.  Only derived, sealed artefacts are committed; raw recordings stay in
`data/` and are never added to the repository.

## Security and compliance checkpoints at install time

Installations that support external actuation keep two extra controls active:

- **Deterministic inputs**: fixed seed and fixed domainpack versions for first runs.
- **Audit-first promotion**: store generated audit files before enabling persistent
  control outputs.

This is the reason this project distinguishes local simulation readiness from
production readiness: both can look healthy, but only the latter has deterministic
replay and explicit policy gate checks completed.

## Where this fits in rollout planning

This page is the environment gate before any operational control path is
activated. Use the sequence below before enabling live actions:

1. install the minimal package set for the target lane,
2. verify `import`, `spo --help`, and one deterministic smoke command,
3. lock optional extras and dependency constraints for the lane,
4. verify audit and replay before any control output is consumed.

If step 4 cannot be completed in the same host profile, keep policy logic in
review mode and schedule the live lane on a separate controlled environment.
