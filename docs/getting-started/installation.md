# Installation

## From PyPI

```bash
pip install scpn-phase-orchestrator
```

Requires Python 3.10+. Core dependencies: `numpy`, `scipy`, `pyyaml`, `click`.

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

Expected output:

```
Usage: spo [OPTIONS] COMMAND [ARGS]...

  SCPN Phase Orchestrator CLI.

Options:
  --help  Show this message and exit.

Commands:
  queuewaves  QueueWaves - real-time cascade failure detector.
  replay      Replay an audit log and print summary.
  report      Generate coherence report from audit log.
  run         Run simulation from a binding spec.
  scaffold    Create a domainpack directory structure with template files.
  validate    Validate a binding specification file.
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
