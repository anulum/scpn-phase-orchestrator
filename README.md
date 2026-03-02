# SCPN Phase Orchestrator

Domain-agnostic coherence control compiler built on Kuramoto/UPDE phase dynamics.

[![CI](https://github.com/anulum/scpn-phase-orchestrator/actions/workflows/ci.yml/badge.svg)](https://github.com/anulum/scpn-phase-orchestrator/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/scpn-phase-orchestrator)](https://pypi.org/project/scpn-phase-orchestrator/)
[![Docs](https://img.shields.io/badge/docs-gh--pages-blue)](https://anulum.github.io/scpn-phase-orchestrator/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## What It Does

Treats Kuramoto phase dynamics as a universal synchrony state-space.
Any hierarchical coupled-cycle system — plasma, cloud infrastructure,
traffic, power grids, factories, biology — maps onto the same engine.

## Core Pipeline

```
Domain Binder → Oscillator Extractors (P/I/S) → UPDE Engine → Supervisor → Actuation Mapper
```

### 3-Channel Oscillator Model

| Channel | Source | Phase Extraction |
|---------|--------|-----------------|
| **Physical (P)** | Continuous waveforms | Hilbert transform, zero-crossing |
| **Informational (I)** | Event/decision streams | Event-phase from message timing |
| **Symbolic (S)** | Discrete state sequences | Ring-phase θ=2πs/N, graph-walk |

### 4 Universal Control Knobs

| Knob | Meaning |
|------|---------|
| **K** | Coupling strength (Knm matrix) |
| **α** | Phase lag (transport/actuator delays) |
| **ζ** | Driver strength (external forcing) |
| **Ψ** | Reference phase (control target) |

### Dual Objective

- **R_good**: Coherence to maintain (actuator ↔ target phase-lock)
- **R_bad**: Coherence to suppress (harmful mode-locking)

## Quickstart

```bash
pip install -e ".[dev]"

# Validate a domain binding spec
spo validate domainpacks/minimal_domain/binding_spec.yaml

# Run a domain simulation
spo run domainpacks/queuewaves/binding_spec.yaml --steps 1000

# Replay from audit log
spo replay audit.jsonl --output report.json
```

## Platform Support

| Platform | Python engine | Rust FFI (optional) |
|----------|--------------|---------------------|
| Linux | Full | Full |
| macOS | Full | Full |
| Windows | Full | Experimental (requires MSVC toolchain) |

The PyPI package is pure Python. Rust FFI provides optional acceleration
and is built from source via `maturin develop`.

## Domainpacks

| Pack | Domain | Purpose |
|------|--------|---------|
| `minimal_domain` | Synthetic | 4-oscillator test harness |
| `queuewaves` | Cloud/queues | Retry storm desynchronisation |
| `geometry_walk` | Graph systems | Random-walk phase coupling |
| `bio_stub` | Biology | SCPN-compatible oscillator template |
| `manufacturing_spc` | Manufacturing | Statistical process control (3 layers, 9 oscillators) |

## Development

```bash
pip install -e ".[dev]"
ruff check src/ tests/
ruff format --check src/ tests/
pytest tests/ -v --tb=short
mkdocs build
```

## License

AGPL-3.0-or-later. Commercial licensing available — contact protoscience@anulum.li.

## Citation

See [CITATION.cff](CITATION.cff).

---

© 1998–2026 Miroslav Šotek. All rights reserved.
Contact: www.anulum.li | protoscience@anulum.li
