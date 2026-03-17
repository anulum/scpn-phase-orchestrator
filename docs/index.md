---
hide:
  - navigation
---

# SCPN Phase Orchestrator

<div align="center">

![Synchronization Manifold](assets/synchronization_manifold.png){ width="720" }

[![CI](https://github.com/anulum/scpn-phase-orchestrator/actions/workflows/ci.yml/badge.svg)](https://github.com/anulum/scpn-phase-orchestrator/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/scpn-phase-orchestrator)](https://pypi.org/project/scpn-phase-orchestrator/)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://anulum.github.io/scpn-phase-orchestrator/)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-purple)](https://github.com/anulum/scpn-phase-orchestrator/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

**1263 Python tests | 191 Rust tests | 100% coverage | 25 domainpacks**

</div>

---

Domain-agnostic coherence control compiler built on Kuramoto/UPDE phase dynamics. Any hierarchical coupled-cycle system --- plasma, cloud infrastructure, traffic, power grids, factories, biology --- maps onto the same engine.

## Architecture

```
Domain Binder ─► Oscillators (P/I/S) ─► UPDE Engine ─► Supervisor ─► Actuation
     │                  │                     │             │             │
binding_spec.yaml   3-channel          Kuramoto ODE     Policy DSL   ControlAction
                    extraction         + Stuart-Landau   + Petri Net   + Projector
                    (Physical /        + RK4/RK45        + Regime FSM
                     Informational /   + Rust FFI
                     Symbolic)
```

## Features

<div class="grid cards" markdown>

-   **25 Domainpacks**

    ---

    Plug-and-play domain bindings: plasma control, power grids, traffic flow, cardiac rhythm, neuroscience EEG, swarm robotics, queuewaves, identity coherence, and 17 more.

    [:octicons-arrow-right-24: Gallery](galleries/domainpack_gallery.md)

-   **3-Channel Model (P/I/S)**

    ---

    Physical, Informational, and Symbolic oscillator extraction. Each domain signal decomposes into one or more channels with dedicated extractors (Hilbert, wavelet, zero-crossing, event, ring, graph).

    [:octicons-arrow-right-24: Oscillators](concepts/oscillators_PIS.md)

-   **Rust-Accelerated**

    ---

    `spo-kernel` FFI via PyO3/maturin. 7.3 us/step for N=16 oscillators. Pure-Python fallback ships by default.

    [:octicons-arrow-right-24: Rust FFI Guide](guide/rust_ffi.md)

-   **Stuart-Landau**

    ---

    Phase + amplitude coupled ODEs. Subcritical bifurcation detection, PAC (phase-amplitude coupling) metrics, and amplitude-aware supervision.

    [:octicons-arrow-right-24: Stuart-Landau Guide](guide/stuart_landau.md)

-   **Policy DSL**

    ---

    YAML-based declarative supervisor rules. Condition-action pairs triggered by regime state and metric thresholds. Rate-limited, TTL-aware, projector-clipped.

    [:octicons-arrow-right-24: Policy DSL Spec](specs/policy_dsl.md)

-   **Petri Net FSM**

    ---

    Multi-phase protocol sequencing via place/transition nets with guard expressions. Regime-place mapping drives supervisor decisions through protocol stages.

    [:octicons-arrow-right-24: Phase Contract](specs/phase_contract.md)

-   **QueueWaves**

    ---

    Real-time cascade failure detector for microservice architectures. Scrapes queue depths, extracts phases, detects desynchronization before cascading failures propagate.

    [:octicons-arrow-right-24: QueueWaves Guide](guide/queuewaves.md)

-   **Deterministic Replay**

    ---

    SHA256-chained audit trail in JSONL format. Every simulation step is hash-linked and re-executable. Bit-exact verification of entire runs.

    [:octicons-arrow-right-24: Audit Trace Spec](specs/audit_trace.md)

</div>

## Quick Install

```bash
pip install scpn-phase-orchestrator
```

```python
from scpn_phase_orchestrator import UPDEEngine
print("OK")
```

[:octicons-arrow-right-24: Installation](getting-started/installation.md){ .md-button }
[:octicons-arrow-right-24: Quickstart](getting-started/quickstart.md){ .md-button .md-button--primary }

## Navigation

| Section | Description |
|---------|-------------|
| [Getting Started](getting-started/installation.md) | Install, quickstart, hello world tutorial |
| [Concepts](concepts/system_overview.md) | System overview, oscillators, control knobs, imprint model |
| [Guides](guide/stuart_landau.md) | Stuart-Landau, QueueWaves, Rust FFI, adapters, production |
| [Specifications](specs/binding_spec.schema.json) | Binding schema, UPDE numerics, policy DSL, all contracts |
| [Tutorials](tutorials/01_new_domain_checklist.md) | New domain checklist, oscillator hunt sheet, Knm templates |
| [API Reference](reference/api/index.md) | Full Python API docs (mkdocstrings) |
| [Gallery](galleries/domainpack_gallery.md) | All 24 domainpacks with descriptions |

---

<div align="center">

**License:** AGPL-3.0-or-later | Commercial licensing available

(c) 1998--2026 Miroslav Sotek. All rights reserved.

[www.anulum.li](https://www.anulum.li) | [protoscience@anulum.li](mailto:protoscience@anulum.li)

</div>
