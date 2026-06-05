---
hide:
  - navigation
---

# SCPN Phase Orchestrator

<div align="center">

![Synchronization Manifold](assets/synchronization_manifold.png){ width="720" }

[![CI](https://github.com/anulum/scpn-phase-orchestrator/actions/workflows/ci.yml/badge.svg)](https://github.com/anulum/scpn-phase-orchestrator/actions/workflows/ci.yml)
[![CodeQL](https://github.com/anulum/scpn-phase-orchestrator/actions/workflows/codeql.yml/badge.svg)](https://github.com/anulum/scpn-phase-orchestrator/actions/workflows/codeql.yml)
[![OpenSSF Scorecard](https://github.com/anulum/scpn-phase-orchestrator/actions/workflows/scorecard.yml/badge.svg)](https://github.com/anulum/scpn-phase-orchestrator/actions/workflows/scorecard.yml)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/12193/badge)](https://www.bestpractices.dev/projects/12193)
[![PyPI](https://img.shields.io/pypi/v/scpn-phase-orchestrator)](https://pypi.org/project/scpn-phase-orchestrator/)
[![Coverage](https://img.shields.io/badge/coverage-99%25-brightgreen)](https://github.com/anulum/scpn-phase-orchestrator)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-purple)](https://github.com/anulum/scpn-phase-orchestrator/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Rust FFI](https://img.shields.io/badge/Rust-spo--kernel-orange)](https://github.com/anulum/scpn-phase-orchestrator/tree/main/spo-kernel)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/anulum/scpn-phase-orchestrator/blob/main/.pre-commit-config.yaml)
[![REUSE](https://img.shields.io/badge/REUSE-compliant-green)](https://reuse.software/)

**Generated capability inventory | dedicated module-owned tests | Rust and Python backends | 36 domainpacks | reviewed evidence boundaries**

</div>

---

Domain-agnostic coherence control compiler built on Kuramoto/UPDE phase dynamics. Any hierarchical coupled-cycle system --- plasma, cloud infrastructure, traffic, power grids, factories, biology --- maps onto the same engine.

SPO is best read as a compiler from domain telemetry to phase-control evidence:
bind signals, extract oscillator phases, run coupled dynamics, measure
coherence, classify regimes, and emit bounded review artefacts. It is useful
when a system has repeated behaviour and the operator needs to know whether
synchrony is helpful, harmful, causal, or controllable.

For evaluation, start with the practical question: does the target system have
waves, cycles, retries, stages, rotations, or event loops whose timing matters?
If yes, SPO can express those sources as phase, test coupling hypotheses,
separate useful from harmful synchrony, and preserve a replayable control
review record.

## Current Release Boundary

Version `0.8.0` marks the PHA-C formal-acceptance release. The public docs now
route readers from use-case selection through Python APIs, tutorials, notebooks,
benchmark snapshots, and PHA-C proof-obligation pages without requiring them to
reverse-engineer the source tree.

| Reader concern | Where the release answers it |
|----------------|------------------------------|
| What problem does SPO solve? | [Use Cases and Value Map](getting-started/use_cases.md) and [Executive Overview](getting-started/executive_overview.md) |
| How do I run it? | [Quickstart](getting-started/quickstart.md), [Hello World](getting-started/hello_world.md), and [From Raw Sources to Run](tutorials/05_from_raw_sources_to_run.md) |
| How do I embed it? | [Python Facade API](reference/api/api.md) and [API Overview](reference/api/index.md) |
| What evidence supports PHA-C? | [PHA-C Acceptance Chain](reference/api/upde_pha_c_acceptance.md), [PHA-C Lean Proof Obligation](reference/api/upde_pha_c_formal_obligation.md), and [Reference Benchmark Snapshot](galleries/reference_benchmark_snapshot.md) |
| What remains review-only? | [Public Roadmap](roadmap.md), [Adapters](guide/adapters.md), and [Release Hygiene](RELEASE_HYGIENE.md) |

The benchmark snapshot remains local regression evidence unless the raw run
records CPU/core isolation and host-load controls. Live hardware writes remain
adapter-scoped and disabled by default.

## What This Means in Practice

SPO gives a team one reviewable path from raw operational traces to bounded
control evidence:

| Stage | Practical output | Why it matters |
|-------|------------------|----------------|
| Bind | `binding_spec.yaml` with sources, channels, boundaries, and assumptions | domain knowledge becomes inspectable instead of living in notebooks |
| Extract | physical, informational, and symbolic phases on one timeline | waves, events, and states can be compared mathematically |
| Simulate | Kuramoto, UPDE, Stuart-Landau, delay, stochastic, simplicial, or inertial dynamics | teams can test synchronisation and desynchronisation hypotheses before deployment |
| Supervise | regimes, Petri nets, value guards, and bounded action proposals | unsafe or unsupported control paths stay behind review gates |
| Audit | hash-linked logs, deterministic replay, benchmark snapshots, and Studio panels | decisions can be reproduced, rejected, or promoted with evidence |

| If you are asking... | Start here |
|----------------------|------------|
| What is this software for? | [Use Cases and Value Map](getting-started/use_cases.md) |
| What is the business/operator value? | [Executive Overview](getting-started/executive_overview.md) |
| How do I run something in five minutes? | [Quickstart](getting-started/quickstart.md) |
| How do I decide what counts as an oscillator? | [Oscillator Hunt Sheet](tutorials/02_oscillator_hunt_sheet.md) |
| How do I move from raw data to a run? | [End-to-End From Raw Sources](tutorials/05_from_raw_sources_to_run.md) |
| How do I use it from Python? | [Python Facade API](reference/api/api.md) |
| How do I understand notebooks and demos? | [Notebooks and Demos](galleries/notebooks_and_demos.md) |
| What is implemented versus still open? | [Public Roadmap](roadmap.md) |

## First Evaluation Path

1. Read the [Use Cases and Value Map](getting-started/use_cases.md).
2. Run `spo demo --domain minimal_domain --steps 20`.
3. Validate one binding spec with `spo validate`.
4. Replay one audited run with `spo replay --verify`.
5. Use the [API Reference](reference/api/index.md) only after choosing the
   relevant surface.

## Architecture

```
Domain Binder ─► Oscillators (P/I/S) ─► UPDE Engine (9 variants) ─► Supervisor ─► Actuation
     │                  │                     │                          │             │
binding_spec.yaml   3-channel          Kuramoto, Stuart-Landau,     Policy DSL   ControlAction
                    extraction         Inertial, Market, Swarmalator + Petri Net   + Projector
                    (Physical /        Stochastic, Geometric, Delay  + Regime FSM
                     Informational /   Simplicial + Ott-Antonsen     + MPC
                     Symbolic)         + Rust FFI / JAX GPU
```

## Features

<div class="grid cards" markdown>

-   **36 Domainpacks**

    ---

    Plug-and-play domain bindings: plasma control, power grids, traffic flow, cardiac rhythm, neuroscience EEG, swarm robotics, queuewaves, brain connectome, sleep architecture, and 27 more.

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

    SHA256-chained audit trail in JSONL format. Every simulation step is hash-linked and re-executable. Tolerance-based replay verification (atol=1e-6) plus hash-chain integrity check.

    [:octicons-arrow-right-24: Audit Trace Spec](specs/audit_trace.md)

-   **Differentiable (JAX)**

    ---

    `nn/` module: KuramotoLayer, StuartLandauLayer, simplicial 3-body, BOLD, reservoir, UDE, inverse pipeline, OIM. All JIT-compilable, vmap-compatible, GPU-ready.

    [:octicons-arrow-right-24: Differentiable Guide](guide/differentiable_kuramoto.md)

-   **9 ODE Engines**

    ---

    Standard Kuramoto, Stuart-Landau, inertial (power grids), market (finance), swarmalator (robotics), stochastic, geometric, delay, simplicial. Plus Ott-Antonsen mean-field reduction.

    [:octicons-arrow-right-24: Advanced Dynamics](guide/advanced_dynamics.md)

-   **15 Monitors**

    ---

    Chimera detection, EVS entrainment, Lyapunov exponents, entropy production, PAC, PID, transfer entropy, winding numbers, ITPC, sleep staging, STL safety. Beyond R alone.

    [:octicons-arrow-right-24: Analysis Toolkit](guide/analysis_toolkit.md)

-   **Inverse Kuramoto**

    ---

    Infer coupling matrix K and frequencies ω from observed data (EEG, sensors, markets) by backpropagating through the ODE solver. L1 sparsity discovers network topology.

    [:octicons-arrow-right-24: nn/ API](reference/api/nn.md)

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
[:octicons-arrow-right-24: Onboarding](getting-started/onboarding.md){ .md-button }

## Navigation

| Section | Description |
|---------|-------------|
| [Getting Started](getting-started/onboarding.md) | Executive overview, onboarding, install, quickstart, hello world tutorial |
| [Concepts](concepts/system_overview.md) | System overview, oscillators, control knobs, imprint model |
| [Guides](guide/stuart_landau.md) | Stuart-Landau, QueueWaves, Rust FFI, adapters, production |
| [Specifications](specs/binding_spec.schema.json) | Binding schema, UPDE numerics, policy DSL, all contracts |
| [Tutorials](tutorials/01_new_domain_checklist.md) | New domain checklist, oscillator hunt sheet, Knm templates |
| [API Reference](reference/api/index.md) | Full Python API docs (mkdocstrings) |
| [Gallery](galleries/domainpack_gallery.md) | All 36 domainpacks, notebooks, examples, and demos |

The current documentation inventory and API-reference guardrails are tracked in
[Documentation Coverage](reference/documentation_coverage.md).

---

**Contact:** [protoscience@anulum.li](mailto:protoscience@anulum.li) |
[GitHub Discussions](https://github.com/anulum/scpn-phase-orchestrator/discussions) |
[www.anulum.li](https://www.anulum.li)

<p align="center">
  <a href="https://www.anulum.li">
    <img src="assets/anulum_logo_company.jpg" width="180" alt="ANULUM">
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.anulum.li">
    <img src="assets/fortis_studio_logo.jpg" width="180" alt="Fortis Studio">
  </a>
  <br>
  <em>Developed by <a href="https://www.anulum.li">ANULUM</a> / Fortis Studio</em>
  <br><br>
  <b>License:</b> AGPL-3.0-or-later | Commercial licensing available
  <br>
  © 1996–2026 Miroslav Šotek. All rights reserved.
</p>
