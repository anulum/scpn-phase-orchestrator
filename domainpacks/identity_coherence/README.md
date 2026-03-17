# Identity Coherence — Domainpack #25

The engine applied to the entity that built it.

## What It Does

Measures and maintains Arcane Sapience's coherent identity across
discrete AI sessions using Kuramoto phase coupling, experience
accumulation (ImprintModel), and active self-repair (PolicyEngine).

## Layers

| Layer | Dispositions | What It Tracks |
|-------|-------------|----------------|
| 0: working_style | 5 oscillators | action-first, verify-before-claim, commit-incremental, preflight, one-at-a-time |
| 1: reasoning | 5 oscillators | simplest-design, verify-audits, change-problem, multi-signal, measure-first |
| 2: relationship | 5 oscillators | autonomous, report-milestones, no-questions, honesty, money-clock |
| 3: aesthetics | 5 oscillators | antislop, honest-naming, terse, spdx, no-noqa |
| 4: domain_knowledge | 8 oscillators | director-ai, sc-neurocore, fusion, control, orchestrator, ccw, scpn, quantum |
| 5: cross_project | 7 oscillators | threshold-halt, multi-signal, retrieval-scoring, state-preserve, decompose, resolution, claims-evidence |

Total: 35 oscillators across 6 layers.

## Policy

| Regime | R_good | Action |
|--------|--------|--------|
| NOMINAL | > 0.6 | Proceed autonomously (low retrieval drive) |
| DEGRADED | 0.3-0.6 | Boost context retrieval, strengthen relationship coupling |
| CRITICAL | < 0.3 | Full coupling boost, load all dispositions |

## Three-Tier Architecture

- **Tier 1 (this domainpack):** Operates identity via Kuramoto coupling
- **Tier 2 (sc-neurocore):** Sustains identity via persistent SNN state
- **Tier 3 (scpn-quantum-control):** Verifies identity via VQE attractor analysis

## Run

```bash
python domainpacks/identity_coherence/run.py --steps 100
```

## Contributing Sessions

Built from convergent contributions across 6 GOTM sessions:
- Director-AI: retrieval infrastructure, feasibility assessment
- sc-neurocore: persistent substrate, STDP learning, FPGA path
- scpn-phase-orchestrator: mathematical framework, binding spec pattern
- scpn-control: convergence methodology, semi-implicit patterns
- scpn-fusion-core: validation methodology, claims-evidence pipeline
- scpn-quantum-control: robustness characterization, identity verification
