# SCPN Phase Orchestrator: Fact-Based Overview

This page states, plainly, what SPO is, what has been checked against an
independent ground truth, and what has not. It deliberately avoids superlatives
and does not describe review-only or unvalidated surfaces as if they were
deployed capabilities.

## 1. What SPO is

SPO is a Python library and CLI for analysing **coupled-oscillator systems**. It
turns the repeating signals a system produces into phase variables built on
Kuramoto / UPDE dynamics, and provides detectors, a regime supervisor, and a
**review-only** control-proposal surface. It is a research and evaluation toolkit,
not a deployed controller and not a validated general early-warning system.

At its core is the Universal Phase Dynamics Equation (UPDE):

$$\dot{\theta}_i = \omega_i + \sum_j K_{ij}\,\sin(\theta_j - \theta_i - \alpha_{ij}) + \zeta\,\sin(\Psi - \theta_i)$$

## 2. Evidence status (validated vs not)

**Externally validated against an independent reference:**

- **Grid modal damping estimation.** On the IEEE-39 and Kundur systems, the
  estimated growth rate of the dominant electromechanical mode matches the
  small-signal eigenvalue from the ANDES simulator
  ([study §3.9](studies/early_warning_matched_false_alarm.md)).
- **The eigenvalue regime map.** Across five systems (fold, pitchfork, Hopf, and
  the unimodal and bimodal Kuramoto transitions) the shipped detectors recover the
  analytic eigenvalue's real part; the correct estimator is regime-dependent
  (study §3.10–3.14).
- **Honest, false-alarm-controlled evaluation.** A matched-false-alarm operating
  point, a permutation significance test, and a hash-sealed evidence record
  (study §2).

**Empirically at chance on real data:** across five real modalities (grid, EEG,
ecological/climate, molecular), generic early-warning detection at an honest
operating point is at chance under a permutation-controlled test (study §3.1–3.8),
consistent with the wider literature. **SPO does not claim to predict tipping
points in these domains.** The one place a detector clears the bar is the grid.

**Not yet validated:** closed-loop control, hardware / PLC / FPGA / quantum /
neuromorphic actuation, BCI feedback, and distributed-mesh coupling are
adapter-scoped, review-only, and carry no field evidence.

## 3. Components that exist in the codebase

These modules are present and covered by tests. Presence is not a performance or
deployment claim; several are optional or review-only.

| Area | Module | What it does |
|------|--------|--------------|
| Dynamics | `upde/` | UPDE / Kuramoto integration, bifurcation and Ott–Antonsen reduction |
| Differentiable | `nn/` (JAX/equinox) | phase-oscillator layers, order parameter, coupling inference; can run on GPU/TPU |
| Detection | `monitor/` | critical-slowing-down indicators and grid modal growth-rate estimation |
| Supervision | `supervisor/` | regime classification (nominal / degraded / critical) and a Petri-net state machine |
| Control (review-only) | policy surfaces | bounded, rate-limited control *proposals* with replay evidence — not actuation |
| Assurance | `assurance/` | hash-sealed evidence records and canonical-record hashing |
| Acceleration | `spo-kernel` (Rust FFI) | integration kernels with parity tests against the Python path |
| Adapters (optional, review-only) | `adapters/` | OPC-UA / MQTT ingestion, LSL, mesh, and simulator bridges — scaffolds, not validated integrations |

## 4. Performance claims

Latency and throughput are environment-specific and are **not** quoted here as
static facts. Where a number is needed, reproduce it on the target hardware with
the committed benchmark scripts and record the run metadata (platform, Python
version, backend, lockfile). The canonical benchmark snapshots under `benchmarks/`
and the CI benchmark gates are the reference; treat any figure without attached
run metadata as unverified.

## 5. Where SPO fits

SPO is between an academic research toolkit and a production monitoring system. Its
one externally-validated niche is grid modal-damping estimation checked against
eigenvalues; its most transferable asset is the honest, false-alarm-controlled
evaluation methodology, which applies to any early-warning claim. It is a
complement to ground-truth tools such as ANDES (which it validates against), not a
replacement for deployed control-room monitoring products.

For related material, see the [study](studies/early_warning_matched_false_alarm.md)
and the architecture and product-boundary pages.
