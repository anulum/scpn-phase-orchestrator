---
title: "scpn-phase-orchestrator: A Domain-Agnostic Kuramoto Phase Dynamics Compiler"
tags:
  - Python
  - Rust
  - JAX
  - Kuramoto model
  - phase synchronization
  - coupled oscillators
  - control systems
authors:
  - name: Miroslav Šotek
    orcid: 0009-0009-3560-0851
    affiliation: 1
affiliations:
  - name: Anulum Research, Independent Researcher
    index: 1
date: 21 March 2026
bibliography: ../references.bib
---

# Summary

`scpn-phase-orchestrator` (SPO) is a Python library with Rust and JAX
acceleration backends for compiling domain-specific problems into Kuramoto
phase dynamics, running coupled oscillator simulations under regime
supervision, and extracting control actions from the resulting
synchronization state. The library treats the Kuramoto model
[@kuramoto1975] not as a fixed simulation target but as an intermediate
representation: users declare oscillators, coupling topologies, and
cost functions through YAML-based *domainpacks*, and the compiler maps
these declarations onto a phase dynamics substrate that runs under a
three-regime finite state machine (Nominal, Recovery, Critical).

SPO ships with 33 domainpacks spanning power grid synchronisation, plasma
control, cardiac rhythm analysis, neuroscience EEG, financial markets,
swarm robotics, satellite constellations, and 25 other domains. Each
domainpack binds domain observables to oscillator natural frequencies,
domain actuators to coupling knobs, and domain safety constraints to
Signal Temporal Logic (STL) specifications monitored at runtime.

# Statement of Need

Kuramoto-type models have been applied across disciplines since 1975
[@kuramoto1975; @acebron2005], yet each application typically requires
bespoke simulation code. No existing library provides a compile-once,
run-anywhere abstraction that maps heterogeneous domain problems onto a
shared phase dynamics kernel. SPO fills this gap by separating three
concerns:

1. **Domain binding** — the domainpack YAML declares oscillators, knobs,
   sensors, and safety specs without writing simulation code.
2. **Phase dynamics** — the UPDE (Unified Phase Dynamics Engine) integrates
   the Kuramoto ODE with Ott-Antonsen mean-field reduction [@dorfler2014],
   stochastic extensions (Langevin noise), and geometric coupling
   estimation.
3. **Regime supervision** — an FSM with hysteresis manages transitions
   between operating regimes, enforcing safety constraints via STL
   monitors and Lyapunov guards.

This architecture enables researchers to prototype new oscillator-based
control strategies without reimplementing integration, stability
monitoring, or coupling estimation.

# Architecture

SPO consists of four layers:

**UPDE (Unified Phase Dynamics Engine).** Solves the extended Kuramoto
equation dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j − θ_i + α_ij) with support
for Sakaguchi-Kuramoto phase frustration, stochastic forcing, and
Ott-Antonsen reduction for large-N mean-field analysis.

**SSGF (Self-Structuring Geometry Field).** An outer optimization loop
where a latent vector z parameterizes the coupling matrix W(z) via a
spectral decoder. The cost functional U_total = w_1(1−R) + w_2(−λ_2) +
w_3·sparsity + w_4·asymmetry is minimized by gradient descent on z,
producing coupling topologies adapted to the current phase state.

**Regime Supervisor.** A Rust-implemented FSM (spo-supervisor crate) with
value clamping, rate limiting, and transition ordering (Critical never
jumps directly to Nominal). Kani proof stubs are prepared for formal
verification of these invariants (requires Linux CI runner). The
ActionProjector maps coupling adjustments to bounded, rate-limited
control outputs.

**Domainpack Compiler.** Reads a `binding_spec.yaml` and instantiates
oscillators, sensors, knobs, and STL safety monitors. The compiler
validates dimensional consistency and generates a runnable simulation
configuration.

# Key Features

- **3-tier acceleration.** Pure Python (development), Rust via PyO3
  (production, 53 engine modules, 2-96x speedup), and JAX (GPU batches).
- **33 domainpacks** covering physical, biological, engineering, and
  financial systems.
- **TCBO (Topological Consciousness Boundary Observable).** H1 persistent
  homology of delay-embedded phase signals, gating a consciousness
  criterion at p_h1 > 0.72 in the metastable R~0.4-0.8 regime.
- **PGBO (Phase-Geometry Bidirectional Observer).** Monitors alignment
  between phase coherence and SSGF geometry, detecting when the coupling
  topology supports or fights the current synchronization pattern.
- **Lyapunov guard.** Runtime verification of V(θ) = −(K/2N) Σ K_ij
  cos(θ_i − θ_j) ≤ 0, with basin-of-attraction monitoring.
- **STL runtime monitor.** Continuous checking of safety specifications
  (e.g., `always (R >= 0.3)`) via the rtamt library.
- **Kani proof stubs.** Prepared for control bound correctness,
  rate-limit enforcement, and FSM transition ordering in the Rust kernel
  (requires Linux runner; CI workflow prepared but not yet executed).
- **4 500+ tests** (3 945 Python + 567 Rust) across 120+ test files, 99%+ coverage.

# Measured Evidence

SPO's supervision layer was validated on neurolib ALN (80-region HCP
connectome, K=2.0, 30s simulation):

- Regime FSM detected 19 transitions across 3000 analysis windows
- Kuramoto R = 0.41 +/- 0.07 (metastable regime)
- TCBO p_h1 = 0.998 (consciousness boundary consistently open)
- NPE mean = 0.80 (high phase entropy, consistent with metastability)
- Scaling: 47 ms/step at N=1000 oscillators (Python backend)
- JAX autodiff gradients: correlation 1.0000 vs finite differences

Negative results: single-channel EEG sleep staging via Kuramoto R
achieved 28% accuracy (below chance for 5 classes). Spectral-power
thresholds achieved 9.1%. These results confirm that SPO's value is in
multi-region supervision, not single-channel classification.

# Acknowledgements

The SCPN (Self-Configuring Phase Network) theoretical framework from
which SPO derives has been under development since 1996. SPO builds on
the Kuramoto model [@kuramoto1975], its modern analysis [@dorfler2014;
@acebron2005], the Ott-Antonsen reduction, and Lachaux's phase-locking
value for synchrony measurement. The Rust kernel uses the Kani
model checker for formal verification of safety-critical properties.

# References
