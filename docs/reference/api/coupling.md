# Coupling

The coupling subsystem builds, adapts, and analyzes the inter-oscillator
coupling matrix K_nm — the central object in Kuramoto dynamics. K_ij
determines how strongly oscillator j pulls oscillator i toward synchrony.

## K_nm Construction

Builds coupling matrices from binding spec parameters: base coupling,
distance decay, boost/penalty on selected edges, geometry constraints.
Optional Rust acceleration via `spo_kernel.PyCouplingBuilder`.

::: scpn_phase_orchestrator.coupling.knm

## Geometry Constraints

Enforces spatial coupling constraints: nearest-neighbor graphs, distance
decay, hierarchical grouping. Ensures K_nm respects the physical geometry
of the domain (e.g., cortical distance for brain networks, line length
for power grids).

::: scpn_phase_orchestrator.coupling.geometry_constraints

## Phase Lag Estimation

Estimates inter-oscillator phase lags α_ij from observed time series.
Lags represent transport delays, actuator latencies, or propagation times.

::: scpn_phase_orchestrator.coupling.lags

## Coupling Templates

Pre-configured coupling patterns: all-to-all, ring, chain, star, random
Erdős-Rényi, small-world Watts-Strogatz. Used by domainpacks as
starting points.

::: scpn_phase_orchestrator.coupling.templates

## Hodge Decomposition

Decomposes the coupling matrix K into three orthogonal components via
Hodge theory (Jiang et al. 2011):

- **Gradient:** conservative phase-locking flow (symmetric part of K).
  Oscillators are pulled toward a potential minimum.
- **Curl:** rotational circulation flow (antisymmetric part).
  Oscillators cycle through phase relationships without converging.
- **Harmonic:** topological residual living in the null space of the
  Hodge Laplacian. Invariant under dynamics — a topological invariant
  of the coupling geometry.

Answers: "Is this synchronization conservative or rotational?"
The harmonic component is particularly significant for the SCPN
consciousness model (topologically protected identity invariant).

::: scpn_phase_orchestrator.coupling.hodge

## Three-Factor Hebbian Plasticity

Coupling adaptation rule inspired by biological synaptic plasticity:

```
ΔK_ij = lr × eligibility_ij × modulator × phase_gate
```

1. **Eligibility:** cos(θ_j - θ_i) — pairwise Hebbian trace
   (in-phase → strengthen, anti-phase → weaken)
2. **Modulator:** scalar neuromodulatory signal from the L16 director
   layer (dopamine/serotonin analog)
3. **Phase gate:** Boolean from the TCBO consciousness boundary
   (only update coupling when the system is in a conscious-like regime)

Grounded in Friston 2005 on free energy and synaptic plasticity.

::: scpn_phase_orchestrator.coupling.plasticity

## Transfer Entropy Adaptive Coupling

Adapts coupling strength based on directed causal information flow:

```
K_ij(t+1) = (1 - decay) · K_ij(t) + lr · TE(i → j)
```

Unlike symmetric Hebbian learning, transfer entropy breaks symmetry
to detect causal direction. Oscillators that causally influence each
other get stronger coupling; uncorrelated pairs decay toward zero.

Lizier 2012, "Local Information Transfer as Spatiotemporal Filter."

::: scpn_phase_orchestrator.coupling.te_adaptive

## HCP Connectome Generator

Generates neuroscience-realistic coupling matrices inspired by the Human
Connectome Project structural connectivity:

- Intra-hemispheric: exponential distance decay
- Inter-hemispheric: corpus callosum pattern (homotopic connections)
- Default Mode Network: hub structure with elevated coupling
- Optional: real HCP data via neurolib bridge

::: scpn_phase_orchestrator.coupling.connectome
