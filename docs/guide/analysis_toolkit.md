# Analysis Toolkit

SPO provides 12 dynamical monitors — most oscillator simulators have 1-2.
Each monitor detects a different aspect of the dynamics that scalar R misses.

## Order Parameter & PLV

Standard Kuramoto order parameter R = |⟨exp(iθ)⟩| and Phase-Locking
Value matrix PLV_ij = |⟨exp(i(θ_i - θ_j))⟩_t|.

::: scpn_phase_orchestrator.upde.order_params

## Phase-Amplitude Coupling (PAC)

Modulation index (MI) via Tort et al. 2010. Bins low-frequency phase,
computes mean amplitude per bin, KL divergence from uniform. N×N PAC
matrix: entry [i,j] = MI(phase_i, amplitude_j).

Central to neuroscience — cross-frequency coupling between brain
oscillation bands (theta-gamma, alpha-beta).

::: scpn_phase_orchestrator.upde.pac

## Chimera State Detection

Detects chimera states: coexisting coherent and incoherent clusters
within the same network. Uses local order parameter R_i based on
neighborhood coupling.

- Coherent: R_i > 0.7
- Incoherent: R_i < 0.3
- Boundary: in-between
- Chimera index = boundary_count / N

Detects phase transitions that global R misses.

::: scpn_phase_orchestrator.monitor.chimera

## Entrainment Verification Score (EVS)

Three-criterion battery for rigorous entrainment validation:

1. ITPC (inter-trial phase coherence) persistence
2. Survival during stimulus pause
3. Frequency specificity (ratio at target vs control frequency)

Distinguishes true entrainment from broadband phase-locking artifacts.

::: scpn_phase_orchestrator.monitor.evs

## Partial Information Decomposition (PID)

Decomposes mutual information into:

- **Redundancy**: shared information from both oscillator groups
- **Synergy**: information present only in the joint group

Detects when groups carry synergistic (non-redundant) information
about global phase (Williams & Beer 2010).

::: scpn_phase_orchestrator.monitor.pid

## Lyapunov Exponent

Real-time estimation of the maximal Lyapunov exponent. Positive =
chaos, zero = edge of chaos (critical), negative = stable attractor.

::: scpn_phase_orchestrator.monitor.lyapunov

## Entropy Production

Measures thermodynamic irreversibility of the phase dynamics.
Higher entropy production = system further from equilibrium.

::: scpn_phase_orchestrator.monitor.entropy_prod

## Winding Number

Topological charge of phase trajectories. Counts how many times
the phase wraps around the circle. Integer-valued topological
invariant.

::: scpn_phase_orchestrator.monitor.winding

## Inter-Trial Phase Coherence (ITPC)

Phase consistency across repeated trials or time windows.
Standard neuroscience measure for event-related phase locking.

::: scpn_phase_orchestrator.monitor.itpc

## Coupling Estimation from Data

Two methods for inferring coupling from observed time series:

1. **Basic**: least-squares fit of dθ/dt - ω = Σ K_ij sin(θ_j - θ_i)
2. **Harmonics**: higher Fourier harmonics for non-sinusoidal coupling

The harmonics method captures real biological coupling shapes
(Stankovski 2017).

::: scpn_phase_orchestrator.autotune.coupling_est

## Synthetic HCP Connectome Generation

Generates neuroscience-realistic coupling matrices inspired by the
Human Connectome Project:

- Intra-hemispheric exponential distance decay
- Inter-hemispheric corpus callosum pattern
- Default Mode Network hub structure

::: scpn_phase_orchestrator.coupling.connectome
