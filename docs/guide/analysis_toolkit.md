# Analysis Toolkit

## Purpose and scope

The toolkit is structured for operational readability: each monitor returns a
different failure or regime signal before the scalar order parameter alone would
show anything unusual. In practice, teams use this page as a first-pass
selection guide, then tune thresholds against their domain trajectories.

## How operators should use this layer

Treat the toolkit as a **multi-signal diagnostic funnel** rather than a single
alarm source:

- start with fast, broad monitors (`order_parameter`, `lyapunov`);
- add structural monitors (`plv`, `chimera`, `winding`) when patterns localise;
- then confirm causal or thermodynamic interpretations (`coupling_est`, `entropy_prod`, `itpc`, `pid`).

This sequencing reduces false positives and gives policy teams a reproducible rationale
for each escalation before any actuation change is promoted.

The sections below are ordered from global coherence to higher-order coupling
relationships because that mirrors a typical diagnostic flow:
global stability -> phase alignment structure -> causality and stability risk ->
topological and thermodynamic drift.

SPO provides 12 dynamical monitors — most oscillator simulators have 1-2.
Each monitor detects a different aspect of the dynamics that scalar R misses.

## Selecting a minimal monitor set

For a first production run, teams usually start with:

- `order_parameter` for synchronization trend and collapse detection,
- `lyapunov` for local stability margin,
- `plv` for pairwise synchrony topology,
- one supervisory metric (`evs` or `pid`) for domain-facing interpretability.

Expanding to all monitors is recommended only after a baseline is stable; this
keeps false alarm fatigue manageable while keeping observability depth.

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
    options:
      show_bases: false
      show_source: false
      members: false

## Entrainment Verification Score (EVS)

Three-criterion battery for rigorous entrainment validation:

1. ITPC (inter-trial phase coherence) persistence
2. Survival during stimulus pause
3. Frequency specificity (ratio at target vs control frequency)

Distinguishes true entrainment from broadband phase-locking artifacts.

::: scpn_phase_orchestrator.monitor.evs
    options:
      show_root_heading: false
      members: false

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

## Read with operational intent

Most monitors are most useful when compared over time and context, not as single
point alarms. A practical dashboard should show short-term and rolling-window
views side by side, then correlate alarms with known interventions.

The intended use is:

- detect onset conditions with one or two fast indicators,
- confirm with a slower structural monitor,
- only then trigger policy or supervisory changes.

## Synthetic HCP Connectome Generation

Generates neuroscience-realistic coupling matrices inspired by the
Human Connectome Project:

- Intra-hemispheric exponential distance decay
- Inter-hemispheric corpus callosum pattern
- Default Mode Network hub structure

::: scpn_phase_orchestrator.coupling.connectome

## Monitoring stack as a decision chain

Treat this page as a decision chain for escalation, not a list of separate tools.
The intended order is:

1. start with one global stability indicator,
2. confirm structural coherence with pairwise and topology-aware indicators,
3. apply causal or energetic checks before any bounded actuation proposal.

The sequence is designed to reduce false positives and preserve audit quality.

## Minimal observability profile

A practical minimum profile for a first production pilot is:

- `order_parameter` for baseline synchrony,
- `lyapunov` for local stability trend,
- one causal or directional metric (`coupling_est` or `itpc`),
- one action governance metric (`evs` or `pid`).

This gives enough signal to decide whether a policy should stay static,
reduce its scope, or escalate to broader review.
