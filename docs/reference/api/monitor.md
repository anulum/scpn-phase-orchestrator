# Monitor

The monitor subsystem provides 12 dynamical observers that detect different
aspects of oscillator network behavior. Most oscillator simulators provide
only the global order parameter R. SPO's monitors detect chimera states,
cross-frequency coupling, causal information flow, topological invariants,
and thermodynamic irreversibility — phenomena that R alone cannot capture.

## Boundary Observer

Detects when oscillator dynamics violate configured safety/performance
boundaries. Fires alerts when R drops below `R_good` threshold or
exceeds `R_bad` threshold. Used by the supervisor to trigger regime
transitions.

::: scpn_phase_orchestrator.monitor.boundaries

## Coherence Monitor

Tracks the Kuramoto order parameter R over time with configurable
thresholds for phase-lock detection. Provides `R_good` (target coherence)
and `R_bad` (harmful mode-locking) as dual objectives.

::: scpn_phase_orchestrator.monitor.coherence

## Session Start Gate

Verifies that the oscillator network reaches a minimum coherence
threshold before the main control loop engages. Prevents the supervisor
from acting on transient startup dynamics.

::: scpn_phase_orchestrator.monitor.session_start

## Chimera State Detection

Detects chimera states: the coexistence of coherent (phase-locked) and
incoherent (desynchronized) clusters within the same network. This is a
fundamentally different phenomenon from uniform synchronization or
uniform incoherence — it requires spatially resolved analysis.

**Theory:** Kuramoto & Battogtokh 2002 discovered that identical
oscillators with identical coupling can spontaneously split into
synchronized and desynchronized subpopulations. This was later confirmed
experimentally in chemical oscillators and electronic circuits.

**Algorithm:**

1. Compute local order parameter R_i for each oscillator based on its
   coupled neighbors (oscillators j where K_ij > 0)
2. Classify: R_i > 0.7 → coherent, R_i < 0.3 → incoherent
3. Chimera index = fraction of oscillators in the boundary region

**Usage:**

```python
from scpn_phase_orchestrator.monitor.chimera import detect_chimera

state = detect_chimera(phases, knm)
# state.coherent_indices: list of phase-locked oscillators
# state.incoherent_indices: list of desynchronized oscillators
# state.chimera_index: 0.0 = pure state, >0 = chimera
```

::: scpn_phase_orchestrator.monitor.chimera

## Entrainment Verification Score (EVS)

Three-criterion battery that distinguishes genuine entrainment
(phase-locking to a stimulus) from broadband artifacts. All three
criteria must pass for `is_entrained=True`:

1. **ITPC persistence:** Mean inter-trial phase coherence across time
   points must exceed threshold (default 0.5)
2. **Survival during pause:** ITPC must remain elevated after the
   stimulus stops, proving the oscillator was entrained (not just
   responding reactively)
3. **Frequency specificity:** ITPC at the target frequency divided by
   ITPC at a control frequency must exceed threshold (default 2.0),
   proving the locking is frequency-specific

**Usage:**

```python
from scpn_phase_orchestrator.monitor.evs import EVSMonitor

monitor = EVSMonitor(
    itpc_threshold=0.5,
    persistence_threshold=0.3,
    specificity_threshold=2.0,
)
result = monitor.evaluate(
    phases_trials,        # (n_trials, T) phase matrix
    pause_start=500,      # timestep where stimulus pauses
    control_frequency=2,  # index of control frequency band
)
# result.is_entrained: bool
# result.itpc_value, result.persistence_score, result.specificity_ratio
```

::: scpn_phase_orchestrator.monitor.evs

## Partial Information Decomposition (PID)

Decomposes mutual information between two oscillator groups and a
target variable into **redundancy** (information both groups share)
and **synergy** (information available only from the joint observation).

**Theory:** Williams & Beer 2010 (arXiv:1004.2515). Circular mutual
information is estimated via binned phase histograms (default 32 bins).

**Usage:**

```python
from scpn_phase_orchestrator.monitor.pid import redundancy, synergy

# phases_a, phases_b: (T,) phase time series from two groups
# phases_target: (T,) target phase series
R = redundancy(phases_a, phases_b, phases_target)  # shared info
S = synergy(phases_a, phases_b, phases_target)      # joint-only info
```

High synergy means the groups carry complementary information — neither
alone predicts the target, but together they do. This detects
higher-order functional relationships invisible to pairwise PLV.

::: scpn_phase_orchestrator.monitor.pid

## Lyapunov Exponent

Real-time estimation of the maximal Lyapunov exponent from phase
trajectories. The Lyapunov exponent characterizes the system's
sensitivity to initial conditions:

- λ > 0: chaotic (exponential divergence of nearby trajectories)
- λ ≈ 0: edge of chaos (critical regime, maximal computational capacity)
- λ < 0: stable attractor (perturbations decay exponentially)

The "edge of chaos" (λ ≈ 0) is where consciousness-like dynamics
operate (PNAS 2022) and where reservoir computing achieves optimal
performance (arXiv:2407.16172).

::: scpn_phase_orchestrator.monitor.lyapunov

## Entropy Production Rate

Measures the thermodynamic irreversibility of the phase dynamics.
Higher entropy production means the system is further from
equilibrium — it is actively dissipating energy to maintain its
current synchronization state.

**Theory:** For Kuramoto dynamics, entropy production rate is
proportional to the mean squared coupling torque. A system at
thermal equilibrium (detailed balance) has zero entropy production;
a synchronized Kuramoto network actively maintained by coupling
has positive entropy production.

::: scpn_phase_orchestrator.monitor.entropy_prod

## Winding Number

Topological invariant counting how many times the phase wraps around
the circle [0, 2π) over a time window. The winding number is an
integer-valued quantity that is robust to noise and small perturbations.

**Usage:**

```python
from scpn_phase_orchestrator.monitor.winding import winding_numbers

# phases_history: (T, N) phase trajectory
w = winding_numbers(phases_history)  # (N,) integer winding numbers
```

Different winding numbers for different oscillators indicate frequency
differences; a sudden change in winding number signals a phase slip
(loss of synchronization with a specific partner).

::: scpn_phase_orchestrator.monitor.winding

## Inter-Trial Phase Coherence (ITPC)

Standard neuroscience measure of phase consistency across repeated
trials or time windows. ITPC = |mean(exp(i*theta))| computed across
trials at each time point.

ITPC = 1: perfect phase alignment across trials (stimulus-locked).
ITPC ≈ 0: random phase relationship (no consistent response).

Used by the EVS monitor as one of three entrainment criteria.

::: scpn_phase_orchestrator.monitor.itpc

## Phase Transfer Entropy

Directed information-theoretic measure of causal influence between
oscillators. Transfer entropy TE(i→j) quantifies how much the past
of oscillator i reduces uncertainty about the future of oscillator j,
beyond what j's own past provides.

**Key property:** Unlike PLV (symmetric), transfer entropy is
**directional** — TE(i→j) ≠ TE(j→i) in general. This detects
causal coupling direction, not just correlation.

Used by the `te_adaptive` coupling module to adapt K_ij based on
measured causal information flow (Lizier 2012).

::: scpn_phase_orchestrator.monitor.transfer_entropy
