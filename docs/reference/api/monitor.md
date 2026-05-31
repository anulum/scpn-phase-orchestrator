# Monitor

The monitor subsystem provides 15+ dynamical observers that detect different
aspects of oscillator network behavior. Most oscillator simulators provide
only the global order parameter R. SPO's monitors detect chimera states,
cross-frequency coupling, causal information flow, topological invariants,
and thermodynamic irreversibility — phenomena that R alone cannot capture.

Detailed module references:

- [Delay embedding and phase-space reconstruction](monitor_embedding.md)

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

## Signal Temporal Logic Runtime Verification

`STLMonitor` evaluates runtime safety formulas over scalar monitor traces.
It uses `rtamt` when available for full STL syntax and includes a builtin
robustness evaluator for common safety forms:

- `always (R >= 0.3)`
- `eventually (R >= 0.8)`
- `always (R >= 0.85 and amplitude_spread < 0.2)`

Positive robustness means the formula is satisfied; negative robustness
means violated. `evaluate_result()` returns an audit-ready result with
the formula, robustness, satisfaction boolean, and backend name.

```python
from scpn_phase_orchestrator.monitor.stl import STLMonitor

monitor = STLMonitor("always (R >= 0.3)")
result = monitor.evaluate_result({"R": [0.9, 0.8, 0.6]})
assert result.satisfied
```

`synthesise_stl_monitoring_automaton()` converts supported builtin formulas
into an audit-ready runtime automaton. The automaton records the state
sequence, trace-indexed transitions, first violation or satisfaction index,
pointwise robustness margins, and final satisfaction result.

```python
from scpn_phase_orchestrator.monitor.stl import (
    synthesise_stl_monitoring_automaton,
)

automaton = synthesise_stl_monitoring_automaton(
    "always (R >= 0.3)",
    {"R": [0.9, 0.2, 0.6]},
)
audit_payload = automaton.to_audit_record()
assert audit_payload["states"][1]["first_hit_index"] == 1
```

Policy YAML integration is available through `load_policy_stl_specs()`,
`evaluate_policy_stl_specs()`, and `synthesise_policy_stl_automata()` in
`scpn_phase_orchestrator.supervisor.policy_rules`. This keeps STL
specification loading in the policy DSL while preserving `STLMonitor` and the
automata synthesizer as runtime evaluators.

`synthesise_stl_controller_candidates()` adds the first controller-synthesis
linkage. It consumes a builtin STL automaton plus the same trace and emits
non-actuating signal-level candidates for the weakest violated predicate.
The result is an audit/review artefact only: `actuating` is always `False`, and
callers must still pass any candidate through policy, projection, safety, and
actuation gates.

```python
from scpn_phase_orchestrator.monitor.stl import (
    synthesise_stl_controller_candidates,
)

synthesis = synthesise_stl_controller_candidates(
    automaton,
    {"R": [0.9, 0.2, 0.6]},
    action_map={"R": "raise_coupling"},
)
audit_payload = synthesis.to_audit_record()
assert audit_payload["actuating"] is False
```

`project_stl_controller_candidates()` then maps those candidates through
explicit policy-approved projection templates and the standard
`ActionProjector`. It still returns a review plan only: `actuating` remains
`False`, unmapped candidates are rejected with reasons, and the approved
entries are bounded `ControlAction` proposals rather than applied commands.

```python
from scpn_phase_orchestrator.monitor.stl import (
    STLActionProjectionTemplate,
    project_stl_controller_candidates,
)

plan = project_stl_controller_candidates(
    synthesis,
    (
        STLActionProjectionTemplate(
            action="raise_coupling",
            knob="K",
            scope="global",
            base_value=0.9,
            step=10.0,
            ttl_s=0.5,
            previous_value=0.9,
            value_bounds=(0.0, 1.0),
            rate_limit=0.05,
        ),
    ),
)
assert plan.to_audit_record()["actuating"] is False
```

`synthesise_stl_closed_loop_plan()` now also records a
`runtime_actuation_gate` audit section. The gate routes projected
`ControlAction` proposals through `ActuationMapper` using the same explicit
projection templates, records deterministic actuator-command evidence, and
keeps `non_actuating` plus `execution_disabled` true. This is the intended use
case for STL closed-loop planning: prove that a violated safety formula can be
translated into bounded, mapper-valid runtime actions for operator review
without enabling live actuation.

```python
from scpn_phase_orchestrator.monitor.stl import (
    synthesise_stl_closed_loop_plan,
)

closed_loop_plan = synthesise_stl_closed_loop_plan(
    automaton,
    {"R": [0.1, 0.2, 0.75]},
    (projection_template,),
    horizon_steps=4,
    action_map={"R": "raise_coupling"},
)
gate = closed_loop_plan.to_audit_record()["runtime_actuation_gate"]
assert gate["execution_disabled"] is True
```

::: scpn_phase_orchestrator.monitor.stl

## Chimera State Detection

Detects chimera states: the coexistence of coherent (phase-locked) and
incoherent (desynchronised) clusters within the same network. This is a
fundamentally different phenomenon from uniform synchronization or
uniform incoherence — it requires spatially resolved analysis.

**Theory:** Kuramoto & Battogtokh 2002 discovered that identical
oscillators with identical coupling can spontaneously split into
synchronised and desynchronised subpopulations. This was later confirmed
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
# state.incoherent_indices: list of desynchronised oscillators
# state.chimera_index: 0.0 = pure state, >0 = chimera
```

::: scpn_phase_orchestrator.monitor.chimera

## Entrainment Verification Score (EVS)

**Detailed documentation:** [EVS (Entrainment) — detailed reference](monitor_evs.md)

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
    options:
      members:
        - EVSMonitor

## Partial Information Decomposition (PID)

Decomposes mutual information between two oscillator groups and a
global phase reference into **redundancy** (information both groups share)
and **synergy** (information available only from the joint observation).

**Theory:** Williams & Beer 2010 (arXiv:1004.2515). Circular mutual
information is estimated via binned phase histograms (default 32 bins).
Inputs, group indices, and optional Rust scalar outputs are validated as
finite real-valued quantities; boolean aliases and complex payloads are
rejected before histogram estimation or backend acceptance.

**Usage:**

```python
from scpn_phase_orchestrator.monitor.pid import redundancy, synergy

# phases: one phase vector; groups are oscillator index sets into that vector
R = redundancy(phases, group_a=[0, 1, 2], group_b=[3, 4, 5])
S = synergy(phases, group_a=[0, 1, 2], group_b=[3, 4, 5])
```

::: scpn_phase_orchestrator.monitor.npe

High synergy means the groups carry complementary information — neither
alone predicts the target, but together they do. This detects
higher-order functional relationships invisible to pairwise PLV.

::: scpn_phase_orchestrator.monitor.pid

## Integrated-Information Monitor

Estimates an approximate Phi-style global integration metric from
phase trajectories. The monitor builds a pairwise circular
mutual-information matrix, evaluates unique bipartitions, and reports
the minimum cross-partition information as `phi`.

This is an engineering proxy for comparing regime traces and writing
audit records. It is not an exact IIT quantity and is not a
consciousness claim.
Phase-series inputs, bin/sample counts, audit scalars, partitions, and
pairwise mutual-information matrices are validated as finite real-valued
contracts. Boolean aliases and complex matrices are rejected before circular
histogram estimation or audit-record acceptance.

**Usage:**

```python
from scpn_phase_orchestrator.monitor import (
    benchmark_integrated_information_approximations,
    integrated_information,
)

# phase_series: (n_oscillators, n_samples)
result = integrated_information(phase_series, n_bins=16)
record = result.to_audit_record()

benchmark = benchmark_integrated_information_approximations()
benchmark_record = benchmark.to_audit_record()
```

`benchmark_integrated_information_approximations()` runs deterministic
synthetic calibration cases for independent, modular, phase-lagged chain, noisy
locked, and globally locked phase regimes. It is a numerical approximation
benchmark, not a hardware performance benchmark; the audit record documents
ordering margins and preserves the same engineering-proxy claim boundary.

::: scpn_phase_orchestrator.monitor.information_integration

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
a synchronised Kuramoto network actively maintained by coupling
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

## Recurrence Quantification Analysis (RQA)

Extracts dynamical invariants from phase trajectories via recurrence
plots. RQA is powerful because it works on short, non-stationary time
series where spectral methods fail.

**Eight measures:**

| Measure | Symbol | Meaning |
|---------|--------|---------|
| Recurrence rate | RR | Density of recurrence points |
| Determinism | DET | Fraction forming diagonal lines → deterministic dynamics |
| Average diagonal | L | Mean diagonal line length → prediction horizon |
| Max diagonal | L_max | Inversely related to max Lyapunov exponent |
| Diagonal entropy | ENTR | Complexity of deterministic structure |
| Laminarity | LAM | Fraction forming vertical lines → laminar states |
| Trapping time | TT | Mean time in laminar state |
| Max vertical | V_max | Longest laminar episode |

**Cross-RQA** extends this to detect synchronization between two
oscillator groups by computing the cross-recurrence matrix.

**Usage:**

```python
from scpn_phase_orchestrator.monitor.recurrence import rqa, cross_rqa

# Auto-RQA on a single trajectory
result = rqa(trajectory, epsilon=0.3, metric="angular")
print(f"DET={result.determinism:.3f}, LAM={result.laminarity:.3f}")

# Cross-RQA between two oscillator groups
cr = cross_rqa(traj_a, traj_b, epsilon=0.3)
print(f"Cross-DET={cr.determinism:.3f}")
```

**References:** Eckmann, Kamphorst & Ruelle 1987; Zbilut & Webber 1992;
Marwan et al. 2007, Phys. Reports 438:237-329.

::: scpn_phase_orchestrator.monitor.recurrence

## Delay Embedding (Attractor Reconstruction)

Reconstructs the full state-space attractor from a scalar observable
using Takens' embedding theorem. This is the prerequisite for computing
correlation dimension, Lyapunov exponents from scalar data, and
recurrence analysis on scalar measurements.

**Three-step procedure:**

1. **Optimal delay τ** via first minimum of average mutual information
   (Fraser & Swinney 1986)
2. **Optimal dimension m** via False Nearest Neighbors
   (Kennel, Brown & Abarbanel 1992)
3. **Embedding** constructs vectors v(t) = [x(t), x(t-τ), ..., x(t-(m-1)τ)]

Inputs and backend outputs are validated as finite real-valued arrays.
Boolean aliases and complex samples are rejected before the
Rust/Mojo/Julia/Go backend chain because Takens delay coordinates,
Fraser-Swinney mutual information, and false-nearest-neighbour
distances are defined over real scalar observations.
The Mojo subprocess adapter also validates raw stdout cardinality for
delay-coordinate rows, mutual-information scalars, and nearest-neighbour
distance/index pairs before numeric parsing, so blank-line insertion or
missing rows cannot be normalised into a plausible embedding payload.

**Usage:**

```python
from scpn_phase_orchestrator.monitor.embedding import auto_embed

# Automatic: determines τ and m, then embeds
result = auto_embed(signal)
print(f"τ={result.delay}, m={result.dimension}")
trajectory = result.trajectory  # (T', m) array

# Manual control
from scpn_phase_orchestrator.monitor.embedding import (
    optimal_delay, optimal_dimension, delay_embed,
)
tau = optimal_delay(signal, max_lag=100)
m = optimal_dimension(signal, delay=tau, max_dim=10)
embedded = delay_embed(signal, delay=tau, dimension=m)
```

**References:** Takens 1981, Lecture Notes in Mathematics 898:366-381.

::: scpn_phase_orchestrator.monitor.embedding

## Psychedelic State Metrics

The psychedelic monitor is a research diagnostic for phase-dispersion
simulation inspired by entropic-brain hypotheses. Public Python calls and
Go/Julia/Mojo entropy adapters reject boolean aliases, complex phases,
object arrays carrying Python or NumPy complex scalar aliases, non-finite
phases, invalid bin counts, complex entropy payloads, and invalid
coupling-reduction backend matrices before results are accepted. This preserves
the circular Shannon entropy and Kuramoto coupling semantics over real-valued
phase observations; it is not a clinical, dosage, or actuation interface.

Direct accelerator boundary contract: Go, Julia, and Mojo entropy adapters use
one shared `float64` validation path before loading shared-library, Julia, or
subprocess runtimes. Empty phase samples return zero entropy without requiring
optional runtimes, matching the public Python fallback and preserving the
Shannon special case for an empty empirical distribution.
Direct backend entropy outputs are also revalidated as finite real scalars in
the physical interval `[0, log(n_bins)]`; malformed Mojo raw stdout line counts,
blank-line insertion, and non-scalar tokens are rejected before the value
reaches downstream monitor logic.

::: scpn_phase_orchestrator.monitor.psychedelic

## Fractal Dimension

Estimates the fractal dimension of attractors from embedded trajectories.
Two complementary measures:

**Correlation dimension D₂** (Grassberger & Procaccia 1983):
Counts the fraction of point pairs within distance ε, then extracts
the power-law exponent C(ε) ~ ε^D₂. The scaling region is
automatically identified as the range with most stable local slopes.

**Kaplan-Yorke dimension D_KY** (Kaplan & Yorke 1979):
Computed from the Lyapunov spectrum as D_KY = j + (Σᵢ₌₁ʲ λᵢ)/|λⱼ₊₁|
where j is the largest index with non-negative cumulative sum.
The Kaplan-Yorke conjecture equates D_KY to the information dimension.

**Usage:**

```python
from scpn_phase_orchestrator.monitor.dimension import (
    correlation_dimension, kaplan_yorke_dimension,
)

# From embedded trajectory
result = correlation_dimension(trajectory, n_epsilons=30)
print(f"D2={result.D2:.2f}, scaling={result.scaling_range}")

# From Lyapunov spectrum
from scpn_phase_orchestrator.monitor.lyapunov import lyapunov_spectrum
spec = lyapunov_spectrum(phases, omegas, knm, alpha)
D_KY = kaplan_yorke_dimension(spec)
print(f"D_KY={D_KY:.2f}")
```

**References:** Grassberger & Procaccia 1983, Phys. Rev. Lett. 50:346-349;
Kaplan & Yorke 1979, Lecture Notes in Mathematics 730:228-237.

::: scpn_phase_orchestrator.monitor.dimension

## Poincare Sections

Detects when a trajectory crosses a hyperplane, extracts the crossing
points (Poincare map), and computes return time statistics. Return time
regularity distinguishes periodic orbits (constant return time) from
chaotic ones (fluctuating return times).

Direct accelerator contracts require finite real crossing coordinates,
strictly increasing sampled crossing times, and Mojo text output with an
explicit crossing-count header plus exact raw-line cardinality.

**Detailed documentation:** [Poincare section monitor](monitor_poincare.md)

**Two interfaces:**

- `poincare_section()`: general hyperplane crossing for any state-space trajectory
- `phase_poincare()`: specialized for phase oscillators — detects when one
  oscillator crosses a reference phase value

**Usage:**

```python
from scpn_phase_orchestrator.monitor.poincare import (
    poincare_section, phase_poincare,
)

# General hyperplane section
result = poincare_section(trajectory, normal=[1, 0, 0])
print(f"Mean return time: {result.mean_return_time:.1f}")
print(f"Return time std: {result.std_return_time:.3f}")

# Phase-specific section
result = phase_poincare(phases, oscillator_idx=0, section_phase=0.0)
```

::: scpn_phase_orchestrator.monitor.poincare

## Sleep Stage Classifier

AASM sleep staging mapped to the Kuramoto order parameter R.
Classifies phases into Wake/N1/N2/N3/REM based on R thresholds
and a functional desynchronisation flag. Includes ultradian
(~90 min) cycle phase estimation.
**Detailed documentation:** [Sleep Staging — detailed reference](monitor_sleep_staging.md)

::: scpn_phase_orchestrator.monitor.sleep_staging

## Hybrid Order Monitoring

Hybrid classical/quantum order-parameter monitors and deterministic example
fixtures for review-only cosimulation evidence.

::: scpn_phase_orchestrator.monitor.hybrid_order

::: scpn_phase_orchestrator.monitor.hybrid_order_examples

## Information Replay Examples

Domain-specific information replay fixtures for cyber-industrial,
infrastructure, and physiology validation paths.
Physiology replay records enforce non-actuating audit boundaries, integer
sample/bin/oscillator counts, finite non-negative metrics, unit-interval
normalised Phi, and boolean-alias-free minimum partitions before replay corpus
relationships are accepted.
Infrastructure replay records apply the same engineering-proxy boundary to
power-grid and traffic-corridor replay corpora: sample/bin/oscillator counts are
integer-only, metrics are finite real non-negative values, normalised Phi is
bounded to the unit interval, and minimum partitions reject boolean aliases
before the re-synchronisation/recovery ordering contracts are accepted.
Cyber-industrial replay records apply the same boundary to lateral-movement and
manufacturing SPC corpora so containment/recovery ordering claims are accepted
only after integer-only record counts, finite real metrics, bounded normalised
Phi, and boolean-alias-free minimum partitions pass validation.

::: scpn_phase_orchestrator.monitor.information_replay_cyber_industrial

::: scpn_phase_orchestrator.monitor.information_replay_infrastructure

::: scpn_phase_orchestrator.monitor.information_replay_physiology

## Self-Model Reconfiguration

Self-model error records and review-only reconfiguration examples.

::: scpn_phase_orchestrator.monitor.self_model

::: scpn_phase_orchestrator.monitor.self_model_examples
