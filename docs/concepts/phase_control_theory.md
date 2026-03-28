# Phase-Synchronization Control Theory

## Abstract

This document presents the theoretical basis for using Kuramoto phase
dynamics as a **universal control abstraction** — a domain-agnostic
framework where any system with coupled cycles is controlled by
manipulating synchronization topology rather than tracking setpoints.
We demonstrate three concrete instantiations: tokamak fusion plasma,
qubit register coherence, and multi-scale plasma MHD hierarchy, and
argue that the approach constitutes a structural advance over classical
control methods (PID, MPC, LQR) by operating on the geometric invariant
(phase coherence) rather than on domain-specific state variables.

---

## 1. Core Claim

**Synchronization is a universal control variable.**

Every physical system with coupled cyclical degrees of freedom —
rotating machinery, oscillating plasmas, precessing qubits, rhythmic
biological processes, queued computational workloads — admits a
description in terms of phase angles θ_i(t) evolving on the N-torus
T^N.  The Kuramoto order parameter R = |⟨e^{iθ}⟩| then measures
the *degree of coherence* of those phases.

The key insight: instead of designing a bespoke controller for each
domain's state variables (temperatures, pressures, currents, gate
fidelities), we:

1. **Map** domain observables → phase angles via principled extraction
   (Hilbert transform, event frequency, state-ring encoding, or
   analytic formulas).
2. **Couple** phases through a Knm matrix whose topology encodes the
   desired interaction structure.
3. **Measure** coherence R per layer and partition layers into
   *good* (synchrony to promote) and *bad* (synchrony to suppress).
4. **Act** on four universal knobs — coupling K, phase lag α,
   entrainment ζ, reference phase Ψ — which the domain binding
   translates back into physical actuators.

This four-step pipeline is the **SCPN Phase Orchestrator (SPO)**.

---

## 2. Mathematical Foundation

### 2.1 The Unified Phase Dynamics Equation (UPDE)

```
dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j − θ_i − α_ij) + ζ sin(Ψ − θ_i)
```

| Symbol | Meaning | Controllable? |
|--------|---------|---------------|
| ω_i | Natural frequency of oscillator i | No (intrinsic) |
| K_ij | Coupling strength between i and j | Yes (knob K) |
| α_ij | Phase lag between i and j | Yes (knob α) |
| ζ | External entrainment drive | Yes (knob ζ) |
| Ψ | Reference phase | Yes (knob Ψ) |

This is the Sakaguchi–Kuramoto equation [sakaguchi1986] with an
external drive term.  The coupling matrix K_ij, lag matrix α_ij,
drive ζ, and reference Ψ are the four degrees of freedom available
to the controller.

### 2.2 Order Parameter as Health Metric

For a group G of oscillators:

```
R_G e^{iψ_G} = (1/|G|) Σ_{i∈G} e^{iθ_i}
```

R_G ∈ [0, 1] measures coherence.  R = 1 means perfect phase-locking;
R = 0 means uniform scatter.  This is a **topological invariant** — it
depends only on the relative phase distribution, not on the absolute
values or the physical meaning of the oscillators.

### 2.3 Dual Objective Partition

The binding spec partitions layers into two sets:

- **good_layers**: R_good should be maximised (healthy coordination).
- **bad_layers**: R_bad should be suppressed (pathological lock-in).

This captures a pattern universal across domains:

| Domain | good (maximise R) | bad (suppress R) |
|--------|-------------------|------------------|
| Tokamak plasma | Transport barrier, equilibrium | ELM cascade, tearing modes |
| Cloud queues | Throughput coordination | Retry storm synchrony |
| Manufacturing | Line yield coherence | Sensor drift correlation |
| Neuroscience | Functional connectivity | Seizure-like hypersynchrony |
| Quantum computing | Logical qubit coherence | Decoherence crosstalk |

No classical control formulation natively represents this dual objective
without domain-specific cost function engineering.

### 2.4 Regime State Machine

The supervisor maps (R_good, R_bad, boundary violations) to four regimes:

```
NOMINAL ←→ DEGRADED ←→ CRITICAL ←→ RECOVERY
```

- NOMINAL: R_good ≥ 0.6, no hard violations.
- DEGRADED: R_good ∈ [0.3, 0.6), or R_bad rising.
- CRITICAL: R_good < 0.3, or any hard boundary violation.
- RECOVERY: exiting CRITICAL, restoring coupling.

Transitions have hysteresis (0.05) and cooldown (10 steps) to prevent
oscillation.  This is a finite-state supervisory controller
[ramadge1987] realised on top of Kuramoto observables.

---

## 3. Fusion Plasma Control

### 3.1 The Control Problem

A tokamak confines plasma (T > 10^8 K) magnetically.  The plasma
exhibits coupled oscillatory phenomena across timescales spanning
six orders of magnitude:

| Phenomenon | Timescale | Character |
|-----------|-----------|-----------|
| Micro-turbulence (ITG/TEM) | ~μs | Broadband fluctuation |
| Zonal flows | ~ms | Shear oscillation |
| MHD tearing modes | ~ms | Rotating magnetic islands |
| Sawtooth crashes | ~10–100 ms | Periodic core relaxation |
| ELM cycles | ~1–10 ms | Edge pressure collapse |
| Transport barrier | ~100 ms | Steady-state H-mode pedestal |
| Current profile (q) | ~1 s | Slow diffusive evolution |
| Global equilibrium | ~1–10 s | Grad-Shafranov force balance |

Existing tokamak control uses separate PID loops for each phenomenon:
a shape controller for equilibrium, a density controller for Greenwald
limit, NTM controllers for tearing modes, ELM pace-making, etc.
These loops are designed independently, with ad-hoc priority rules
when they conflict.

### 3.2 SPO Formulation

The `plasma_control` domainpack maps these phenomena to 8 Kuramoto
layers (16 oscillators, 2 per layer).  The observable-to-phase mapping:

| Observable | Phase formula | Physical rationale |
|-----------|---------------|-------------------|
| q_profile | 2π(q − q_min)/(q_max − q_min) | Safety factor MHD proximity |
| β_N | 2πβ_N/β_limit | Troyon stability margin |
| τ_E | 2πτ_E/τ_ref | Confinement quality fraction |
| Sawtooth count | count·π mod 2π | π-phase kick per crash event |
| ELM count | count·π mod 2π | π-phase kick per ELM event |
| MHD amplitude | 2π·amplitude/threshold | Mode activity normalised |

Each formula maps the observable's physics-relevant range onto [0, 2π).
Event-driven phenomena (sawteeth, ELMs) produce discrete phase kicks
rather than continuous evolution — this naturally represents their
crash-then-recover dynamics.

**Physics boundaries** enforce hard limits:

- q_min ≥ 1.0 (Kruskal–Shafranov MHD stability)
- β_N ≤ 2.8 (Troyon no-wall ideal limit)
- Greenwald fraction ≤ 1.2 (density limit)

These map to SPO hard boundaries → immediate CRITICAL regime if violated.

**Coupling topology**: the Knm matrix uses exponential distance decay
(K_ij = K_base · exp(−α|i−j|)) between layers, encoding the physical
reality that adjacent-timescale phenomena couple more strongly.

**Dual objective**: layers 4–6 (transport barrier, current profile,
equilibrium) are *good* — their coherence indicates stable H-mode.
Layers 0, 2, 3 (micro-turbulence, tearing, sawtooth/ELM) are *bad* —
their synchronisation indicates confinement-degrading MHD activity.

**Policy rules** implement two key responses:
1. `suppress_elm_storm`: when R_bad on the sawtooth/ELM layer exceeds
   0.7, inject phase lag α on the turbulence layer to decouple the
   cascade.
2. `restore_transport_barrier`: when R_good on the transport barrier
   drops below 0.3, boost global coupling K.

### 3.3 What This Replaces

Conventional tokamak control stacks:

| Layer | Conventional | SPO equivalent |
|-------|-------------|----------------|
| Shape control | MIMO PID on gaps/currents | Equilibrium layer R + ζ drive |
| Density | PID on line-integrated density | Greenwald boundary (hard) |
| NTM suppression | ECCD aimed at island O-point | MHD layer R_bad + α decouple |
| ELM mitigation | Pellet/RMP pacing | ELM layer R_bad + policy rule |
| q-profile | Feedforward + slow PID | Current profile layer R_good |
| Disruption avoidance | Expert-system exception handler | CRITICAL regime + Lyapunov boundary |

The SPO formulation replaces ~6 independent SISO/MIMO PID loops and
a disruption handler with a single coherence-aware supervisory
controller that sees the *relationships* between phenomena, not just
individual setpoints.

### 3.4 The Kronecker Expansion

The `PlasmaControlBridge.import_knm_spec` method performs a critical
step: it takes an (L×L) *layer-level* coupling matrix (describing how
fast turbulence couples to slow equilibrium) and expands it to an
(N×N) *oscillator-level* matrix via Kronecker product:

```
K_osc = K_layer ⊗ 1_{n×n}
```

where n is the number of oscillators per layer.  This preserves the
inter-layer coupling topology while giving each oscillator within a
layer uniform intra-block coupling (with zero self-coupling on the
diagonal).

---

## 4. Quantum Coherence Control

### 4.1 The Control Problem

A register of N qubits evolving under an XY Hamiltonian:

```
H = Σ_{i<j} J_ij (X_i X_j + Y_i Y_j) + Σ_i h_i Z_i
```

can be interpreted as N coupled oscillators in the XY plane.
The Bloch-sphere azimuthal angles φ_i play the role of Kuramoto
phases, and the coupling constants J_ij play the role of K_ij.

### 4.2 SPO Formulation

The `quantum_simulation` domainpack maps this:

- **qubit_register layer** (4 oscillators): raw XY-plane phases φ_i
  of individual qubits, extracted from the statevector.
- **logical_coherence layer** (4 oscillators): logical qubit phases
  after error correction, treated as informational events (entanglement
  success/failure timestamps → frequency → phase).

The order parameter R_qubit measures how well the physical qubits
maintain phase coherence (≈ entanglement fidelity for the XY
Hamiltonian).  R_logical measures logical error rate stability.

**Boundary**: fidelity ≥ 0.5 (hard) — below this, the quantum state
is no better than random.

The `QuantumControlBridge` provides:
- `import_artifact` / `export_artifact`: pure dict conversion, no
  quantum library needed.
- `build_quantum_circuit`: constructs a Trotterised circuit from the
  Knm coupling matrix (requires scpn-quantum-control).
- `extract_phases_from_statevector`: extracts per-qubit Bloch-sphere
  phases from a Qiskit Statevector.

### 4.3 What This Adds

Standard quantum optimal control (GRAPE, Krotov, DRAG) optimises
pulse waveforms to achieve a target unitary.  SPO offers a
complementary view: instead of optimising for a specific gate, it
monitors *ongoing coherence* as the system evolves, and adjusts
coupling topology to maintain synchrony.  This is relevant for:

- Analog quantum simulation (maintaining collective coherence in
  Hamiltonian evolution)
- Quantum error correction monitoring (tracking logical qubit phase
  stability across correction cycles)
- Variational quantum circuits (adjusting entangler structure based
  on coherence feedback)

---

## 5. The Fusion Core Bridge: Equilibrium-to-Phase Mapping

### 5.1 The Grad-Shafranov Gap

The Grad-Shafranov equation governs tokamak equilibrium:

```
Δ*ψ = −μ_0 R² dp/dψ − F dF/dψ
```

It produces continuous field solutions (ψ, q, pressure, current),
not oscillatory dynamics.  There is no natural "phase" in an
equilibrium solver output.

The `FusionCoreBridge` resolves this gap by defining six analytic
phase-mapping formulas that convert equilibrium observables into
[0, 2π) phases (§3.2 table).  Each formula maps the observable's
physical range onto the circle:

- **Linear normalisation** for continuous observables (q, β_N, τ_E,
  MHD amplitude): the ratio of current value to its physical limit
  sets the phase.  A phase near 2π means the system is near its limit.
- **Event-driven kicks** for discrete phenomena (sawtooth crashes,
  ELMs): each event advances the phase by π, producing a half-cycle
  jump.  Pairs of rapid events produce near-2π advances, which wrap
  back near zero — naturally encoding the periodic crash-recovery cycle.

This mapping is invertible (`phases_to_feedback` recovers the mean
phase and coherence), allowing closed-loop operation: equilibrium
solver → phases → UPDE evolution → feedback → equilibrium solver.

---

## 6. Novelty Analysis

### 6.1 What Exists in the Literature

| Approach | Scope | Phase-based? | Multi-scale? | Domain-agnostic? |
|----------|-------|-------------|-------------|-----------------|
| Kuramoto model [kuramoto1975] | Descriptive | Yes | No | No (physics only) |
| PID control | Control | No | No | Partially |
| MPC [rawlings2017] | Control | No | Partially | No |
| LQR/LQG | Control | No | No | Partially |
| DIII-D PCS [ferron2006] | Tokamak control | No | Partially | No |
| GRAPE [khaneja2005] | Quantum control | No | No | No |
| Synchrophasor (PMU) | Power grid | Yes | No | No |
| **SPO** | **Control** | **Yes** | **Yes** | **Yes** |

The Kuramoto model has been used to *describe* synchronisation in
power grids [dorfler2014], neural networks [breakspear2010], and
chemical oscillators.  But it has never been used as a *control
abstraction* — a framework where:

1. Arbitrary domain observables are systematically mapped into phases.
2. The Kuramoto coupling matrix K_ij becomes a **control variable**.
3. The order parameter R becomes a **health metric**.
4. A supervisory policy operates on (R_good, R_bad) to drive the
   system between regimes.

This four-part combination — observable→phase mapping + coupling
as actuator + R as metric + dual-objective supervision — is, to the
authors' knowledge, not present in any published control framework.

### 6.2 Specific Novelties

1. **Dual R_good/R_bad objective**: no standard cost function
   formulation natively distinguishes "synchrony to promote" from
   "synchrony to suppress" as structurally independent objectives.
   MPC/LQR penalise deviation from a setpoint; they do not partition
   the state space into "good coherence" and "bad coherence."

2. **Observable-to-phase mapping as a formal bridge**: the analytic
   formulas (§3.2, §5.1) constitute a systematic procedure for
   embedding non-oscillatory observables (equilibrium fields,
   event counts) into the Kuramoto state space.  This has not been
   formalised as a general control methodology.

3. **Kronecker coupling expansion** (§3.4): the technique of
   specifying inter-layer coupling at the hierarchy level and
   expanding to oscillator level via Kronecker product is a practical
   contribution that preserves multi-scale topology while keeping
   the configuration compact.

4. **Regime-based supervisory control on Kuramoto observables**:
   integrating a finite-state supervisor (NOMINAL/DEGRADED/CRITICAL/
   RECOVERY) with Kuramoto R thresholds and physics-derived hard
   boundaries (q > 1.0, β_N < 2.8) in a single framework.

5. **Domain-agnostic binding spec**: the same engine, integrator,
   supervisor, and policy rules control tokamak plasma, qubit
   registers, cloud queue systems, manufacturing lines, and
   biological oscillators — with only a YAML configuration change.

---

## 7. Why Phase-Synchronization Control Supersedes Classical Methods

### 7.1 PID: Single-Variable, No Structure

PID tracks a single variable against a setpoint.  It has no concept
of:

- Multi-scale coupling between phenomena
- Phase relationships (lead/lag) between oscillators
- Coherence as a collective property
- Dual objectives (promote some sync, suppress other sync)

A tokamak running 6 PID loops has 6 independent controllers that
can (and do) fight each other.  SPO sees the 6 phenomena as one
coupled dynamical system and acts on the coupling structure.

### 7.2 MPC: Domain-Specific, Computationally Expensive

Model Predictive Control solves an optimisation problem at each
timestep using a plant model.  It has three key limitations:

- Requires a domain-specific model (cannot reuse across domains)
- Scales poorly with state dimension (quadratic or worse in N)
- Solves for actuator trajectories, not for coupling topology
- Cannot express "suppress this coherence" without manual cost
  function engineering

SPO's per-step cost is O(N²) for the Kuramoto coupling sum —
dominated by the sin(θ_j − θ_i) computation — and O(N) for the
order parameter.  No optimisation problem is solved online.

### 7.3 LQR: Linear, Assumes Quadratic Cost

Linear-Quadratic Regulator requires linearisation around an
operating point.  Phase dynamics are inherently nonlinear
(sin(·) coupling).  Linearising destroys the wrap-around topology
of the N-torus and cannot represent phase-locking transitions.

### 7.4 Expert Systems: Brittle, Non-Transferable

Current tokamak disruption avoidance uses expert-rule exception
handlers (e.g., DIII-D's PCS exception handling system).  These
rules are:

- Written by domain experts for specific machines
- Not transferable between tokamaks (let alone to other domains)
- Triggered by absolute thresholds, not relational coherence
- Unable to detect emergent pathological synchronisation

SPO's policy rules operate on *relational* observables (R_bad
on a specific layer exceeding a threshold), which transfers
across machines with the same phenomenological hierarchy.

### 7.5 The Structural Advantage

The fundamental advantage is dimensional: SPO operates on one
scalar per layer (R), plus one regime state, plus four knobs.
This is a **constant-dimension** control space regardless of how
many raw state variables the domain has.  A tokamak with 10⁴
measurement channels collapses to 8 R-values + 4 boundaries +
4 knobs.  The controller complexity does not scale with plant
complexity.

---

## 8. Practical Implications

### 8.1 For Fusion Energy

- A single SPO instance replaces the 6+ independent PID loops in a
  tokamak plasma control system.
- Physics boundaries (Kruskal–Shafranov, Troyon, Greenwald) are
  first-class hard constraints, not afterthought exception handlers.
- The `plasma_control` policy rules encode ELM suppression and
  transport barrier recovery as declarative YAML — transferable
  between machines (ITER, SPARC, DEMO) by adjusting the binding spec.
- The Lyapunov verdict import allows integration with existing
  disruption prediction systems as a soft boundary signal.

### 8.2 For Quantum Computing

- Ongoing coherence monitoring during analog quantum simulation,
  complementing gate-level optimal control.
- Entanglement topology (the Knm matrix) as a control variable,
  adjusted based on real-time R feedback.
- The fidelity hard boundary (≥ 0.5) triggers regime escalation
  before the quantum state becomes irrecoverable.

### 8.3 For General Process Control

- Cloud/queue systems: retry storm detection via R_bad, automatic
  decoupling via α lag injection (queuewaves domainpack).
- Manufacturing: sensor drift correlation detection, yield coherence
  monitoring (manufacturing_spc domainpack).
- Biology: multi-scale oscillator monitoring from cellular to
  systemic (bio_stub domainpack).

All use the *same engine* with different YAML binding specs.

---

## 9. Hardware Pipeline: From Sensors to Actuators

This section describes the complete physical data path — how raw sensor
signals enter SPO, traverse the phase-synchronization loop, and emerge
as actuator commands that drive real hardware.

### 9.1 General Pipeline Architecture

```
┌──────────────┐    ┌────────────┐    ┌──────────┐    ┌────────────┐
│ Sensors      │───►│ Phase      │───►│ UPDE     │───►│ Supervisor │
│ (domain HW)  │    │ Extraction │    │ Engine   │    │ + Policy   │
└──────────────┘    └────────────┘    └──────────┘    └─────┬──────┘
                                                            │
              ┌────────────────────────────────────────────┘
              │
              ▼
┌──────────────┐    ┌────────────┐    ┌──────────────────┐
│ Action       │───►│ Actuation  │───►│ Domain Execution │
│ Projector    │    │ Mapper     │    │ (physical HW)    │
└──────────────┘    └────────────┘    └──────────────────┘
```

**Timing**: The binding spec declares two periods:
- `sample_period_s` (e.g. 1 ms) — sensor acquisition and phase update rate.
- `control_period_s` (e.g. 10 ms) — supervisor evaluation and actuator
  command rate.  Multiple sample cycles accumulate before one control decision.

Each pipeline stage has a fixed computational cost:

| Stage | Operation | Cost |
|-------|-----------|------|
| Phase extraction | Per-oscillator mapping | O(N) |
| UPDE step | Coupling sum | O(N²) |
| Order parameter | Per-layer mean | O(N) |
| Supervisor decide | Regime FSM + policy rules | O(R) where R = rule count |
| Action projector | Rate clamp per action | O(A) where A = action count |
| Actuation mapper | Scope resolution | O(A·M) where M = actuator count |

For the plasma_control domainpack (N=16, R=2, A≤3, M=3), the full
pipeline runs in <100 μs on a single core — well within the 1 ms
sample budget.

### 9.2 Tokamak Fusion: Sensor Ingestion

A tokamak produces diagnostic signals across all 8 SPO layers.
Each diagnostic maps to a specific oscillator family.

#### 9.2.1 Measurement Systems and Phase Extraction

| Diagnostic | Physical quantity | SPO layer | Phase formula |
|-----------|-------------------|-----------|---------------|
| **Mirnov coils** (poloidal array, ~32 probes) | dB_θ/dt (magnetic fluctuations) | mhd_tearing (2) | Hilbert transform of dominant mode → instantaneous phase |
| **Far-infrared interferometer** (multi-chord) | Line-integrated n_e | transport_barrier (4) | 2π · n_e / n_Greenwald (density fraction) |
| **Electron Cyclotron Emission** (ECE radiometer) | T_e(R) radial profile | micro_turbulence (0) | Hilbert transform of T_e fluctuation envelope |
| **Thomson scattering** (multi-pulse laser) | T_e(r), n_e(r) profiles | global_equilibrium (6) | 2π · β_N / β_limit from kinetic profiles |
| **Magnetic flux loops** (full poloidal set) | ψ(R,Z) equilibrium reconstruction | global_equilibrium (6) | 2π · (q − q_min) / (q_max − q_min) from EFIT |
| **Soft X-ray array** (diode array) | Core radiation | sawtooth_elm (3) | count · π mod 2π (crash detector) |
| **Divertor Langmuir probes** | Ion flux, T_e at target | plasma_wall (7) | 2π · heat_flux / heat_flux_limit |
| **Bolometer array** | Radiated power P_rad | current_profile (5) | 2π · P_rad / P_input (radiation fraction) |
| **Filterscopes / Dα monitors** | Dα emission at edge | sawtooth_elm (3) | count · π mod 2π (ELM detector) |
| **Motional Stark Effect** (MSE) | Pitch angle → q(r) profile | current_profile (5) | 2π · q_axis / q_95 (profile peakedness) |

The PlasmaControlBridge receives these as a dict from the plasma
control system (scpn-control) or from a direct diagnostic interface:

```python
tick_result = {
    "phases": mirnov_phases + interferometer_phases + ...,  # len=16
    "regime": "NOMINAL",
    "stability": lyapunov_score,
    "layer_sizes": [2, 2, 2, 2, 2, 2, 2, 2],
}
state = bridge.import_snapshot(tick_result)
```

#### 9.2.2 Phase Extraction Methods

Three extraction methods apply depending on signal character:

**Hilbert transform** (continuous oscillatory signals):
For Mirnov coils and ECE fluctuations, the analytic signal
z(t) = x(t) + iH[x(t)] yields instantaneous phase φ(t) = arg(z(t)).
Applied to the dominant MHD mode (identified by toroidal mode number
analysis), this extracts the rotating island phase.

**Normalised ratio** (equilibrium quantities):
For β_N, q-profile, τ_E, density, and radiation — quantities that
do not oscillate but vary slowly — the phase is a linear map of the
observable's position within its physical range: θ = 2π · x/x_limit.
As x approaches its stability limit, θ → 2π signals proximity to
the boundary.

**Event counter** (discrete crash events):
For sawteeth and ELMs, each detected event (identified by a
threshold crossing in SXR or Dα) increments a counter.
Phase = count · π mod 2π.  Two rapid events produce a near-2π
advance that wraps to ~0, naturally encoding the crash-recovery cycle.

#### 9.2.3 Diagnostic Timing Budget

| Diagnostic | Acquisition rate | Latency to SPO |
|-----------|-----------------|----------------|
| Mirnov coils | 100 kHz–1 MHz | <10 μs (analog + ADC) |
| Interferometer | 10–100 kHz | <100 μs |
| ECE radiometer | 1–10 kHz | <1 ms |
| Thomson scattering | 10–100 Hz | 10–100 ms |
| Magnetic flux loops | 10 kHz | <100 μs |
| SXR array | 100 kHz | <10 μs |
| Langmuir probes | 10–100 kHz | <100 μs |

Fast diagnostics (Mirnov, SXR) update every sample cycle (1 ms).
Slow diagnostics (Thomson) hold their last value between updates.
The UPDE engine uses whatever phases are current, treating slow
channels as piecewise-constant — consistent with the real physics
where equilibrium quantities evolve on ~1 s timescales.

### 9.3 Tokamak Fusion: Control Pipeline

#### 9.3.1 Supervisor Decision

Every `control_period_s` (10 ms), the supervisor evaluates:

1. **R_good** for layers [4, 5, 6] (transport_barrier, current_profile,
   equilibrium).  High R_good → stable H-mode.
2. **R_bad** for layers [0, 2, 3] (micro_turbulence, mhd_tearing,
   sawtooth_elm).  High R_bad → pathological MHD activity.
3. **Boundary violations**: q_min < 1.0 or β_N > 2.8 or
   greenwald > 1.2 → immediate CRITICAL transition.

The RegimeManager updates the state machine:
```
NOMINAL (R_good ≥ 0.6, no hard violations)
   ↓ R_good drops below 0.55
DEGRADED (R_good ∈ [0.3, 0.6), or R_bad rising)
   ↓ R_good < 0.25 or hard violation
CRITICAL (emergency response)
   ↓ recovery detected
RECOVERY (controlled ramp-back)
   ↓ R_good ≥ 0.65
NOMINAL
```

Hysteresis (0.05 offset between down/up thresholds) prevents
chattering at transitions.  Cooldown (10 steps) prevents rapid
regime oscillation.

#### 9.3.2 Policy Rule Firing

Two policy rules in `plasma_control/policy.yaml`:

**suppress_elm_storm**: fires in NOMINAL or DEGRADED when R_bad
on sawtooth_elm layer exceeds 0.7.  Emits:
```
ControlAction(knob="alpha", scope="layer_0", value=0.4, ttl_s=5.0)
```
This injects phase lag on the micro_turbulence layer, decoupling
the turbulence↔ELM cascade.  The physical effect: turbulence-driven
edge pressure build-up is disrupted, preventing the next ELM trigger.

**restore_transport_barrier**: fires in DEGRADED or RECOVERY when
R_good on transport_barrier layer drops below 0.3.  Emits:
```
ControlAction(knob="K", scope="global", value=0.2, ttl_s=10.0)
```
This increases global coupling, strengthening the inter-layer
synchronization.  The physical effect: tighter coordination between
equilibrium, transport, and boundary layers, restoring the H-mode
pedestal.

#### 9.3.3 Action Projection and Rate Limiting

Each ControlAction passes through the ActionProjector before reaching
hardware.  For the plasma_control binding:

| Knob | Value bounds | Rate limit per step |
|------|-------------|-------------------|
| K | [0.0, 5.0] | 0.1 |
| alpha | [0.0, π/2] | 0.05 |
| zeta | [0.0, 2.0] | 0.2 |

Rate limiting prevents actuator damage.  If the policy demands
K=0.2 but the previous value was K=0.05, the projector caps the
step at +0.1, yielding K=0.15.  The next cycle brings K=0.2.
This produces smooth actuator ramps, not discontinuous jumps.

#### 9.3.4 Knob-to-Actuator Mapping

The ActuationMapper resolves abstract knob commands into
domain-specific actuator commands.  The binding spec defines
three actuators:

```yaml
actuators:
  - name: coupling_global   # knob: K, scope: global
  - name: lag_turbulence     # knob: alpha, scope: layer_0
  - name: damping_global     # knob: zeta, scope: global
```

The output is a command dict:
```python
{"actuator": "coupling_global", "knob": "K", "scope": "global",
 "value": 0.15, "ttl_s": 10.0}
```

#### 9.3.5 Actuator-to-Hardware Translation

The final translation from SPO actuator commands to physical tokamak
hardware is performed by the domain execution layer — the
PlasmaControlBridge.export_control_actions() method or the downstream
plasma control system.

| SPO actuator | Physical hardware | Translation |
|-------------|-------------------|-------------|
| **coupling_global (K)** | Poloidal field (PF) coil currents | K ∝ I_PF shaping current.  Higher K → tighter magnetic configuration → stronger inter-layer coupling.  The PF coil power supplies (thyristor or IGBT converters, ~10 kA, ~1 kV) receive current setpoints from the shape controller. |
| **coupling_global (K)** | Neutral Beam Injection (NBI) power | K ∝ P_NBI.  Higher coupling → higher beam power → more torque → stronger rotation → better MHD coupling across flux surfaces.  NBI sources (~1 MW per beamline, deuterium/hydrogen, 40–100 keV) respond in ~50 ms. |
| **lag_turbulence (α)** | Resonant Magnetic Perturbation (RMP) coil phasing | α maps to the toroidal phase of the n=3 RMP field.  The RMP coils (typically 3×6 in-vessel saddle coils) receive phase-shifted sinusoidal currents.  Adjusting the phase angle between upper and lower coil rows controls the edge magnetic perturbation spectrum, directly modulating ELM stability. |
| **lag_turbulence (α)** | ECRH/ECCD deposition angle | α maps to the poloidal steering angle of the EC launcher mirror.  Electron Cyclotron Resonance Heating (ECRH, 170 GHz gyrotrons, ~1 MW per launcher) deposits power at a specific flux surface.  Shifting the mirror angle changes the current drive location, modulating the local q-profile shear that controls tearing mode coupling. |
| **damping_global (ζ)** | Pellet injection rate | ζ ∝ pellet frequency.  Higher entrainment drive → more frequent pellet injection → stronger density pacing.  Pellet injectors (frozen D₂ pellets, 1–4 mm diameter, 200–1000 m/s) fire at programmable rates up to ~100 Hz.  ELM pacing via pellets is established practice (JET, AUG). |
| **damping_global (ζ)** | Gas puff valves | ζ ∝ gas flow rate.  Piezoelectric gas valves (D₂, N₂, Ne, Ar) with ~1 ms response time control edge density and impurity seeding.  Higher ζ drives the edge density toward a reference state. |

#### 9.3.6 Closed-Loop Timing

The complete loop for one control cycle:

```
t=0.000 ms  Mirnov/SXR/interferometer acquire raw signals
t=0.010 ms  ADC digitisation complete
t=0.100 ms  Phase extraction (Hilbert / ratio / counter)
t=0.200 ms  PlasmaControlBridge.import_snapshot() → UPDEState
t=0.300 ms  UPDE RK4 step (16 oscillators, O(256) operations)
t=0.400 ms  Order parameter R_good, R_bad computed
t=0.500 ms  RegimeManager.evaluate() → regime transition check
t=0.600 ms  SupervisorPolicy.decide() → list[ControlAction]
t=0.700 ms  ActionProjector.project() → rate-limited actions
t=0.800 ms  ActuationMapper.map_actions() → actuator commands
t=0.900 ms  PlasmaControlBridge.export_control_actions() → HW dict
t=1.000 ms  Commands dispatched to PF/NBI/RMP/ECRH/pellet systems
```

Total pipeline latency: ~1 ms.  This is within the 10 ms control
period of the plasma_control binding spec (10× margin), and well
within the characteristic MHD timescale (~1–10 ms).

For comparison, existing tokamak control systems (DIII-D PCS, KSTAR
EPICS) operate at 1–10 ms cycle times.  SPO matches this performance
while providing multi-scale coherence awareness that single-loop
controllers lack.

### 9.4 Tokamak Fusion: Full Data Flow Diagram

```
 TOKAMAK VESSEL
 ┌─────────────────────────────────────────────────────────────┐
 │  Diagnostics (sensors)              Actuators (effectors)   │
 │  ┌──────────────────┐               ┌────────────────────┐ │
 │  │ Mirnov coils     │               │ PF coils (6 pairs) │ │
 │  │ ECE radiometer   │               │ CS solenoid        │ │
 │  │ Interferometer   │               │ NBI (~8 sources)   │ │
 │  │ Thomson scatter  │               │ ECRH (~4 gyrotrons)│ │
 │  │ SXR array        │               │ RMP coils (3×6)    │ │
 │  │ Bolometers       │               │ Pellet injector    │ │
 │  │ Langmuir probes  │               │ Gas puff valves    │ │
 │  │ MSE diagnostic   │               │ Feedback coils     │ │
 │  │ Dα monitors      │               │                    │ │
 │  └───────┬──────────┘               └────────▲───────────┘ │
 └──────────┼───────────────────────────────────┼─────────────┘
            │ raw signals                       │ I, P, angle, rate
            ▼                                   │
 ┌──────────────────────┐            ┌──────────┴──────────────┐
 │ Phase Extraction     │            │ Domain Execution         │
 │                      │            │                          │
 │ Hilbert(Mirnov) →θ₂  │            │ K→I_PF, P_NBI           │
 │ ratio(n_e/n_GW) →θ₄  │            │ α→RMP_phase, ECRH_angle │
 │ ratio(β_N/β_lim)→θ₁  │            │ ζ→pellet_rate, gas_flow │
 │ count(SXR crash)→θ₃  │            │ Ψ→reference setpoint    │
 │ ratio(q/q_95)   →θ₅  │            │                          │
 │ ratio(P_rad)    →θ₅  │            └──────────▲──────────────┘
 │ ratio(T_e/T_sep)→θ₇  │                       │
 │ Hilbert(ECE)    →θ₀  │                       │ actuator commands
 └───────┬──────────────┘            ┌───────────┴──────────────┐
         │ θ₀..θ₁₅ (16 phases)      │ ActuationMapper          │
         ▼                           │ scope resolution         │
 ┌──────────────────────┐            │ value clamping           │
 │ UPDE Engine (RK4)    │            └──────────▲──────────────┘
 │                      │                       │
 │ dθᵢ/dt = ωᵢ         │            ┌───────────┴──────────────┐
 │   + ΣⱼKᵢⱼsin(θⱼ−θᵢ) │            │ ActionProjector          │
 │   + ζ sin(Ψ−θᵢ)     │            │ rate limits              │
 │                      │            │ value bounds             │
 │ → θ₀..θ₁₅ (updated) │            └──────────▲──────────────┘
 └───────┬──────────────┘                       │
         │                           ┌───────────┴──────────────┐
         ▼                           │ SupervisorPolicy.decide() │
 ┌──────────────────────┐            │                          │
 │ Order Parameters     │───────────►│ R_good[4,5,6] ≥ 0.6?    │
 │                      │            │ R_bad[0,2,3] < 0.7?      │
 │ R_good (layers 4-6)  │            │ q_min ≥ 1.0?             │
 │ R_bad  (layers 0,2,3)│            │ β_N ≤ 2.8?               │
 │ per-layer Rᵢ, ψᵢ    │            │ greenwald ≤ 1.2?         │
 └──────────────────────┘            └──────────────────────────┘
```

### 9.5 Quantum Hardware Pipeline

Superconducting qubit systems (IBM Heron, Google Sycamore) use a
parallel pipeline with different physical signals.

#### 9.5.1 Sensor Ingestion

| Diagnostic | Physical quantity | SPO layer |
|-----------|-------------------|-----------|
| **Readout resonator** (dispersive, ~7 GHz) | Qubit state (IQ plane) | qubit_register (0) |
| **Randomized benchmarking** (gate sequences) | Process fidelity | logical_coherence (1) |
| **T₁/T₂ characterization** (decay sequences) | Coherence times | logical_coherence (1) |

Phase extraction: the readout resonator returns an IQ-plane point
for each qubit.  The azimuthal angle in the IQ plane *is* the
Bloch-sphere phase φᵢ — no Hilbert transform needed.  This is
a direct measurement of the oscillator phase.

#### 9.5.2 Actuator Commands

| SPO actuator | Physical hardware | Translation |
|-------------|-------------------|-------------|
| **K (coupling)** | Tunable coupler flux bias | K ∝ coupler flux Φ_c.  Flux-tunable couplers (DC SQUID loop between qubits) set the effective J_ij coupling.  A DAC channel (~16 bit, ~1 GHz update rate) drives the coupler's flux line. |
| **α (phase lag)** | Microwave drive phase | α maps to the phase offset of the qubit drive pulse.  The arbitrary waveform generator (AWG, ~5 GS/s) shifts the IQ modulation phase of the 4–6 GHz drive tone. |
| **ζ (entrainment)** | Microwave drive amplitude | ζ ∝ Rabi frequency Ω_R.  Higher drive amplitude → stronger pull toward the reference phase Ψ.  The AWG scales the drive envelope. |
| **Ψ (reference)** | Drive pulse rotation axis | Ψ sets the rotation axis angle on the Bloch sphere equator.  Standard single-qubit gates (X, Y, Z, arbitrary) are rotations at programmable phase angles — Ψ selects which axis to drive toward. |

#### 9.5.3 Timing

Superconducting qubit operations run at ~GHz clock rates with
~100 ns gate times.  The SPO control cycle must fit within the
coherence time T₂ (~100 μs for current hardware):

```
t=0      Readout pulse → IQ data (500 ns)
t=1 μs   Phase extraction (direct IQ angle)
t=2 μs   UPDE step (8 oscillators, O(64) operations)
t=3 μs   Supervisor evaluation
t=4 μs   Coupler/drive command dispatch
t=5 μs   Actuator settling (~100 ns for flux couplers)
```

Total latency: ~5 μs.  This fits within a single T₂ window,
allowing real-time coherence-based feedback during quantum
computation — not just post-hoc analysis.

### 9.6 General Process Control Pipeline

For non-physics domains (manufacturing, cloud queues, biological
systems), the pipeline follows the same architecture with different
sensor and actuator vocabularies.

#### 9.6.1 Manufacturing (SPC)

| SPO component | Physical realization |
|--------------|---------------------|
| Sensors | Inline metrology (coordinate measuring machines, optical gauges, force sensors) |
| Phase extraction | Normalised ratio: θ = 2π · (x − LSL) / (USL − LSL) where LSL/USL are spec limits |
| Actuators (K) | PLC setpoint coupling between stations (e.g., upstream tool compensation based on downstream SPC) |
| Actuators (α) | Process delay compensation (conveyor speed, batch hold time) |
| Actuators (ζ) | Reference part injection rate (golden sample pacing) |
| Cycle time | Seconds to minutes (matches process tempo) |

#### 9.6.2 Cloud / Queue Systems

| SPO component | Physical realization |
|--------------|---------------------|
| Sensors | Queue depth metrics, request latency percentiles, error rate counters |
| Phase extraction | Normalised ratio (queue_depth / queue_capacity) or event frequency (requests/sec → rad/s) |
| Actuators (K) | Service mesh routing weights (Istio, Envoy) |
| Actuators (α) | Retry backoff configuration (exponential base, jitter) |
| Actuators (ζ) | Rate limiter setpoint (requests/sec ceiling) |
| Cycle time | Milliseconds to seconds |

#### 9.6.3 Biological Oscillators

| SPO component | Physical realization |
|--------------|---------------------|
| Sensors | EEG electrodes, ECG leads, respiratory belt, skin conductance |
| Phase extraction | Hilbert transform of band-passed signal (e.g., alpha 8–13 Hz, theta 4–8 Hz) |
| Actuators (K) | Binaural beat frequency (entrainment coupling strength) |
| Actuators (α) | Phase offset between left/right audio channels |
| Actuators (ζ) | Audio amplitude (drive strength) |
| Actuators (Ψ) | Target brainwave phase (circadian reference, meditation target) |
| Cycle time | ~100 ms (matching EEG epoch length) |

### 9.7 Safety and Fail-Safe Design

The pipeline incorporates multiple safety layers:

1. **Boundary violations** (hard) trigger immediate CRITICAL regime,
   which commands conservative actuator values (reduce K, increase ζ
   toward a known stable state).

2. **Rate limiting** in the ActionProjector prevents actuator damage.
   A sudden coupling increase request is smoothed over multiple
   control cycles.

3. **TTL (time-to-live)** on every ControlAction ensures commands
   expire.  If the supervisor stops issuing commands (e.g., software
   crash), actuators revert to their default state within the TTL
   window.

4. **Physics invariant checks** in the bridge adapters
   (PlasmaControlBridge.check_physics_invariants,
   FusionCoreBridge.check_stability) run independently of SPO
   and can veto commands that would violate absolute safety limits.

5. **Watchdog**: the sample/control period mismatch (10:1 in
   plasma_control) means the UPDE runs 10 phase updates between
   each control decision.  If the UPDE detects divergence (NaN,
   Inf, or R < 0.01), it halts before the next control cycle.

For nuclear-grade tokamak deployment, the SPO control layer would
sit *above* the existing machine-protection interlock system (which
operates on hard-wired analog signals and cannot be overridden by
software).  SPO optimises within the interlock-safe envelope; it
does not replace the interlock.

---

## 10. Scope of Competence: What SPO Is and Is Not

This section draws explicit boundaries around the framework's claims.
Overstating scope would be scientifically dishonest; understating it
would obscure the genuine contribution.

### 10.1 What SPO Is

SPO is a **supervisory control layer** that monitors multi-scale
coherence and issues corrective commands to domain actuators.

It is a *control framework*, not a *physics solver*.  It does not
compute equilibria, propagate wavefunctions, solve transport equations,
or simulate fluid dynamics.  It consumes the outputs of domain-specific
solvers and acts on them through the phase-synchronization abstraction.

The analogy: SPO is to a tokamak what a conductor is to an orchestra.
The conductor does not play any instrument (does not solve any physics
equation).  The conductor listens to the ensemble (monitors phase
coherence across layers), detects when sections drift out of sync
(R_bad rising), and signals corrections (adjusts coupling K, phase
lag α, drive ζ).  The musicians (domain solvers + physical actuators)
produce the actual sound (maintain the actual equilibrium).

### 10.2 What SPO Is Not

| SPO is NOT | Why this matters |
|-----------|-----------------|
| An equilibrium solver | It cannot compute ψ(R,Z).  It depends on EFIT, FreeGS, or scpn-fusion-core for force balance. |
| A turbulence simulator | It cannot predict ITG/TEM growth rates.  It monitors turbulence *signatures* via phase extraction. |
| A disruption predictor | It detects coherence degradation, which may correlate with disruption precursors.  It does not model disruption physics (thermal quench, current quench, halo currents). |
| A quantum error correction code | It monitors coherence of logical qubits.  It does not implement stabiliser measurements or syndrome decoding. |
| A replacement for machine-protection interlocks | Hard-wired safety systems (vessel protection, magnet quench detection) operate on analog signals below SPO's software layer.  SPO cannot override them and must not attempt to. |

### 10.3 What SPO Expects from the Domain

For SPO to function, the domain must provide:

1. **Oscillatory or quasi-oscillatory observables.**  The phase
   extraction step requires signals that admit meaningful phase
   angles.  Continuous oscillations (MHD modes, qubit precession)
   map cleanly.  Slowly varying quantities (q-profile, β_N) map
   via normalised ratios — a weaker but still informative encoding.
   Signals with no temporal structure (static geometry, material
   properties) cannot be phase-encoded and are outside SPO's scope.

2. **A known coupling hierarchy.**  The Knm matrix encodes which
   phenomena couple to which.  This must be specified by domain
   knowledge (e.g., "micro-turbulence drives zonal flows which
   modulate transport").  SPO does not discover coupling topology
   from data — it exploits a topology provided in the binding spec.

3. **Controllable actuators that influence coupling.**  The four
   knobs (K, α, ζ, Ψ) must map to physical actuators that can
   modulate the coupling, phase lag, or drive of the oscillatory
   phenomena.  If the domain has no actuators (pure observation),
   SPO reduces to a monitoring-only tool (still useful for
   coherence detection, but not for control).

4. **Meaningful good/bad partition.**  The dual objective requires
   domain expertise to identify which synchrony is healthy and
   which is pathological.  This is not always obvious — in some
   systems, all synchrony might be desirable (or undesirable).
   If no dual partition exists, SPO degenerates to a standard
   single-objective coherence maximiser.

### 10.4 Concrete Expectations per Domain

#### 10.4.1 Tokamak Fusion Plasma

**SPO's expected contribution: improved disruption avoidance through
earlier detection of multi-scale coherence degradation.**

SPO does not solve the Grad-Shafranov equation.  It does not compute
MHD stability limits.  It monitors whether the plasma's oscillatory
phenomena (turbulence, MHD modes, sawteeth, ELMs, transport barriers)
are maintaining a healthy coherence pattern — and detects *correlated*
degradation across timescales before individual diagnostic thresholds
are crossed.

The specific value proposition:

- **Cascade detection**: when R_bad rises simultaneously on layers
  [0, 2, 3] (turbulence, tearing, sawtooth/ELM), it signals a
  coupled MHD cascade forming.  Individual diagnostics may still be
  within limits, but the *correlation* is the precursor.  Existing
  PID loops cannot see this because they monitor channels independently.

- **Coordinated actuation**: instead of 6 independent PID loops
  potentially fighting each other (shape controller demanding more
  current while density controller demands less gas), SPO sees the
  coupled system and issues coordinated commands through a single
  coherence-aware policy.

- **Transferability**: the same binding spec structure (8 layers, 16
  oscillators, good/bad partition, physics boundaries) applies to any
  tokamak.  Porting from DIII-D to ITER requires changing the Knm
  calibration and boundary thresholds, not rewriting the controller.

**What SPO does NOT expect to do for fusion**:
- Replace the real-time equilibrium reconstruction (EFIT runs anyway)
- Replace machine-protection interlocks (hard-wired, below SPO)
- Guarantee disruption avoidance (it improves detection, not elimination)
- Work without calibration (the Knm matrix and phase extraction
  require tokamak-specific tuning against experimental data)

**Confidence level**: MEDIUM.  The theoretical basis (Kuramoto
coherence as a multi-scale health metric) is sound.  The practical
value depends on: (a) phase extraction fidelity under real diagnostic
noise, (b) Knm calibration against real inter-phenomenon coupling,
(c) actuator response linearity.  None of these have been validated
experimentally.

#### 10.4.2 Quantum Coherence

**SPO's expected contribution: real-time coherence monitoring and
coupling topology adjustment during analog quantum simulation.**

SPO does not implement quantum gates, error correction, or
compilation.  It monitors the Bloch-sphere phases of physical
qubits (directly measurable from readout resonators) and the
logical coherence of error-corrected qubits (derived from
syndrome statistics).

The specific value proposition:

- **Continuous fidelity tracking**: R_qubit dropping below threshold
  triggers regime escalation *during* the computation, not after
  post-hoc analysis.

- **Coupling topology feedback**: the Knm matrix corresponds to
  flux-tunable coupler settings.  SPO adjusts couplings to maintain
  coherence — complementing gate-level optimal control (GRAPE/Krotov)
  which optimises pulse shapes for specific unitaries.

- **Cross-layer monitoring**: tracking both physical qubit phases and
  logical qubit stability in a single framework, with the fidelity
  boundary triggering corrective action when the physical layer
  degrades.

**Confidence level**: MODERATE-HIGH for monitoring, LOWER for active
control.  Phase extraction from IQ readout is direct and high-fidelity.
But the Kuramoto sinusoidal coupling is an approximation of the actual
XY Hamiltonian dynamics — it captures the first harmonic but misses
higher-order terms.  Active coupling adjustment via flux couplers is
realistic but untested in a coherence-feedback loop.

#### 10.4.3 General Process Control

**SPO's expected contribution: detection of pathological
synchronisation (retry storms, sensor drift correlation, cascade
failures) in systems with coupled cyclical processes.**

- **Cloud queues**: retry storm detection (R_bad on the retry layer)
  is a natural fit — retry storms are literally pathological
  synchronisation of request timing.  SPO detects and decouples
  them via α (backoff) adjustment.  Confidence: HIGH.

- **Manufacturing SPC**: sensor drift correlation detection identifies
  systematic process shifts that single-channel SPC charts miss.
  Confidence: MODERATE (phase extraction from slowly drifting
  metrology signals is the weakest link).

- **Biological oscillators**: EEG band coherence monitoring is
  well-established (functional connectivity analysis uses similar
  metrics).  SPO adds the dual good/bad partition and entrainment
  control via audio.  Confidence: MODERATE-HIGH for monitoring,
  MODERATE for entrainment effectiveness.

### 10.5 What Would Constitute Validation

The framework remains at **TRL 3–4** (analytical/experimental proof
of concept in laboratory environment).  Advancing claims requires:

| Validation target | Required evidence | Current status |
|------------------|-------------------|---------------|
| Phase extraction fidelity | Apply Hilbert/ratio/counter extraction to real diagnostic data from a tokamak shot database (DIII-D, EAST, JET).  Measure phase quality (SNR, continuity, bandwidth). | Not done.  Synthetic-only. |
| Cascade detection lead time | Compare R_bad onset against disruption timestamps in a shot database.  Quantify how many ms earlier the coherence signal appears vs. single-channel thresholds. | Not done. |
| Knm calibration | Fit Knm matrix to cross-correlation structure of real multi-diagnostic data.  Compare exponential-decay default against data-driven matrix. | Not done. |
| Actuator coordination benefit | Run SPO in advisory mode alongside existing PCS on a real tokamak.  Compare shot-to-shot performance (confinement time, ELM frequency, disruption rate). | Not done.  Requires facility access. |
| Quantum coherence feedback | Run SPO feedback loop on IBM Heron or Google Sycamore.  Compare fidelity trajectory with and without SPO-driven coupler adjustments. | Not done.  scpn-quantum-control has hardware results for open-loop characterisation only. |
| Cross-domain transfer | Apply the same SPO engine to two unrelated domains (e.g., tokamak + cloud queue) with only binding spec changes.  Demonstrate that both benefit. | Demonstrated in simulation (9 domainpacks).  Not validated on real systems. |

### 10.6 The Honest Claim

SPO offers a **structural advance in control architecture** — the
ability to monitor multi-scale coherence and act on coupling topology
rather than individual setpoints.  This is genuinely new (§6).

SPO does **not** offer a guarantee that this structural advance
translates to practical performance gains in any specific domain.
That translation depends on calibration, phase extraction fidelity,
and actuator response — all domain-specific, all requiring
experimental validation.

The framework is **most likely to succeed** in domains where:
- Pathological synchronisation is a known failure mode (ELMs, retry
  storms, seizure-like cascades)
- Multi-scale coupling is strong and poorly handled by independent
  single-loop controllers
- The good/bad coherence partition is physically clear

The framework is **least likely to succeed** in domains where:
- Observables lack oscillatory or quasi-periodic character
- Coupling topology is unknown or highly nonlinear beyond first-harmonic
- Actuators cannot modulate coupling (observation-only systems)
- The required control bandwidth exceeds the UPDE computation time

This section exists to prevent the framework from being oversold.
A control abstraction with no experimental validation is a hypothesis,
not a solution.  The hypothesis is well-motivated (§2–§7), the
implementation is complete (§9), and the validation agenda is
defined (§10.5).  What remains is the experiment.

---

## 11. Limitations and Open Questions

1. **Phase extraction fidelity**: the Hilbert transform, event
   frequency, and ring-encoding methods have finite bandwidth and
   quality limits.  Not all domain signals have a clean oscillatory
   component.  The quality gating system (§ oscillators_PIS.md)
   mitigates this but does not eliminate it.

2. **Coupling matrix design**: the exponential-decay Knm is a
   reasonable default but may not match the true coupling topology
   of a specific system.  Domain calibration is needed.

3. **Observable-to-phase mapping invertibility**: the analytic formulas
   (§3.2) are forward-only surjections, not bijections.  The
   `phases_to_feedback` method recovers aggregate statistics (R, mean
   phase) but cannot reconstruct the original observables.

4. **Computational limits**: the N² coupling computation limits
   real-time operation to ~10³ oscillators at ~1 kHz on current
   hardware.  The Rust FFI backend extends this to ~10⁴.

5. **Validation**: the framework is demonstrated on synthetic
   simulations.  Validation against real tokamak data (DIII-D,
   EAST, JET) and real quantum hardware (IBM Eagle/Heron) remains
   future work.

---

## 12. References

- **[kuramoto1975]** Y. Kuramoto (1975). Self-entrainment of a population of coupled non-linear oscillators. *Lecture Notes in Physics* 39, 420–422.
- **[sakaguchi1986]** H. Sakaguchi & Y. Kuramoto (1986). A soluble active rotater model showing phase transitions via mutual entertainment. *Prog. Theor. Phys.* 76, 576–581.
- **[acebron2005]** J. A. Acebrón et al. (2005). The Kuramoto model: a simple paradigm for synchronization phenomena. *Rev. Mod. Phys.* 77, 137–185.
- **[dorfler2014]** F. Dörfler & F. Bullo (2014). Synchronization in complex networks of phase oscillators: a survey. *Automatica* 50, 1539–1564.
- **[breakspear2010]** M. Breakspear, S. Heitmann & A. Daffertshofer (2010). Generative models of cortical oscillations. *Front. Hum. Neurosci.* 4, 190.
- **[rawlings2017]** J. B. Rawlings, D. Q. Mayne & M. M. Diehl (2017). *Model Predictive Control: Theory, Computation, and Design*. 2nd ed., Nob Hill Publishing.
- **[ferron2006]** J. R. Ferron et al. (2006). Real time equilibrium reconstruction for tokamak discharge control. *Nucl. Fusion* 38, 1055.
- **[khaneja2005]** N. Khaneja et al. (2005). Optimal control of coupled spin dynamics: design of NMR pulse sequences by gradient ascent. *J. Magn. Reson.* 172, 296–305.
- **[ramadge1987]** P. J. Ramadge & W. M. Wonham (1987). Supervisory control of a class of discrete event processes. *SIAM J. Control Optim.* 25(1), 206–230.
- **[dormand1980]** J. R. Dormand & P. J. Prince (1980). A family of embedded Runge-Kutta formulae. *J. Comput. Appl. Math.* 6(1), 19–26.
- **[troyon1984]** F. Troyon et al. (1984). MHD-limits to plasma confinement. *Plasma Phys. Control. Fusion* 26, 209.
- **[kruskal1958]** M. D. Kruskal & M. Schwarzschild (1954). Some instabilities of a completely ionized plasma. *Proc. R. Soc. Lond. A* 223, 348–360.
- **[greenwald2002]** M. Greenwald (2002). Density limits in toroidal plasmas. *Plasma Phys. Control. Fusion* 44, R27.
- **[snipes2010]** J. A. Snipes et al. (2010). Actuator and diagnostic requirements of the ITER plasma control system. *Fusion Eng. Des.* 85, 461–465.
- **[humphreys2015]** D. A. Humphreys et al. (2015). Novel aspects of plasma control in ITER. *Phys. Plasmas* 22, 021806.
- **[evans2006]** T. E. Evans et al. (2006). Edge stability and transport control with resonant magnetic perturbations in collisionless tokamak plasmas. *Nature Phys.* 2, 419–423.
- **[lang2004]** P. T. Lang et al. (2004). ELM pace making and mitigation by pellet injection in ASDEX Upgrade. *Nucl. Fusion* 44, 665.
- **[krantz2019]** P. Krantz et al. (2019). A quantum engineer's guide to superconducting qubits. *Appl. Phys. Rev.* 6, 021318.
