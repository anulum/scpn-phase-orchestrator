# Control Knobs: K, alpha, zeta, Psi

Four knobs parameterise the UPDE (Universal Phase Dynamics Equation).
The supervisor adjusts them in real time to steer coherence, suppress
pathological synchronisation, and recover from faults. Understanding
the physical and mathematical meaning of each knob is essential for
configuring the system correctly.

## The Equation

```
dtheta_i/dt = omega_i
            + sum_j K_ij sin(theta_j - theta_i - alpha_ij)
            + zeta sin(Psi - theta_i)
```

Each term:

- `omega_i` — natural frequency of oscillator i (intrinsic dynamics).
- `K_ij sin(theta_j - theta_i - alpha_ij)` — pairwise coupling from
  oscillator j to oscillator i, shifted by phase lag alpha.
- `zeta sin(Psi - theta_i)` — external periodic drive pulling
  oscillator i toward reference phase Psi.

The supervisor controls the last two terms. It cannot change omega
(that comes from the physical/informational/symbolic source), but it
can modulate how oscillators interact (K, alpha) and how an external
reference influences them (zeta, Psi).

---

## K — Coupling Strength

`K_ij` is the (i, j) entry of the coupling matrix K_nm. It controls
how strongly oscillator j pulls oscillator i toward their shared
phase relationship.

### Mathematical Properties

- **Non-negative**: `K_ij >= 0`. Negative coupling is achieved through
  phase lags (alpha), not negative K.
- **Zero diagonal**: `K_ii = 0`. An oscillator does not couple to itself.
- **Symmetric by default**: `K_ij = K_ji` unless directed coupling is
  explicitly configured (e.g., feedforward neural pathways).
- **Sparse**: for large N, only a subset of pairs have non-zero coupling.
  The `SparseUPDEEngine` uses CSR format for efficient computation.

### Construction

The `CouplingBuilder` constructs K_nm from the binding spec. Default
form for layer-structured systems:

```
K_ij = base_strength * exp(-decay_alpha * |layer_i - layer_j|)
```

For SCPN-calibrated systems, the builder uses empirically validated
anchors stored in `SCPN_CALIBRATION_ANCHORS` — see the
[knm_calibration](../specs/knm_calibration.md) spec.

### Effect on Dynamics

- **Increasing K globally** raises the order parameter R (stronger
  synchronisation). Above a critical coupling `K_c`, the system
  transitions from incoherence to partial synchrony (Kuramoto
  transition).
- **Decreasing K globally** allows oscillators to free-run at their
  natural frequencies. R drops toward the incoherent baseline.
- **Layer-selective K changes** can synchronise one functional group
  while leaving others decoupled. This is the primary mechanism for
  layer-specific regime control.

### Critical Coupling

The critical coupling strength `K_c` depends on the frequency
distribution. For Lorentzian-distributed natural frequencies with
half-width gamma:

```
K_c = 2 * gamma
```

For general distributions, `find_critical_coupling()` in the
`upde.bifurcation` module computes K_c numerically via bisection.

### Supervisor Use

| Regime | K adjustment |
|--------|-------------|
| NOMINAL | No change. K set by binding spec. |
| DEGRADED | Boost global K to recover sync. |
| CRITICAL | Reduce K on affected layers to prevent cascading lock-in. |
| RECOVERY | Gradually restore K toward nominal values. |

---

## alpha — Phase Lag

`alpha_ij` shifts the coupling function. It determines the preferred
phase relationship between oscillators i and j. This is the
Sakaguchi-Kuramoto extension of the original Kuramoto model.

### Physical Interpretation

- `alpha = 0`: coupling is purely attractive. Oscillators prefer
  in-phase synchrony (theta_i = theta_j).
- `alpha = pi/4`: oscillators prefer a 45-degree phase lead/lag.
- `alpha = pi/2`: coupling becomes purely imaginary — no net
  synchronising force. This is the "frustrated" state.
- `alpha > pi/2`: coupling becomes repulsive. Oscillators are pushed
  apart, which can model competitive or inhibitory interactions.

### Sources of Phase Lag

1. **Propagation delays**: signal travel time between spatially
   separated oscillators (e.g., axonal conduction delay in neural
   systems, network latency in distributed computing).
2. **Processing delays**: time to extract phase from raw signals
   (P-channel Hilbert transform introduces half-cycle delay at
   the filter edge).
3. **Imprint modulation**: the memory imprint model can shift alpha
   based on cumulative exposure history, encoding long-term adaptation.

### Lag Matrix

Alpha is stored as a matrix with the same shape and sparsity pattern
as K_nm. The `LagModel` class constructs and manages the lag matrix:

```python
lag_model = LagModel(n=8)
lag_model.set_uniform(alpha=0.1)    # all pairs
lag_model.set_layer_pair(2, 5, 0.3) # specific pair
alpha_matrix = lag_model.build()
```

### Supervisor Use

| Regime | alpha adjustment |
|--------|-----------------|
| DEGRADED | Increase alpha on pathological layers to break lock-in. |
| CRITICAL | Set alpha = pi/2 on affected pairs to fully decouple. |
| RECOVERY | Gradually reduce alpha toward nominal. |

---

## zeta — Driver Strength

Scalar strength of the external periodic drive. Pulls all oscillators
toward the reference phase Psi.

### Dynamics

The drive term `zeta * sin(Psi - theta_i)` acts uniformly on all
oscillators. Its effect depends on the phase difference between each
oscillator and the reference:

- When `theta_i ≈ Psi`: `sin(Psi - theta_i) ≈ 0`. No force (already
  aligned).
- When `theta_i` leads Psi: positive force (slow down).
- When `theta_i` lags Psi: negative force (speed up).

This is identical to a single "pacemaker" oscillator coupled to all
others with uniform strength zeta.

### Values

- `zeta = 0`: free-running dynamics. No external influence.
- `zeta = 0.1–0.5`: gentle entrainment. Network partially follows
  the reference while maintaining internal dynamics.
- `zeta = 1.0–2.0`: strong entrainment. Network locked to reference.
- `zeta > K_mean`: external drive dominates coupling. All oscillators
  converge to Psi regardless of inter-oscillator dynamics.

### Supervisor Use

| Regime | zeta adjustment |
|--------|----------------|
| NOMINAL | zeta = 0 (free-running). |
| DEGRADED | Small zeta to nudge toward target coherence. |
| CRITICAL | Large zeta to damp oscillators toward known stable phase. |
| RECOVERY | Gradually reduce zeta as internal coupling stabilises. |

### Active Inference Controller

The `ActiveInferenceAgent` sets zeta and Psi based on Friston's
Variational Free Energy Principle. It minimises prediction error
between observed R and target R:

```python
agent = ActiveInferenceAgent(n_hidden=4, target_r=0.8, lr=1.0)
zeta, psi = agent.control(r_obs=0.3, psi_obs=1.5, dt=0.01)
```

If `R_obs < target`: agent increases zeta and aligns Psi to encourage
synchronisation. If `R_obs > target`: agent increases zeta but
anti-aligns Psi (shifts by pi) to suppress excess sync.

---

## Psi — Reference Phase

The target phase for the external drive. Only effective when zeta > 0.

### Interpretation by Domain

- **Neuroscience**: Psi can represent a circadian rhythm reference,
  a sensory stimulus phase, or a motor command target.
- **Engineering**: Psi represents a clock signal, PLL reference, or
  coordination target in distributed systems.
- **Finance**: Psi can encode a market cycle reference for regime-aware
  trading.

### Wrapping

Psi is always in `[0, 2*pi)`. Setting Psi outside this range is
valid — it is wrapped modulo 2*pi internally. The coupling term
`sin(Psi - theta_i)` handles wrapping implicitly.

### Supervisor Use

| Regime | Psi adjustment |
|--------|---------------|
| NOMINAL | Not used (zeta = 0). |
| RECOVERY | Set to mean phase of healthy oscillators for guided re-entrainment. |
| CRITICAL | Set to the last known stable mean phase. |

---

## Scope and Actuation

Each `ControlAction` specifies which knob to change, what value, and
at what scope:

```python
ControlAction(
    knob="K",
    scope="layer_3",
    value=2.5,
    ttl_s=10.0,
    justification="R_3 below threshold, boosting coupling"
)
```

### Scope Resolution

- `"global"` — applies to all oscillators / all matrix entries.
- `"layer_{n}"` — applies to oscillators in SCPN hierarchy layer n.

The `ActuationMapper` resolves scope to specific matrix indices. The
`ActionProjector` clips values to bounds and enforces rate limits
before the action is applied to the integration engine.

### Actuation Range and Limits

Each run applies controls through two guardrails:

1. **Domainpack actuator limits** from `binding_spec.yaml` (`actuators[].limits`)
2. **Action projector defaults** in the CLI when actuator limits are missing

Domainpack defaults in the current roadmap baseline are typically:

| Knob | Common bound | Typical rate limit |
|------|--------------|-------------------|
| K | [0, 5.0] | 0.1 per step |
| alpha | [−π, π] | 0.1 per step |
| zeta | [0, 2.0] | 0.2 per step |
| Psi | [0, 2π) | 0.5 per step |

The CLI fallback (used when limits are not specified in the binding spec) is:

| Knob | Fallback bound | Rate limit |
|------|----------------|-----------|
| K | [−0.5, 0.5] | 0.1 |
| alpha | [−1.0, 1.0] | 0.1 |
| zeta | [0.0, 0.5] | 0.2 |
| Psi | [−∞, ∞] | 0.5 |

`ActionProjector` applies both value clipping and ramp limiting before
the action reaches the engine.

## Delay and Lag

In many domains coupling is not instantaneous. The delayed model uses:

```text
dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j(t - τ_ij) - θ_i(t) - α_ij) + ζ sin(Ψ - θ_i)
```

where `τ_ij` is a transport delay. This appears in the delayed engine as
state lookup from a circular buffer.

- **Small delay** shifts the effective coupling phase and often behaves
  like an extra lag component.
- **Larger delay** can stabilise or destabilise depending on network
  structure and coupling strength.

Use delayed dynamics when transport or inference latency is a governing
feature of the physical process you model.

## Coupling Priors

Before policy control starts, base coupling shape still comes from priors:

- `binding_spec.coupling.base_strength` and `decay_alpha`
- `UniversalPrior` defaults (`K_base_mean = 0.47`, `decay_alpha_mean = 0.25`)
  for Bayesian auto-tune starting points

Why this matters:

- It defines a safe initial operating regime.
- It determines the starting point for any `K`/`alpha` trajectory.
- It reduces brittle over-fitting to one particular topology.

For more detail, see [coupling_prior.md](../reference/api/coupling_prior.md).

## Damping

`damping` is the rate-loss parameter in inertial models:

```text
dω/dt = (P - Dω + coupling + injections) / M
```

where `D` is damping and `M` is inertia. Higher damping damps oscillatory
ringing after disturbances; lower damping responds faster but can oscillate.

SPO currently routes this through domainpack-specific actuator wiring.
In practice this is often declared as a named actuator (for example,
`damping`) while still using the existing action contract and mapped to
domain physics.

## Supervisor Thresholds

Supervisor thresholds are defined in code and tuned to finite-N dynamics:

- `_R_CRITICAL = 0.3` (`supervisor/regimes.py`)
- `_R_DEGRADED = 0.6`
- `hysteresis = 0.05`
- `cooldown_steps = 10`

Policy action tuning values are:

- `_K_BUMP = 0.05`
- `_ZETA_BUMP = 0.1`
- `_K_REDUCE = -0.03`
- `_RESTORE_FRACTION = 0.5`

These numbers appear in the default `SupervisorPolicy` and `RegimeManager`.

## Actuation Limits in Practice

For policy-engine actions, the order is:

1. Validate action schema.
2. Resolve scope and matching actuators.
3. Apply rate and value projection.
4. Emit actuator commands to domain runtime.

The explicit actuator table in `binding_spec.yaml` is therefore the
first source of truth for operator-facing limits.

## Interaction Between Knobs

The four knobs are not independent. Their combined effect determines
the system's dynamical regime:

- **K high, alpha low, zeta = 0**: strong synchronisation, free-running.
  Classical Kuramoto near or above K_c.
- **K high, alpha = pi/2**: frustrated coupling. No net sync despite
  strong interaction. Used to deliberately suppress pathological lock.
- **K low, zeta high**: externally driven. Network follows the
  reference. Internal coupling negligible.
- **K moderate, zeta moderate**: mixed regime. Internal coupling and
  external drive compete. Rich dynamics including multistability and
  chimera states.

The supervisor's challenge is navigating this four-dimensional control
space to maintain target coherence while respecting safety boundaries.
The Active Inference controller automates this for the zeta/Psi pair;
K and alpha adjustments are policy-driven.

## Petri Net Representation

The `PetriNetAdapter` maps knob adjustments to Petri net transitions.
Each knob has an "increase" and "decrease" transition gated by the
current regime:

- **Place `regime_nominal`** enables only small K adjustments.
- **Place `regime_critical`** enables large zeta increases and K
  reductions.
- **Inhibitor arcs** prevent contradictory actions (e.g., increasing
  K while simultaneously increasing alpha on the same layer).

This formalises the control policy as a verifiable state machine,
enabling model-checking of liveness and safety properties before
deployment.

## References

- **[sakaguchi1986]** H. Sakaguchi & Y. Kuramoto (1986). A soluble active rotater model showing phase transitions via mutual entertainment. *Prog. Theor. Phys.* 76, 576-581. — Phase-lag (alpha) coupling model.
- **[dorfler2014]** F. Dorfler & F. Bullo (2014). Synchronization in complex networks of phase oscillators: a survey. *Automatica* 50, 1539-1564. — Coupling strength (K) and synchronisation conditions.
- **[friston2010]** K. J. Friston (2010). The free-energy principle: a unified brain theory? *Nature Rev. Neuroscience* 11, 127-138. — Active Inference for zeta/Psi control.
- **[acebron2005]** J. A. Acebron et al. (2005). The Kuramoto model: a simple paradigm for synchronization phenomena. *Rev. Mod. Phys.* 77, 137-185. — Critical coupling and order parameter transitions.
