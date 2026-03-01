# Control Knobs: K, alpha, zeta, Psi

Four knobs parameterise the UPDE. The supervisor adjusts them to steer coherence.

## The Equation

```
dtheta_i/dt = omega_i
            + sum_j K_ij sin(theta_j - theta_i - alpha_ij)
            + zeta sin(Psi - theta_i)
```

## K -- Coupling Strength

`K_ij` is the (i,j) entry of the Knm matrix. Controls how strongly oscillator j pulls oscillator i.

- Increasing K raises synchrony (R increases).
- Decreasing K allows oscillators to decouple.
- Matrix is symmetric, non-negative, zero diagonal.
- Built from `base_strength * exp(-decay_alpha * |i-j|)` by default.

**Supervisor use:** Boost global K when R_good is low (DEGRADED regime). Reduce K on specific layers to suppress R_bad.

## alpha -- Phase Lag

`alpha_ij` shifts the coupling function. At `alpha=0`, coupling is purely attractive. At `alpha=pi/2`, coupling becomes repulsive (Sakaguchi-Kuramoto model).

- Models transport delays, propagation latencies.
- Non-zero alpha creates phase-locked states with constant offset.
- The imprint model can shift alpha based on exposure history.

**Supervisor use:** Increase alpha on bad layers to desynchronise pathological lock-in (retry storms, seizure-like cascades).

## zeta -- Driver Strength

Scalar strength of the external periodic drive `zeta sin(Psi - theta_i)`. Pulls all oscillators toward the reference phase.

- `zeta=0`: free-running (no external drive).
- `zeta>0`: entrainment toward Psi. Higher zeta = stronger pull.
- Used for pacing, reference tracking, or forced entrainment.

**Supervisor use:** Increase zeta during CRITICAL regime to damp oscillators toward a known stable phase.

## Psi -- Reference Phase

The target phase for the external drive. Only effective when `zeta > 0`.

- In biological systems, Psi can represent a circadian reference.
- In engineering, Psi can represent a clock signal or coordination target.

**Supervisor use:** Adjusted during RECOVERY regime to guide re-entrainment.

## Scope

Each `ControlAction` specifies a scope:

- `"global"` -- applies to all oscillators.
- `"layer_{n}"` -- applies to oscillators in hierarchy layer n.

The `ActuationMapper` resolves scope to specific actuator commands. The `ActionProjector` clips values to bounds and enforces rate limits.

## Constraints

| Knob | Typical range | Rate limit |
|------|--------------|------------|
| K | [0, 5.0] | 0.1 per step |
| alpha | [-pi, pi] | 0.05 per step |
| zeta | [0, 2.0] | 0.2 per step |
| Psi | [0, 2*pi) | no limit |

Ranges and rate limits are domain-specific, configured in the binding spec `actuators` section.
