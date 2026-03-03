# Metaphysics Demo Domainpack

Exercises all three oscillator channels (P/I/S), imprint modulation,
geometry projection, and policy-driven control in a single run.

## Why Kuramoto Fits This Domain

Reference implementation demonstrating the full SPO pipeline.  Three
channels (Physical, Informational, Symbolic) show that any cyclic
process -- continuous, event-driven, or discrete-state -- maps onto
Kuramoto dynamics.  The ablation run quantifies imprint contribution.

## Layers

| Layer | Oscillators | Channel | Purpose |
|-------|------------|---------|---------|
| physical | 3 | P (hilbert) | Continuous waveform oscillators |
| informational | 2 | I (event) | Event-driven phase oscillators |
| symbolic | 2 | S (symbolic) | Discrete state-sequence oscillators |

## Boundaries

- **R_floor**: R >= 0.1 (hard) -- minimum global coherence
- **R_2_ceiling**: R_2 <= 0.9 (soft) -- symbolic layer cap

## Actuators

| Actuator | Knob | Physical Meaning |
|----------|------|-----------------|
| coupling_global | K | Global coupling strength |
| phase_target | Psi | Reference phase target |
| damping | zeta | External drive / damping |

## Imprint

Demonstration imprint: decay_rate=0.01, saturation=2.0, modulates K and
alpha.  Ablation run (imprint ON vs OFF) quantifies the effect.

## Scenario

200 steps (x2 ablation): baseline -> perturbation -> policy response ->
recovery.  Comparison plot saved to `ablation.png` with matplotlib.
