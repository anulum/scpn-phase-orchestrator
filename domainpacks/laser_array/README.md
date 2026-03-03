# Laser Array Domainpack

Maps semiconductor laser array phase-locking to SPO's Kuramoto/UPDE framework.

## Why Kuramoto Fits This Domain

Evanescent waveguide coupling between adjacent ridge lasers implements
sin(phi_m - phi_n) coupling directly in the optical field.  Phase-locking
of laser arrays is one of the foundational applications of coupled-oscillator
theory in photonics.  Winful & Wang, Appl Phys Lett 53(20), 1988;
Kozyreff et al., PRL 85(18), 2000.

## Layers

| Layer | Oscillators | Channel | Purpose |
|-------|------------|---------|---------|
| single_laser | 4 | P (hilbert) | Individual laser optical fields |
| array_coupling | 2 | P (hilbert) | Evanescent inter-guide coupling modes |
| external_cavity | 2 | I (event) | External feedback cavity modes |

## Boundaries

- **phase_variance**: < 0.3 rad (hard) -- coherent beam combining threshold
- **feedback_strength**: < 0.5 (hard) -- coherence collapse onset
- **power_imbalance**: < 20% (soft) -- near-field uniformity

## Actuators

| Actuator | Knob | Physical Meaning |
|----------|------|-----------------|
| injection_current | zeta | Pump current above threshold |
| evanescent_coupling | K | Inter-guide coupling via ridge spacing |
| detuning_offset | alpha | Individual laser frequency detuning |
| feedback_phase | Psi | External cavity round-trip phase |

## Imprint

Mirror degradation: facet reflectivity loss from COD (catastrophic optical
damage) accumulates and modulates coupling strength.

## Scenario

200 steps: phase-locked array -> detuning sweep -> external feedback onset ->
coherence collapse -> policy recovery.
