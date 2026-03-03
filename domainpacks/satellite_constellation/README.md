# Satellite Constellation Domainpack

Phase-locking of LEO satellite nodes for coordinated beam-forming.

## Why Kuramoto Fits This Domain

Coordinated satellite beam-forming requires tight phase alignment across
orbital slots and communication links.  Each satellite's local oscillator
drifts due to Doppler shifts and orbital mechanics; Kuramoto coupling
models the distributed phase-lock loops that maintain coherent beam
steering across the constellation.

## Layers

| Layer | Oscillators | Channel | Purpose |
|-------|------------|---------|---------|
| orbit_mech | 2 | P (hilbert) | Orbital slot phase references |
| comms_link | 3 | I (event) | Inter-satellite link timing |
| beam_sync | 3 | P (hilbert) | Beam-forming phase alignment |

## Boundaries

- **doppler_shift_max**: < 0.8 (soft) -- Doppler-induced frequency offset
- **link_budget_min**: > 0.3 (hard) -- minimum inter-satellite link margin
- **handover_gap_max**: < 0.5 (soft) -- beam handover timing gap

## Actuators

| Actuator | Knob | Physical Meaning |
|----------|------|-----------------|
| global_coupling | K | Phase-lock loop gain |
| beam_steering | zeta | Beam steering drive strength |

## Scenario

200 steps: orbital lock-up -> beam synchronisation -> orbital slot drift
perturbation at step 100 -> recovery via coupling boost.
