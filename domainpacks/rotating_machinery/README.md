# Rotating Machinery Domainpack

Maps rotor dynamics and vibration monitoring to SPO's Kuramoto/UPDE framework.

## Why Kuramoto Fits This Domain

Rotating machinery vibration is fundamentally a coupled oscillator problem.
Shaft harmonics (1X, 2X, 3X), bearing defect frequencies (BPFI, BPFO),
blade-pass frequency, and structural modes all interact through mechanical
coupling (shared shaft, bearing stiffness, casing).  Phase relationships
between these harmonics diagnose fault type: 1X+2X in-phase = misalignment,
specific bearing frequencies = inner/outer race defect.

## Layers

| Layer | Oscillators | Channel | Freq (×RPM) | Purpose |
|-------|------------|---------|-------------|---------|
| shaft_rotation | 3 | P (hilbert) | 1X/2X/3X | Running speed harmonics |
| blade_dynamics | 2 | P (hilbert) | 5X/0.3X | Blade pass + subsync flutter |
| bearing_condition | 3 | I (event) | 1.68/1.2/0.07 | BPFI/BPFO/FTF (SKF 6205) |
| structural_resonance | 2 | I (event) | 3.6X/7.2X | FEA-derived mode shapes |

## Boundaries

- **vibration_velocity**: < 7.1 mm/s (hard) — ISO 10816-3 zone C/D
- **shaft_displacement**: < 100 um (soft) — API 670
- **bearing_temperature**: < 105°C (hard)

## Actuators

| Actuator | Knob | Physical Meaning |
|----------|------|-----------------|
| speed_setpoint | zeta | VFD speed reference |
| bearing_stiffness | K | Support stiffness (pedestal adjustment) |
| damper_viscosity | alpha | Squeeze-film damper viscosity |

## Imprint

Bearing wear: ISO 15243 spalling progression accumulates and modulates
mechanical coupling, representing gradual bearing degradation.

## Scenario

200 steps: cold start ramp -> 1st critical speed crossing -> nominal speed ->
bearing degradation onset -> blade flutter event -> policy response.
