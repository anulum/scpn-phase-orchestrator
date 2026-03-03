# Power Grid Domainpack

Maps interconnected power system dynamics to SPO's Kuramoto/UPDE framework.

## Why Kuramoto Fits This Domain

The swing equation `dδ/dt = ω, dω/dt = (P_m - P_e - Dω)/(2H)` is a
second-order Kuramoto model.  PMU voltage phasor angle IS the oscillator
phase — no extraction needed.  Coupling constant equals line admittance.

Dorfler, Chertkov, Bullo (2013) "Synchronization in complex oscillator
networks and smart grids" proved this equivalence rigorously.

## Layers

| Layer | Oscillators | Channel | Purpose |
|-------|------------|---------|---------|
| generator_rotor | 3 | P | Machine rotor angles |
| area_frequency | 2 | P | Control area deviations |
| tie_line | 2 | I | Inter-area power flows |
| load_demand | 3 | I | Aggregate load fluctuations |
| renewable_intermittency | 2 | I | Solar/wind variability |

## Boundaries

- **frequency_deviation**: ±0.5 Hz (hard) — NERC BAL-003-2
- **voltage_magnitude**: 0.95–1.05 pu (hard) — ANSI C84.1
- **rotor_angle**: < 90° (hard) — transient stability limit

## Actuators

| Actuator | Knob | Physical Meaning |
|----------|------|-----------------|
| governor_droop | K | Governor droop gain |
| agc_bias | zeta | AGC frequency bias |
| load_shedding | alpha | Under-frequency load shed |
| renewable_curtailment | Psi | Curtailment anti-phase signal |

## Scenario

250 steps: steady-state → sudden load step → renewable ramp → generator
trip fault → AGC + policy restore synchronism.
