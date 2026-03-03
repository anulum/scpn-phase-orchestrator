# Chemical Reactor Domainpack

Maps CSTR (continuous stirred-tank reactor) oscillation dynamics to SPO's
Kuramoto/UPDE framework.

## Why Kuramoto Fits This Domain

CSTR systems undergo Hopf bifurcations where concentration and temperature
oscillate with well-defined phase relationships.  The heat-mass coupling
(Arrhenius rate ↔ heat generation) creates limit cycles that are naturally
modelled as coupled oscillators.  Coolant-jacket dynamics add a second
timescale whose phase relationship to the reactor determines stability.

Fogler (2020) "Elements of Chemical Reaction Engineering" Ch. 12.

## Layers

| Layer | Oscillators | Channel | Purpose |
|-------|------------|---------|---------|
| reaction_kinetics | 3 | P (hilbert) | Concentration oscillations (Hopf) |
| heat_transfer | 3 | P (hilbert) | Reactor/jacket/coolant temperatures |
| pressure_vessel | 2 | I (event) | Headspace pressure + vent |
| feed_flow | 2 | I (event) | Feed and coolant flow rates |

## Boundaries

- **temperature_limit**: < 450°C (hard) — Semenov ignition
- **pressure_limit**: < 15 bar (hard) — ASME Section VIII MAWP
- **concentration_floor**: > 0.1 (soft) — reactant depletion
- **coolant_flow_min**: > 10 LPM (soft) — minimum cooling

## Actuators

| Actuator | Knob | Physical Meaning |
|----------|------|-----------------|
| coolant_flow | K | Coolant valve position |
| feed_rate | zeta | Feed pump speed |
| agitator_speed | alpha | Mixing intensity |
| jacket_setpoint | Psi | Jacket temperature SP |

## Imprint

Catalyst deactivation / fouling: very slow decay (months timescale)
modulates coupling as active surface area decreases.

## Scenario

250 steps: stable CSTR → Hopf bifurcation onset → coolant failure
transient → emergency quench → feed rate recovery with fouling.
