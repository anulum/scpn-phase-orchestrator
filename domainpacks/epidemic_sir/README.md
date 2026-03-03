# Epidemic SIR Domainpack

Maps epidemic wave dynamics to SPO's Kuramoto/UPDE framework.

## Why Kuramoto Fits This Domain

Epidemic waves oscillate with well-defined periods driven by seasonal
forcing, immunity waning, and intervention cycles.  The SIR model
produces damped oscillations that are naturally phase-coupled across
regions via mobility.  Seasonal forcing acts as external drive (zeta).
Earn et al. (2000) demonstrated Kuramoto-like synchronisation in measles
epidemics across cities.

## Layers

| Layer | Oscillators | Channel | Period | Purpose |
|-------|------------|---------|--------|---------|
| infection_wave | 3 | P (hilbert) | ~2-3 weeks | S, I, R compartments |
| intervention | 3 | I (event) | weeks-months | Vaccination, distancing, treatment |
| mobility | 2 | I (event) | days | Local movement, inter-region travel |

## Boundaries

- **case_rate**: < 100/100k (hard) -- WHO threshold
- **hospital_occupancy**: < 80% (hard) -- healthcare capacity
- **reproduction_number**: Rt < 1.5 (soft) -- transmission control

## Actuators

| Actuator | Knob | Physical Meaning |
|----------|------|-----------------|
| social_measures | zeta | NPI drive strength |
| vaccination_rate | K | Intervention coordination coupling |
| mobility_restriction | alpha | Travel restriction phase lag |
| lockdown | Psi | Lockdown target phase |

## Imprint

Waning immunity: antibody titres decay over months post-infection or
vaccination (Antia et al. 2018), modulating inter-region coupling as
population susceptibility rises.

## Scenario

250 steps: endemic baseline -> imported case surge -> community transmission
wave -> NPI intervention -> vaccination campaign -> suppression.
