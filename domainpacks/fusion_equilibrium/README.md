# Fusion Equilibrium Domainpack

Maps Grad-Shafranov equilibrium observables to SPO's Kuramoto/UPDE framework.

## Why Kuramoto Fits This Domain

Tokamak plasma exhibits coupled oscillatory behaviour across multiple
timescales: MHD equilibrium (Alfven time), sawtooth crashes (~ms),
ELM cycles (~10 ms), and transport barrier oscillations (~100 ms).
These form a natural Kuramoto hierarchy where coupling constants
relate to plasma transport coefficients.  ITER Physics Basis (2007).

## Layers

| Layer | Oscillators | Channel | Purpose |
|-------|------------|---------|---------|
| equilibrium | 2 | P | q-profile and poloidal flux |
| stability | 2 | S | beta_N and MHD mode state |
| transport | 2 | P | Energy confinement and diffusivity |
| events | 2 | I | Sawtooth and ELM crash events |
| boundary | 2 | P | Separatrix and wall interaction |
| actuators | 2 | P | NBI and EC current drive |

## Boundaries

- **q_min** >= 1.0 (hard) -- Kruskal-Shafranov stability
- **beta_n** <= 2.8 (hard) -- Troyon no-wall limit
- **tau_e_ratio** >= 0.5 (soft) -- minimum confinement fraction

## Actuators

| Actuator | Knob | Physical Meaning |
|----------|------|-----------------|
| coupling_global | K | Inter-layer transport coupling |
| entrainment | zeta | NBI/ECCD drive strength |

## Imprint

Plasma-facing component erosion: first-wall sputtering accumulates over
campaigns, modulating coupling via impurity concentration.

## Scenario

200 steps: equilibrium setup -> beta ramp -> ELM onset -> sawtooth crash ->
transport barrier formation -> steady-state H-mode.

## Optional Dependency

Install `scpn-fusion-core` for direct equilibrium import and q-profile parsing.
