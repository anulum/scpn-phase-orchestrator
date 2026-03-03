# Plasma Control Domainpack

Maps tokamak plasma dynamics to SPO's Kuramoto/UPDE framework.

## Why Kuramoto Fits This Domain

Tokamak plasmas exhibit coupled oscillations across many timescales:
micro-turbulence (ion gyro-period), zonal flows (turbulent decorrelation
time), MHD tearing modes (~ms), sawteeth/ELMs (~10 ms), and transport
barrier formation (~100 ms).  The predator-prey relationship between
turbulence and zonal flows (Diamond et al., Plasma Phys Control Fusion
2005) is a classic coupled-oscillator problem.

## Layers

| Layer | Oscillators | Channel | Purpose |
|-------|------------|---------|---------|
| micro_turbulence | 2 | P | Ion/electron-scale turbulence |
| zonal_flow | 2 | P | Zonal flow shearing rate |
| mhd_tearing | 2 | I | Tearing mode island width |
| sawtooth_elm | 2 | I | Sawtooth/ELM crash cycles |
| transport_barrier | 2 | P | ITB/ETB formation/collapse |
| current_profile | 2 | P | q-profile evolution |
| global_equilibrium | 2 | P | MHD equilibrium state |
| plasma_wall | 2 | P | Plasma-wall interaction |

## Boundaries

- **q_min** >= 1.0 (hard) -- Kruskal-Shafranov stability limit
- **beta_n** <= 2.8 (hard) -- Troyon no-wall limit
- **greenwald** <= 1.2 (hard) -- Greenwald density limit
- **lyapunov_score** >= 0.3 (soft) -- Lyapunov stability indicator

## Actuators

| Actuator | Knob | Physical Meaning |
|----------|------|-----------------|
| coupling_global | K | Inter-layer transport coupling |
| damping | zeta | Feedback stabilisation drive |

## Imprint

None. Plasma dynamics are fast relative to wall conditioning timescales.

## Scenario

200 steps: equilibrium -> turbulence onset -> ELM storm -> transport
barrier formation -> H-L back-transition -> policy recovery.

## Optional Dependency

Install `scpn-control` for direct KnmSpec import and Lyapunov verdict parsing.
