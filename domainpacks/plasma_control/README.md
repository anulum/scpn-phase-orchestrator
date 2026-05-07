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
| lag_turbulence | alpha | Turbulence phase-lag shaping |
| damping_global | zeta | Feedback stabilisation drive |

## Value-Alignment Guard

The binding spec includes a `value_alignment` template for review-time plasma
control checks. It bounds inter-layer transport coupling, turbulence phase-lag
changes, and global damping-drive changes, then falls back to a damping hold
when a candidate action exceeds those priors.

This template is for simulation, replay, and policy review. It is not a live
tokamak protection system.

## Imprint

None. Plasma dynamics are fast relative to wall conditioning timescales.

## Scenario

200 steps: equilibrium -> turbulence onset -> ELM storm -> transport
barrier formation -> H-L back-transition -> policy recovery.

## Higher-Order Topology Demo

`topology_adaptation_demo.py` demonstrates the supervisor-side higher-order
topology editor on this domainpack without applying live actuation. It builds
the plasma coupling matrix from `binding_spec.yaml`, evaluates a deterministic
low-global-coherence phase state, and emits an audit payload for proposed
triadic simplices.

The demo uses `TopologyMutationPolicy.simplex_pairwise_support_floor` so a
new 2-simplex is only proposed when every supporting pairwise edge is already
above the configured coupling floor. This keeps the higher-order topology
mutation reviewable and prevents unsupported triads from appearing only
because three phases are momentarily aligned.

Run:

```bash
PYTHONPATH=src python domainpacks/plasma_control/topology_adaptation_demo.py
```

## Morphogenetic Field Demo

`morphogenetic_field_demo.py` demonstrates reaction-diffusion-style topology
field shaping for an edge-localised stress replay. Transport-barrier,
current-profile, and global-equilibrium layers remain locally aligned while
micro-turbulence, tearing, ELM, and wall-interaction phases stress the field.

Run:

```bash
PYTHONPATH=src python domainpacks/plasma_control/morphogenetic_field_demo.py
```

The replay is non-actuating. It validates the binding spec, builds the
configured layer coupling, reports grown and shrunk topology-field edges, and
exports dependency-free field snapshot rows for audit or later UI rendering.

## Optional Dependency

Install `scpn-control` for direct KnmSpec import and Lyapunov verdict parsing.
