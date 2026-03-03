# Plasma Control Domainpack

Maps tokamak plasma dynamics to SPO's Kuramoto/UPDE framework.

## Layers

| Layer | Oscillators | Channel | Purpose |
|-------|------------|---------|---------|
| micro_turbulence | turb_0–1 | P | Ion/electron-scale turbulence |
| zonal_flow | zonal_0–1 | P | Zonal flow shearing |
| mhd_tearing | mhd_0–1 | I | Tearing mode activity |
| sawtooth_elm | saw_0–1 | I | Sawtooth/ELM crashes |
| transport_barrier | tb_0–1 | P | ITB/ETB formation |
| current_profile | cp_0–1 | P | q-profile evolution |
| global_equilibrium | eq_0–1 | P | MHD equilibrium |
| plasma_wall | pw_0–1 | P | Plasma-wall interaction |

## Physics Boundaries

- **q_min** >= 1.0 (hard) — Kruskal-Shafranov stability limit
- **beta_n** <= 2.8 (hard) — Troyon no-wall limit
- **greenwald** <= 1.2 (hard) — Greenwald density limit
- **lyapunov_score** >= 0.3 (soft) — Lyapunov stability indicator

## Policy

- `suppress_elm_storm`: decouple turbulence layer when ELM activity is high
- `restore_transport_barrier`: boost global coupling when transport barrier degrades

## Optional Dependency

Install `scpn-control` for direct KnmSpec import and Lyapunov verdict parsing.
