# Fusion Equilibrium Domainpack

Maps Grad-Shafranov equilibrium observables to SPO's Kuramoto/UPDE framework.

## Layers

| Layer | Oscillators | Channel | Purpose |
|-------|------------|---------|---------|
| equilibrium | eq_q, eq_psi | P | q-profile and poloidal flux |
| stability | stab_beta, stab_mhd | S | beta_N and MHD mode state |
| transport | tr_tau, tr_chi | P | Energy confinement and diffusivity |
| events | ev_saw, ev_elm | I | Sawtooth and ELM crash events |
| boundary | bnd_sep, bnd_wall | P | Separatrix and wall interaction |
| actuators | act_nbi, act_eccd | P | NBI and EC current drive |

## Observable-to-Phase Mapping

| Observable | Phase Formula |
|---|---|
| q_profile | 2*pi*(q - q_min)/(q_max - q_min) |
| beta_n | 2*pi*beta_n/beta_limit |
| tau_e | 2*pi*tau_e/tau_ref |
| sawtooth_count | count*pi mod 2*pi |
| elm_count | count*pi mod 2*pi |
| mhd_amplitude | 2*pi*amplitude/threshold |

## Boundaries

- **q_min** >= 1.0 (hard) — Kruskal-Shafranov stability
- **beta_n** <= 2.8 (hard) — Troyon no-wall limit
- **tau_e_ratio** >= 0.5 (soft) — Minimum confinement fraction

## Optional Dependency

Install `scpn-fusion-core` for direct equilibrium import and q-profile parsing.
