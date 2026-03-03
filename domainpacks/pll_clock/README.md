# PLL Clock Domainpack

Maps phase-locked loop network clock synchronisation to SPO's Kuramoto/UPDE framework.

## Why Kuramoto Fits This Domain

A PLL's phase detector computes sin(theta_ref - theta_vco) -- the exact
Kuramoto coupling function.  A network of PLLs hierarchically slaved to
a reference is literally a Kuramoto network with asymmetric coupling.
Strogatz & Mirollo, SIAM J Appl Math 1988; ITU-T G.811 stratum hierarchy.

## Layers

| Layer | Oscillators | Channel | Purpose |
|-------|------------|---------|---------|
| local_vco | 4 | P (physical) | Voltage-controlled oscillator phases |
| network_pll | 2 | P (physical) | Master/slave PLL lock indicators |
| stratum_hierarchy | 2 | I (event) | Stratum 1/2 reference status |

## Boundaries

- **phase_error**: < 100 ns (hard) -- IEEE 1588 precision time protocol
- **freq_drift**: < 10 ppm (hard) -- ITU-T G.811 stratum 1 spec
- **holdover**: < 1000 s (soft) -- maximum holdover before re-acquisition

## Actuators

| Actuator | Knob | Physical Meaning |
|----------|------|-----------------|
| loop_bandwidth | K | PLL loop filter bandwidth |
| frequency_trim | alpha | VCO frequency offset trim |
| reference_drive | zeta | External reference injection |
| phase_target | Psi | Target lock phase |

## Imprint

Crystal aging: quartz oscillator frequency drift (ppm/year) accumulates
and modulates both coupling and lag, representing long-term oscillator aging.

## Scenario

200 steps: locked to reference -> reference loss (holdover) -> VCO drift ->
PLL re-acquisition -> phase step -> recovery.
