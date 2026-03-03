# Firefly Swarm Domainpack

Maps firefly flash synchronisation to SPO's Kuramoto/UPDE framework.

## Why Kuramoto Fits This Domain

Firefly flash synchronisation is the canonical biological example of
pulse-coupled oscillators.  Each firefly has an intrinsic flash period;
visual coupling advances the phase of neighbours upon flash detection.
Mirollo & Strogatz (SIAM J Appl Math 1990) proved global synchrony
emerges for all-to-all pulse coupling.  Buck (Q Rev Biol 1988) documented
Pteroptyx malaccae achieving millisecond-precision synchrony in swarms
of thousands.

## Layers

| Layer | Oscillators | Channel | Purpose |
|-------|------------|---------|---------|
| individual_flash | 6 | P (hilbert) | Individual firefly flash cycles |
| swarm_coherence | 2 | I (event) | Cluster-level synchrony indicators |

## Boundaries

- **flash_variance**: < 0.5 s (soft) -- inter-flash timing spread
- **swarm_density**: > 0.1 (soft) -- minimum spatial density for visual coupling

## Actuators

| Actuator | Knob | Physical Meaning |
|----------|------|-----------------|
| visual_coupling | K | Line-of-sight coupling strength |
| environmental_light | zeta | Ambient light (suppresses flash visibility) |
| flash_target | Psi | Artificial pacer flash phase |

## Imprint

None. Firefly flash coupling is memoryless on relevant timescales.

## Scenario

200 steps: random flashing -> gradual synchronisation -> disturbance splits
swarm into two clusters -> re-emergence of global sync -> full synchrony.
