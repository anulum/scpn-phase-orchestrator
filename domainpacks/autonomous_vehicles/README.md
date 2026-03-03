# Autonomous Vehicles Domainpack

Platoon coordination via phase-coupled vehicle controllers.

## Why Kuramoto Fits This Domain

Vehicle platooning requires followers to maintain tight phase alignment
with the leader's speed and braking signals.  Each vehicle's throttle/brake
controller acts as an oscillator coupled to its neighbours.  Kuramoto
coupling models the inter-vehicle communication links (V2V) that propagate
braking and acceleration signals through the platoon.

## Layers

| Layer | Oscillators | Channel | Purpose |
|-------|------------|---------|---------|
| leader | 1 | P (hilbert) | Lead vehicle speed reference |
| followers | 4 | P (hilbert) | Following vehicles' throttle phase |
| road_env | 3 | I (event) | Road condition oscillators (friction, slope, traffic) |

## Boundaries

- **gap_distance_min**: > 0.2 (hard) -- minimum inter-vehicle gap
- **speed_delta_max**: < 0.5 (soft) -- max speed difference across platoon
- **brake_reaction_max**: < 0.3 (hard) -- max brake propagation delay

## Actuators

| Actuator | Knob | Physical Meaning |
|----------|------|-----------------|
| platoon_coupling | K | V2V communication coupling strength |
| throttle_drive | zeta | Throttle response drive |

## Scenario

200 steps: platoon formation -> cruise -> leader emergency brake at step 120
-> followers recover phase alignment.
