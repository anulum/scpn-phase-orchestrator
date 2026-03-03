# Swarm Robotics Domainpack

Maps collective robot motion to SPO's Kuramoto/UPDE framework.

## Why Kuramoto Fits This Domain

The Vicsek model (PRL 75(6), 1995) of collective motion updates each
agent's heading by averaging neighbours' headings plus noise -- a
discrete-time Kuramoto model on the unit circle.  Cucker & Smale
(IEEE TAC 2007) proved flocking emergence under decaying communication
kernels, directly analogous to distance-decayed Kuramoto coupling.

## Layers

| Layer | Oscillators | Channel | Purpose |
|-------|------------|---------|---------|
| individual_heading | 4 | P (physical) | Robot heading angles |
| formation_shape | 2 | I (event) | Formation radius/angle indicators |
| flock_direction | 2 | I (event) | Global flock heading/speed |

## Boundaries

- **formation_error**: < 2.0 m (hard) -- maximum deviation from target shape
- **collision_dist**: > 0.5 m (hard) -- minimum inter-robot distance
- **heading_variance**: < 1.0 rad (soft) -- heading alignment quality

## Actuators

| Actuator | Knob | Physical Meaning |
|----------|------|-----------------|
| alignment_coupling | K | Vicsek alignment strength |
| formation_drive | zeta | Formation-keeping drive |
| obstacle_avoidance | alpha | Repulsive heading offset near obstacles |
| target_heading | Psi | Global waypoint heading |

## Imprint

None. Stateless heading dynamics with no memory effects.

## Scenario

200 steps: random headings -> alignment emergence -> obstacle breaks formation ->
re-form after obstacle -> leader failure -> swarm self-recovery.
