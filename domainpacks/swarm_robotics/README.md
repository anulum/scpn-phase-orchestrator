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

## Value-Alignment Guard

The binding spec includes a `value_alignment` template for review-time swarm
actuation checks. It bounds alignment coupling, formation drive, obstacle
avoidance, and target-heading changes, then falls back to a zero-formation-drive
hold when a candidate action exceeds those priors.

This template is for simulation, replay, and policy review. It is not a live
multi-robot safety controller.

## Imprint

None. Stateless heading dynamics with no memory effects.

## Scenario

200 steps: random headings -> alignment emergence -> obstacle breaks formation ->
re-form after obstacle -> leader failure -> swarm self-recovery.

## Morphogenetic Field Demo

`morphogenetic_field_demo.py` demonstrates reaction-diffusion-style topology
field shaping for this domainpack without live actuation. It builds the swarm
coupling matrix from `binding_spec.yaml`, evaluates a deterministic split-flock
phase state, and emits:

- the next reviewable `K_nm` field audit payload,
- grown and shrunk edge deltas,
- dependency-free field snapshot statistics,
- ASCII heatmap rows for report/UI previews,
- strongest field-edge records.

Run:

```bash
PYTHONPATH=src python domainpacks/swarm_robotics/morphogenetic_field_demo.py
```
