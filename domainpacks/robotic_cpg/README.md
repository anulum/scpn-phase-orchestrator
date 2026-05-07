<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Robotic CPG Domainpack -->

# Robotic CPG Domainpack

Maps central-pattern-generator locomotion to phase oscillators. Leg and
arm joint oscillators can be coupled, driven, and phase-biased to study
gait coordination under supervisor constraints.

## Layers

| Layer | Oscillators | Channel | Purpose |
|-------|-------------|---------|---------|
| left_leg | 2 | P | Hip and knee CPG phase |
| right_leg | 2 | P | Contralateral leg CPG phase |
| left_arm | 2 | P | Shoulder and elbow swing phase |
| right_arm | 2 | P | Contralateral arm swing phase |

## Boundaries

- `joint_limit`: hard joint-angle range.
- `torque_ceiling`: hard torque ceiling.

## Value-Alignment Guard

The binding spec includes a `value_alignment` template for review-time
locomotion-control checks. It bounds gait coupling, stride-frequency drive,
and phase-bias steps, then falls back to a zero-stride safe hold when a
candidate action exceeds those priors.

This template is for simulation, replay, and policy review. It is not a live
robot safety controller.

## Run

```bash
spo validate domainpacks/robotic_cpg/binding_spec.yaml
spo run domainpacks/robotic_cpg/binding_spec.yaml --steps 100
python domainpacks/robotic_cpg/run.py
```

## Read Next

- [Advanced Dynamics](../../docs/guide/advanced_dynamics.md)
- [Swarmalator API](../../docs/reference/api/upde_swarmalator.md)
- [Control Systems](../../docs/guide/control_systems.md)
