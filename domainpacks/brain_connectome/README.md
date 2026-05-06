<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Brain Connectome Domainpack -->

# Brain Connectome Domainpack

Maps a compact HCP-inspired cortical parcellation to phase oscillators.
Visual, motor, default-mode, and frontoparietal layers expose structural
and functional synchronisation surfaces for research simulations.

## Layers

| Layer | Oscillators | Channel | Purpose |
|-------|-------------|---------|---------|
| visual | 3 | P | Alpha-band visual regions |
| motor | 3 | P | Sensorimotor oscillators |
| default_mode | 3 | P | Default-mode network |
| frontoparietal | 3 | P | Task-control network |

## Boundaries

- `global_sync_ceiling`: seizure-like global synchrony ceiling.
- `dmn_suppression_floor`: low default-mode activation guard.
- `functional_connectivity`: minimum functional-connectivity magnitude.

## Run

```bash
spo validate domainpacks/brain_connectome/binding_spec.yaml
spo run domainpacks/brain_connectome/binding_spec.yaml --steps 100
python domainpacks/brain_connectome/run.py
```

## Read Next

- [Monitor API](../../docs/reference/api/monitor.md)
- [BOLD and neural-network reference](../../docs/reference/api/nn.md)
- [Domain Utilisation Schemas](../../docs/concepts/domain_utilisation_schemas.md)
