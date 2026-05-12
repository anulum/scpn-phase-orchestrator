<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Musical Acoustics Domainpack -->

# Musical Acoustics Domainpack

Maps harmonic modes, onset events, and tonal state to phase dynamics.
The domainpack is useful for demonstrating consonance, groove, and
dissonance boundaries with coupled oscillators.

## Layers

| Layer | Oscillators | Channel | Purpose |
|-------|-------------|---------|---------|
| bass | 3 | P | Fundamental and harmonic bass modes |
| harmony | 3 | P | Root, third, and fifth phase relation |
| melody | 3 | P | Lead, ornament, and vibrato phase |

## Boundaries

- `dissonance_ceiling`: soft roughness-index ceiling.
- `amplitude_ceiling`: hard sound-pressure ceiling.

## Actuation Guard

The binding spec includes a `value_alignment` template for review-time harmonic
coupling, tempo-drive, and tuning-offset actuation guards.

## Run

```bash
spo validate domainpacks/musical_acoustics/binding_spec.yaml
spo run domainpacks/musical_acoustics/binding_spec.yaml --steps 100
python domainpacks/musical_acoustics/run.py
```

## Read Next

- [Kuramoto Theory](../../docs/concepts/kuramoto_theory.md)
- [Analysis Toolkit](../../docs/guide/analysis_toolkit.md)
- [Monitor API](../../docs/reference/api/monitor.md)
