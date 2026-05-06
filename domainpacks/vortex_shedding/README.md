<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Vortex Shedding Domainpack -->

# Vortex Shedding Domainpack

Maps a cylinder-wake style vortex street to Stuart-Landau phase-amplitude
dynamics. Upstream, midstream, and downstream wake layers expose lock-in
and wake-stabilisation objectives.

## Layers

| Layer | Oscillators | Channel | Purpose |
|-------|-------------|---------|---------|
| upstream | 3 | P | Wake onset modes |
| midstream | 3 | P | Convective wake modes |
| downstream | 3 | P | Wake recovery modes |

## Boundaries

- `strouhal_band`: expected shedding-frequency band.
- `amplitude_ceiling`: hard lift-coefficient ceiling.

## Run

```bash
spo validate domainpacks/vortex_shedding/binding_spec.yaml
spo run domainpacks/vortex_shedding/binding_spec.yaml --steps 100
python domainpacks/vortex_shedding/run.py
```

## Read Next

- [Stuart-Landau Guide](../../docs/guide/stuart_landau.md)
- [UPDE Stuart-Landau API](../../docs/reference/api/upde_stuart_landau.md)
- [Envelope API](../../docs/reference/api/upde_envelope.md)
