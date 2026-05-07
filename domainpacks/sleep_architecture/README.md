<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Sleep Architecture Domainpack -->

# Sleep Architecture Domainpack

Maps sleep-band EEG dynamics to coupled oscillators. Delta, theta,
alpha, and gamma layers support sleep-stage phase analysis and boundary
checks for arousal and band-power conditions.

## Layers

| Layer | Oscillators | Channel | Purpose |
|-------|-------------|---------|---------|
| delta | 2 | P | Slow-wave sleep band |
| theta | 2 | P | NREM/REM transition band |
| alpha | 2 | P | Wake and arousal band |
| gamma | 2 | P | REM and binding-band activity |

## Boundaries

- `delta_power_ceiling`: soft delta-power ratio ceiling.
- `arousal_index`: hard arousals-per-hour ceiling.

## Value-Alignment Guard

The binding spec includes a `value_alignment` template for review-time
sleep-architecture control checks. It bounds global coupling, circadian-drive,
and phase-advance proposals, then falls back to a zero-drive safe hold when a
candidate action exceeds those priors.

This template is for simulation, replay, and policy review. It is not a live
sleep intervention protocol or medical-device control policy.

## Run

```bash
spo validate domainpacks/sleep_architecture/binding_spec.yaml
spo run domainpacks/sleep_architecture/binding_spec.yaml --steps 100
python domainpacks/sleep_architecture/run.py
```

## Read Next

- [Sleep Staging API](../../docs/reference/api/monitor_sleep_staging.md)
- [Analysis Toolkit](../../docs/guide/analysis_toolkit.md)
- [Notebook 16](../../notebooks/16_sleep_staging.ipynb)
