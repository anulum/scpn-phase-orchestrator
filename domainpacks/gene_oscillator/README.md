<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Gene Oscillator Domainpack -->

# Gene Oscillator Domainpack

Models synthetic gene-circuit phase dynamics with repressilator and
quorum-sensing layers. It is a research domainpack for mapping gene
expression cadence, promoter events, and cell-cycle state into P/I/S
channels.

## Layers

| Layer | Oscillators | Channel | Purpose |
|-------|-------------|---------|---------|
| repressilator_a | 2 | P | First negative-feedback loop |
| repressilator_b | 2 | P | Second negative-feedback loop |
| quorum | 2 | P | Population-level quorum oscillator |

## Boundaries

- `expression_ceiling`: soft protein-concentration ceiling.
- `quorum_threshold`: soft AHL concentration floor.

## Run

```bash
spo validate domainpacks/gene_oscillator/binding_spec.yaml
spo run domainpacks/gene_oscillator/binding_spec.yaml --steps 100
python domainpacks/gene_oscillator/run.py
```

## Read Next

- [Phase Contract](../../docs/specs/phase_contract.md)
- [Coupling API](../../docs/reference/api/coupling.md)
- [Stuart-Landau Guide](../../docs/guide/stuart_landau.md)
