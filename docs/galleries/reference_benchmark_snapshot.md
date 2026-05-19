<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Reference Benchmark Snapshot -->

# Reference Benchmark Snapshot

This page publishes a dated benchmark snapshot from the reference suite. Treat
these numbers as historical measurements for the listed environment, not as
fresh validation unless the command is rerun and the JSON artefact is updated.

## Reproduction Metadata

| Field | Value |
|-------|-------|
| Snapshot date | `2026-05-20` |
| Suite version | `reference_suite_v1` |
| Command | `PYTHONPATH=src python benchmarks/reference_suite.py` |
| Backend | `python_numpy` |
| Python | `CPython 3.12.3` |
| NumPy | `2.4.4` |
| Platform | `Linux-6.17.0-23-generic-x86_64-with-glibc2.39` |
| Executable | `/media/anulum/724AA8E84AA8AA75/aaa_God_of_the_Math_Collection/03_CODE/SCPN-PHASE-ORCHESTRATOR/.venv/bin/python` |
| JSON artefact | `benchmarks/results/reference_suite.json` |

## Historical Results

| Suite ID | Reference surface | Size | Steps | Wall time (s) | Steps/s | Summary value |
|----------|-------------------|------|-------|---------------|---------|---------------|
| `auto_binding_synthetic_quality` | Synthetic auto-binding extractor/K proposal quality | 4 fixtures | 4 domain gates | 0.06027109699789435 | 66.3668026506925 | validation errors = 0; extractor coverage = 1.0; expected edge recall = 1.0; proposed edges = 33; accepted domains = 4/4 |
| `kuramoto_reference_strogatz_2000` | Strogatz-style all-to-all Kuramoto reference | 64 oscillators | 1000 | 0.16471287398599088 | 6071.17085507871 | final `R` = 1.0 |
| `stuart_landau_reference_pikovsky_2001` | Pikovsky-style coupled amplitude/phase reference | 64 oscillators | 1000 | 0.31584654201287776 | 3166.094501548248 | final mean amplitude = 3.6193922141707704 |
| `petri_net_reachability` | Supervisor reachability traversal | 4 places | 5000 | 0.02358600898878649 | 211990.0828655307 | reachable markings = 4 |

## Auto-Binding Acceptance Gates

The auto-binding benchmark now evaluates larger deterministic domain-like
datasets instead of only toy smoke fixtures. Each fixture has explicit
domain-specific thresholds for minimum extractor coverage, expected-edge recall,
maximum validation errors, minimum sample count, and maximum proposed-edge
multiplier. The 2026-05-20 snapshot passed all four gates:

| Domain fixture | Samples | Expected-edge recall | Extractor coverage | Proposed-edge multiplier | Accepted |
|----------------|--------:|---------------------:|-------------------:|-------------------------:|----------|
| `phase_chain` | 128 | 1.0 | 1.0 | 6.0 | yes |
| `industrial_sensor_chain` | 128 | 1.0 | 1.0 | 6.0 | yes |
| `cardiac_rhythm_surrogate` | 160 | 1.0 | 1.0 | 4.5 | yes |
| `power_grid_surrogate` | 192 | 1.0 | 1.0 | 6.0 | yes |

## Use Policy

- Re-run `PYTHONPATH=src python benchmarks/reference_suite.py` before using the
  values for a release note, paper table, or performance claim.
- Keep the command, backend, versions, platform, and snapshot date next to any
  copied result.
- Do not compare this page against a different host or backend without adding a
  separate dated artefact.
