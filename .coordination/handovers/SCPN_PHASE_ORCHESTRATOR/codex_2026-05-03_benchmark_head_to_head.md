# Handover — Benchmark Head-to-Head Pages

## Completed
- Added `docs/galleries/cardiac_rhythm_benchmark.md` with results and reproducibility commands.
- Added SPDX header block to `docs/galleries/power_grid_benchmark.md`.
- Wired both benchmark pages into MkDocs Gallery nav.
- Marked the roadmap checkpoint for deferred benchmark pages as completed.

## Validation
- `PYTHONPATH=src python examples/cardiac_rhythm.py` succeeded and produced the reference script metrics.
- `/home/anulum/.local/bin/mkdocs build --clean` and strict build both fail due existing `mkdocs.yml` config issue:
  - `Config value 'nav': Expected nav to be a list, got None` (`{'Tutorials': None}` entry).

## Handoff Notes
- Full policy-enabled domainpack runs for `power_grid` and `cardiac_rhythm` exceed short shell timeouts under the current working-tree engine performance; use `timeout` if re-running locally.
- Remaining action: if desired, profile `power_grid` and `cardiac_rhythm` runtime with this branch before publishing benchmark deltas as fresh numbers.
