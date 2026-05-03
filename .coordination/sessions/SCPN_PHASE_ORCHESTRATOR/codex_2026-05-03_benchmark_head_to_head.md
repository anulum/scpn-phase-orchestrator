# Codex session: benchmark head-to-head pages

Date: 2026-05-03
Project: SCPN-PHASE-ORCHESTRATOR

## Scope

- Continue the roadmap work on deferred benchmark publication for domainpacks.
- Finish and publish the cardiac benchmark page and wire benchmark pages into docs nav.
- Mark the roadmap item complete in `ROADMAP.md`.

## Changes

- Added `docs/galleries/cardiac_rhythm_benchmark.md` with:
  - domain mapping for cardiac_rhythm
  - head-to-head results table (policy-enabled vs policy-disabled)
  - baseline reference outputs from `examples/cardiac_rhythm.py`
  - reproducibility commands
- Added SPDX header block to `docs/galleries/power_grid_benchmark.md`.
- Added both benchmark pages to MkDocs Gallery nav in `mkdocs.yml`:
  - `galleries/cardiac_rhythm_benchmark.md`
  - `galleries/power_grid_benchmark.md`
- Updated roadmap item to completed:
  - `Publish head-to-head benchmark pages for domainpacks ...` now struck through with done note.

## Validation

- `PYTHONPATH=src python examples/cardiac_rhythm.py` completed with output lines for normal rhythm, AV block, and pacemaker scenarios.
- `mkdocs build` via `/home/anulum/.local/bin/mkdocs build --clean` currently fails with pre-existing config issue:
  - `Config value 'nav': Expected nav to be a list, got None` caused by existing `{'Tutorials': None}` entry.

## Note

- Domainpack policy-enabled runs for `power_grid` and `cardiac_rhythm` were observed to exceed short command timeouts in this environment, so values were retained from the previously captured local benchmark outputs.
