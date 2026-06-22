<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# `bench/` — performance benchmark harnesses (experimental tier)

This directory holds performance benchmark harnesses and their baseline records:
`run_benchmarks.py` (the runner CI invokes), the per-feature `bench_*.py`
harnesses, the ablation harness, and `baseline.json` / `current.json` snapshots.

## Status: experimental — measurement, not a guarantee

CI runs `python bench/run_benchmarks.py` to track performance trends, but the
benchmarks are **measurement instruments, not a stability contract**:

- **Numbers are environment-dependent.** Timings vary with CPU, load, and which
  of the polyglot backends (Rust / Mojo / Julia / Go / Python) is active, so a
  given figure is not a guaranteed or portable value.
- **Not in the test suite or coverage scope.** `pytest` collects only `tests/`
  and coverage measures only the `scpn_phase_orchestrator` package, so the
  harnesses never gate the test or coverage lanes; the dedicated benchmark job is
  a trend signal, and wall-clock thresholds there are advisory.
- **Relaxed linting.** Docstring rules are disabled for `benchmarks/**`; the
  harness code is held to a lower bar than the package.
- **Opt-in.** Run a harness directly to profile a path. The supported, tested
  behaviour lives in the `scpn_phase_orchestrator` package and `tests/`.
