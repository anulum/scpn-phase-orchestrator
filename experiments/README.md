<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# `experiments/` — exploratory research scripts (experimental tier)

This directory holds one-off research scripts: model replications, parameter
studies, dataset explorations, and method comparisons (for example
`deco_replication.py`, `autotune_on_neurolib.py`, the anaesthesia and
seizure-detection studies) together with the result artefacts they produce.

## Status: experimental — not part of the supported product

These scripts are **not** part of the supported `scpn_phase_orchestrator` API and
carry **no stability, compatibility, or correctness guarantee**:

- **Not in the test suite.** `pytest` collects only `tests/` (`testpaths =
  ["tests"]`), so nothing here is exercised by CI's test lanes.
- **Not in the coverage scope.** Coverage measures only the
  `scpn_phase_orchestrator` package (`source = ["scpn_phase_orchestrator"]`), so
  these files never count toward the coverage gate.
- **Relaxed linting.** Docstring rules are disabled for `experiments/**` in
  `pyproject.toml`; the code is held to a lower bar than the package.
- **Opt-in.** Run a script directly and at your own discretion. Inputs, outputs,
  and APIs may change or break without notice, and result JSONs are
  reproducibility snapshots, not maintained data.

For supported, tested, documented functionality use the
`scpn_phase_orchestrator` package and its CLI (`spo …`).
