<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# `fuzzers/` — fuzzing harnesses (experimental tier)

This directory holds fuzzing harnesses that feed randomised and malformed inputs
at SPO's parsing and validation surfaces to hunt for crashes and unhandled
errors — for example `policy_yaml_fuzzer.py`, which fuzzes the policy-YAML loader.

## Status: experimental — opt-in robustness exploration

The harnesses are a **robustness-exploration aid**, not part of the supported
product or the regular test run:

- **Not in the test suite.** `pytest` collects only `tests/` (`testpaths =
  ["tests"]`); the fuzzers are run by hand, on demand, typically for a long
  campaign rather than a single CI pass.
- **Not in the coverage scope.** Coverage measures only the
  `scpn_phase_orchestrator` package, so the harnesses never count toward the
  coverage gate.
- **Relaxed linting.** Docstring rules are disabled for `fuzzers/**` in
  `pyproject.toml`.
- **Opt-in.** Run a harness directly. A finding is a lead to file as a regular
  test in `tests/`, where the supported guarantees live; the harness itself
  carries no stability guarantee.
