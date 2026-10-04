<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Phase Orchestrator — Contributing Guide
-->

# Contributing

## Dev Setup

```bash
git clone https://github.com/anulum/scpn-phase-orchestrator.git
cd scpn-phase-orchestrator
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Required branch coverage profiles

The branch ratchet consumes measurements from the same revision, complete
installed Python package, locked dependencies and coverage configuration. CI
produces a genuine `spo-kernel` profile, a profile with the kernel genuinely
absent, and a defective native profile with two compiled variants. Raw branch
databases, package and input hashes, extension provenance and public contract
results accompany each artifact. A missing, stale, statement-only or mismatched
artifact fails the required coverage category.

`python -m tools.branch_coverage_profiles record` observes each installed runtime.
`python -m tools.branch_coverage_profiles combine` admits the four artifacts,
uses `coverage combine` without deleting originals, and verifies that the
combined data holds exactly the raw source membership and arc union. `--root`
names the checkout; the command can be started from any directory, and
relative paths are taken from that directory. A checkout
source that changed after recording, or that is a symbolic link, is refused.
`tools/coverage_guard.py` then applies the existing file, domain and
global thresholds to the aggregate XML. The guard qualification for E/I, sheaf
and sparse engines is a separate check; it does not establish global coverage.

The Rust extension in `tests/native_output_fixture/` deliberately emits invalid
native outputs or omits stepper classes. It runs only in isolated environments
after `prove-absent` records actual kernel absence. It implements no successful
numerical solver. Its coverage credit is restricted to the reviewed defensive
statements and branch outcomes in `tools/branch_profile_residuals.json`; genuine
profiles must cover every other success path. Both variants are compiled with
the checked-in lock, checked by clippy and covered by the blocking Cargo audit.
The fixture is excluded from release wheels and source distributions. Its wheels
are not uploaded as CI artifacts or installed as a production backend.

Run the explicit fixture contracts in their corresponding prepared environment:
`output_contracts.py` for invalid outputs and `missing_classes_contracts.py` for
both the missing-classes variant and genuine absence. After all actual profile
artifacts are available under `native/`, `absent/`, `defective-output/` and
`defective-missing/`, set `SPO_BRANCH_PROFILE_INPUTS` to their parent directory
and run `pytest tests/native_output_fixture/admission_contracts.py`. These tests
invoke the public admission CLI and verify refusal using copied metadata faults
or actual incomplete and statement-only measurements.

## Adding Domainpacks

Create a directory under `domainpacks/<name>/` with a `binding_spec.yaml`:

```yaml
name: my_domain
version: "0.1.0"
safety_tier: research
sample_period_s: 0.01
control_period_s: 0.1

layers:
  - name: lower
    index: 0
    oscillator_ids: [osc_0, osc_1]

oscillator_families:
  base:
    channel: P
    extractor_type: physical

coupling:
  base_strength: 0.45
  decay_alpha: 0.3

drivers:
  physical: {}
  informational: {}
  symbolic: {}

objectives:
  good_layers: [0]
  bad_layers: []

boundaries:
  - name: r_floor
    variable: R
    lower: 0.2
    severity: hard
actuators:
  - name: coupling_knob
    knob: K
    scope: global
```

Add a `policy.yaml` alongside the binding spec for domain-specific supervisor rules:

```yaml
rules:
  - name: boost_coupling
    condition: {metric: R, operator: "<", threshold: 0.5}
    regime: degraded
    actions:
      - {knob: K, scope: global, value: 0.05, ttl_s: 5.0}
```

See `domainpacks/minimal_domain/` for a complete example.

## Adding Oscillators

Subclass `PhaseExtractor` in `src/scpn_phase_orchestrator/oscillators/`:

```python
from scpn_phase_orchestrator.oscillators.base import PhaseExtractor

class MyExtractor(PhaseExtractor):
    def extract(self, raw_signal: np.ndarray) -> np.ndarray:
        # Return instantaneous phase array
        ...
```

## Pre-push Preflight (mandatory)

Every push is gated by a local CI mirror. Set it up once:

```bash
git config core.hooksPath .githooks
git config commit.template .gitmessage
```

This installs a `commit-msg` hook that requires the project authorship trailer
and a `pre-push` hook that runs `tools/preflight.py` — the same 10 gates CI
enforces (ruff, format, version-sync, mypy, module-linkage, pytest, bandit,
cargo fmt, cargo clippy, cargo test). Push is blocked if any gate fails.

Every new commit message must include exactly one project authorship trailer:

```text
Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
```

Historical `Co-Authored-By: Arcane Sapience <protoscience@anulum.li>` trailers
are left unchanged, but the old trailer is no longer accepted for new commits.

To run manually at any time:

```bash
python tools/preflight.py                   # full (~3 min)
python tools/preflight.py --no-tests        # lint-only (~5 sec)
python tools/preflight.py --coverage        # full + line-coverage guard
python tools/preflight.py --branch-coverage # full + branch-coverage guard
                                            # (performance tests deselected)
```

## Running Tests

```bash
pytest                                         # core test suite
ruff check src/ tests/
ruff format --check src/ tests/
```

### Coverage across Python processes

The development profile requires coverage 7.16.1 or newer and pytest-cov 7.1
or newer. `pyproject.toml` enables the coverage subprocess patch and separate
process data files. Python children inherit measurement, including fresh
interpreters started with `-I`, when they preserve the coverage environment and
exit normally. Native Julia/Go/Mojo execution is outside Python coverage; their
existing parity lanes remain responsible for those implementations.

For a focused pytest run, pytest-cov combines the parent and child files before
reporting. CI uploads that combined `.coverage` under its lane-specific name;
the coverage assurance job merges the lane artefacts and applies the existing
module thresholds. Avoid concurrent runs sharing a `COVERAGE_FILE` location.

```bash
pytest tests/test_network_security.py --cov=scpn_phase_orchestrator.runtime.network_security
```

For direct coverage runs, combine process files explicitly before reporting:

```bash
coverage run --source=scpn_phase_orchestrator.runtime.network_security -m pytest tests/test_network_security.py
coverage combine
coverage report --include='*/runtime/network_security.py'
```

This configuration does not recover data from `SIGKILL`, `os._exit`, or children
that discard the measurement environment. Wait for measured children to finish
before combining. Subprocess transport and line/branch collection are exercised
by `tests/test_subprocess_coverage.py` through a real rate limiter in an isolated child
and the same data-file rename/combine sequence used by CI.

### nn/ Module Physics Validation (requires JAX)

The nn/ module has a dedicated 194-test physics validation suite that
verifies the JAX backend against known analytical results. Requires
`pip install -e ".[nn]"` (installs JAX + equinox + optax).

```bash
# All phases except P7 (~13 min)
pytest tests/test_nn_physics_validation.py \
       tests/test_nn_physics_validation_p{2..6}.py \
       tests/test_nn_physics_validation_p{8..13}.py

# Phase 7 FIM validation (~32 min, Python loops)
pytest tests/test_nn_physics_validation_p7.py

# Single phase (fast)
pytest tests/test_nn_physics_validation_p4.py -v
```

See `docs/reference/nn_physics_validation_plan.md` for the full test
matrix, results, and 14 documented findings.

## Commit Style

Imperative mood, under 72 characters. Examples:

- `Add fusion domainpack with MHD extractor`
- `Fix coupling decay exponent off-by-one`
- `Remove unused spectral helper`

## Module Size

Line count is a proxy, never the gate: single-responsibility code is fine at any
length, and multi-responsibility code must be split at any length. A module over
~900 lines is a review trigger and over ~1200 a strong split signal — but the
decisive test is the AST call-graph, not the number. See
[`docs/module_size_policy.md`](docs/module_size_policy.md).

Survey the source tree with:

```bash
python tools/check_module_size.py          # warning report (exit 0)
python tools/check_module_size.py --check   # ratchet: fail on un-allowlisted >1200
```
