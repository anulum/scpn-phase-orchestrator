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
```

This installs a `pre-push` hook that runs `tools/preflight.py` — the same
10 gates CI enforces (ruff, format, version-sync, mypy, module-linkage,
pytest, bandit, cargo fmt, cargo clippy, cargo test). Push is blocked if
any gate fails.

To run manually at any time:

```bash
python tools/preflight.py            # full (~3 min)
python tools/preflight.py --no-tests # lint-only (~5 sec)
python tools/preflight.py --coverage # full + coverage guard
```

## Running Tests

```bash
pytest                                         # core test suite
ruff check src/ tests/
ruff format --check src/ tests/
```

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
