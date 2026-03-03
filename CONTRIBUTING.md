<!--
SCPN Phase Orchestrator
Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
ORCID: https://orcid.org/0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
License: GNU AGPL v3 | Commercial licensing available
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

boundaries: []
actuators: []
```

See `domainpacks/minimal_domain/binding_spec.yaml` for a complete example.

## Adding Oscillators

Subclass `PhaseExtractor` in `src/scpn_phase_orchestrator/oscillators/`:

```python
from scpn_phase_orchestrator.oscillators.base import PhaseExtractor

class MyExtractor(PhaseExtractor):
    def extract(self, raw_signal: np.ndarray) -> np.ndarray:
        # Return instantaneous phase array
        ...
```

## Running Tests

```bash
pytest
ruff check src/ tests/
ruff format --check src/ tests/
```

## Commit Style

Imperative mood, under 72 characters. Examples:

- `Add fusion domainpack with MHD extractor`
- `Fix coupling decay exponent off-by-one`
- `Remove unused spectral helper`
