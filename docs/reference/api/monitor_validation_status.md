<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 | Contact: www.anulum.li | protoscience@anulum.li -->

# Monitor — Validation status

The machine-readable record of how much external evidence each monitor family
carries. SPO ships dozens of monitors, but only one detection niche — grid modal
damping — has been checked against an independent ground truth. This registry
keeps a broad monitor gallery from reading as a broad set of field-ready
detectors.

## Why it exists

The three tiers restate the repository's own `README` §*Evidence status* in
structured form, so downstream code and the Studio can filter on validation
posture without parsing prose:

- **`EXTERNALLY_VALIDATED`** (`"external"`) — clears a matched-false-alarm
  operating point *and* a permutation significance test on independent real
  data. Today: the grid modal envelope-growth detector and its streaming form.
- **`SYNTHETIC_ONLY`** (`"synthetic-only"`) — recovers an analytic or planted
  ground truth on synthetic data, yet is at chance on real data under the same
  honest test. The generic early-warning suite and the matrix-pencil estimator.
- **`RESEARCH`** (`"research"`) — an exploratory diagnostic with no external- or
  synthetic-reference validation record. The conservative default: a monitor is
  never promoted above this tier without a citable study section.

## Usage

```python
from scpn_phase_orchestrator.monitor import (
    MonitorValidationStatus,
    monitors_by_status,
    validation_record,
    validation_summary,
)

validation_record("grid_modal_growth").status
# <MonitorValidationStatus.EXTERNALLY_VALIDATED: 'external'>

[r.monitor for r in monitors_by_status(MonitorValidationStatus.EXTERNALLY_VALIDATED)]
# ['grid_modal_growth', 'grid_modal_stream']

{s.value: n for s, n in validation_summary().items()}
# {'external': 2, 'synthetic-only': 6, 'research': 28}
```

A test drift-guard fails closed if a newly added public monitor module is left
neither classified nor explicitly excluded, so the honest posture cannot silently
rot as the monitor suite grows.

::: scpn_phase_orchestrator.monitor.validation_status
