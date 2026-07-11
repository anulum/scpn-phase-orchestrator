<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->

# Monitor validation status

SCPN Phase Orchestrator ships dozens of dynamical monitors. Only **one** of them
has been checked against an independent ground truth. This page states, per
monitor family, how much external evidence it carries — so a broad gallery is
never mistaken for a broad set of field-ready detectors.

The classification is machine-readable: every monitor carries a
[`MonitorValidationStatus`][scpn_phase_orchestrator.monitor.validation_status.MonitorValidationStatus]
you can query with
[`validation_record()`][scpn_phase_orchestrator.monitor.validation_status.validation_record].
It restates the repository README's own *Evidence status* table verbatim in
structured form, so it cannot quietly overclaim.

## The three tiers

| Tier | Token | Meaning |
| --- | --- | --- |
| **Externally validated** | `external` | Clears a matched-false-alarm operating point *and* a permutation significance test on independent **real** data. |
| **Synthetic-only** | `synthetic-only` | Recovers an analytic or planted ground truth on **synthetic** data, yet is at chance on real data under the same honest test. |
| **Research** | `research` | An exploratory diagnostic with **no** external- or synthetic-reference validation record. The conservative default. |

Today the registry holds **2** externally-validated, **6** synthetic-only, and
**28** research-tier monitors.

## Externally validated (real-data evidence)

These are the only monitors with field evidence. Both are the grid modal-growth
detector — the one place a detector clears the honest bar, because the signature
is a physically deterministic growing mode.

| Monitor | Basis |
| --- | --- |
| **Grid modal envelope-growth detector** (`grid_modal_growth`) | Leads 36/90 real PSML growing-instability transitions at permutation `p = 0.0001` (held-out 24/45, `p = 0.0002`); the eigenvalue growth rate is a checkable physical quantity. |
| **Grid modal-growth streaming monitor** (`grid_modal_stream`) | The certified detector re-certified for causal streaming on the real PSML corpus with a hash-sealed operating point. |

## Synthetic-only (analytic ground truth; at chance on real data)

These recover the analytic eigenvalue on the synthetic transition suite but are
demonstrated **at chance** on real data — SPO reports that rather than
overclaiming.

| Monitor | Basis |
| --- | --- |
| **Critical slowing down** (`critical_slowing_down`) | Generic early-warning member (rising variance and lag-one autocorrelation); at chance on real data in all four domains. |
| **Rising synchronisation** (`synchronisation`) | Generic early-warning member (robust z-score of the Kuramoto order parameter). |
| **Ordinal-pattern transition entropy** (`opt_entropy`) | Generic early-warning member (ordinal-pattern transition entropy of the phase field). |
| **Ensemble early warning** (`ensemble_warning`) | Weighted fusion of the generic suite; inherits its synthetic-only posture. |
| **Domain-adaptable early-warning suite** (`early_warning_suite`) | The neutral-observable harness hosting the generic members; no member reaches significance on real data. |
| **Inter-area oscillation modes (matrix pencil)** (`oscillation_modes`) | Matrix-pencil modal damping recovers a planted growth rate exactly on synthetic data, yet on short real windows is at chance. |

## Research (no validation record)

The remaining **28** monitors are exploratory diagnostics. They may be useful,
but they carry **no** external- or synthetic-reference validation record and must
not be treated as field-ready detectors. Query their status directly:

```python
from scpn_phase_orchestrator.monitor import (
    MonitorValidationStatus,
    monitors_by_status,
)

for record in monitors_by_status(MonitorValidationStatus.RESEARCH):
    print(record.monitor, "—", record.basis)
```

## In one line

SPO is a synchronisation-dynamics and honest-evaluation toolkit whose one
externally-validated detection niche is grid modal damping against eigenvalues.
Everything else is synthetic-only or research. See the
[validation-status API](../reference/api/monitor_validation_status.md) for the
full machine-readable registry.
