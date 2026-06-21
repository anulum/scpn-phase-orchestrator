# Assurance Case

The assurance subsystem composes existing SPO runtime evidence — audit-chain
integrity, replay determinism, formal verification, twin-confidence scoring, and
the conformal admission gate — into a single hash-sealed **assurance-case
bundle**, and maps that evidence to the published clauses of three standards a
regulated deployment is commonly measured against:

- Regulation (EU) 2024/1689 (the EU AI Act) — high-risk requirements;
- ISO/IEC 42001:2023 — AI management system clauses and Annex A controls;
- ANSI/UL 4600 — claim-based safety case for autonomous products.

The bundle is **review-only**: `actuation_permitted` is always `False`, the
bundle hash seals the evidence and conformance records deterministically, and a
disclaimer states that the bundle is a technical evidence-mapping aid, not a
legal conformity assessment. Clauses with no contributing technical evidence are
recorded as `not_addressed` so coverage gaps are explicit rather than implied.

```bash
spo assurance-case --system my-deployment \
  --audit-log run.jsonl \
  --evidence-file twin_confidence.json \
  --output assurance_bundle.json
```

## Regulatory clause catalogue

`scpn_phase_orchestrator.assurance.standards` records each referenceable clause
with its standard, identifier, official title, and a provenance note. Clause
identifiers and titles are taken from the public structure of each standard; the
clause text must be confirmed against the official standard before any external
submission.

::: scpn_phase_orchestrator.assurance.standards

## Evidence items

`scpn_phase_orchestrator.assurance.evidence` wraps the JSON-safe audit record of
an originating surface in a content-addressed `EvidenceItem`, so the bundle can
reference evidence by a stable identifier and detect later mutation.

::: scpn_phase_orchestrator.assurance.evidence

## Bundle assembly

`scpn_phase_orchestrator.assurance.case` maps each catalogued clause to the
evidence that addresses it, records the conformance status and rationale, and
seals the result into a deterministic, fail-closed bundle.

::: scpn_phase_orchestrator.assurance.case

## Oscillation-monitoring evidence (NERC PRC-028 / PRC-030)

`scpn_phase_orchestrator.assurance.prc_oscillation` is the audit-package end of
the dVOC grid pack. `screen_oscillation_modes` takes the modes recovered by the
[matrix-pencil estimator](monitor_oscillation_modes.md), screens each damping
ratio against the oscillation-monitoring practice underlying NERC PRC-028 and the
proposed PRC-030 — undamped (non-positive damping) and poorly-damped (below a few
percent) modes are flagged — and seals the screening into a content-addressed,
review-only `PRCOscillationEvidence` record. The capture timestamp is supplied by
the caller (the measurement time of the event), so the record is deterministic and
reproducible. Like the assurance-case bundle, it is a technical evidence-mapping
aid, not a legal conformity assessment, and it never actuates.

::: scpn_phase_orchestrator.assurance.prc_oscillation
