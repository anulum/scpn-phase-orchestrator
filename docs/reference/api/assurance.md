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
  --output assurance_bundle.json \
  --report-out conformity_report.md \
  --report-pdf-out conformity_report.pdf
```

`--report-out` additionally renders a human-readable Markdown conformity report
from the same sealed bundle — a per-standard, clause-by-clause table of
conformance status, contributing evidence, and rationale, anchored to the bundle
hash for traceability. `--report-pdf-out` renders the same report as a
deterministic, dependency-free text PDF — the distributable artefact an assessor
files.

For operator review packages, `spo certification-evidence` wraps the same
assurance bundle with deterministic test vectors and a manifest:

```bash
spo certification-evidence --system my-deployment \
  --run-result run_summary.json \
  --output-dir review_package
```

`--run-result` takes a serialised `SimulationResult` summary and auto-derives the
run's audit-stream integrity and conformal admission-gate evidence, so a package
can be assembled from a run summary without hand-authoring evidence JSON
(`--audit-log` and `--evidence-file` remain available and compose with it). With
`--audit-log`, adding `--verify-determinism` re-executes the logged run and
records a `replay_determinism` evidence item for the reproducibility clauses.
`--formal-package` takes a serialised `FormalVerificationPackage` manifest (from
the supervisor formal exporters) and adds a `formal_verification` evidence item for
the formal-argument clauses, recording which model-checking properties were posed
against which exported artefacts.

With `--audit-log`, `--sign-envelope` additionally writes `signed_envelope.json` —
a deterministic binding of the package hash to the run's audit-chain tip, so the
package is anchored to a specific, tamper-evident, replayable execution.
`--signing-seed-file` supplies an ML-DSA seed (FIPS 204) and adds a post-quantum
seal over that tip to the envelope, making the binding publicly verifiable
(it implies `--sign-envelope`). The ML-DSA seal needs the `pqc` extra and an
OpenSSL 3.5+ backend.

The package directory contains:

- `manifest.json` — file digests, the assurance bundle hash, standards covered,
  coverage summary, package hash, and review-only disclaimers;
- `assurance_bundle.json` — the existing `scpn_assurance_case_bundle_v1`
  payload;
- `conformity_report.md` — a human-readable, per-standard clause-by-clause
  conformity report rendered from the bundle and sealed into the manifest digest;
- `conformity_report.pdf` — the same conformity report as a deterministic text
  PDF (the filable artefact), also sealed into the manifest digest;
- `test_vectors.json` — recomputable evidence content-hash vectors and
  clause-rationale hash vectors;
- `signed_envelope.json` — *(only with `--sign-envelope`)* the package hash bound
  to the run's audit-chain tip, optionally carrying a post-quantum ML-DSA seal.

The package is standards-shaped evidence for reviewer triage. It does not claim
legal compliance, certification, or runtime actuation permission.

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
The shared canonical hashing path accepts only strict JSON records: `NaN`,
`Infinity`, and `-Infinity` are rejected before any digest is emitted, so hashes
remain portable across JSON implementations and non-Python verifiers.

::: scpn_phase_orchestrator.assurance.evidence

## Run-derived evidence

`scpn_phase_orchestrator.assurance.run_evidence` maps the trust-relevant fields
of a serialised `SimulationResult` record — the close-time audit-stream integrity
result and the conformal admission-gate decisions — into evidence items. It
consumes the JSON-safe record (not the runtime object), so the assurance package
stays free of the numeric runtime import chain, and it emits nothing for a
surface that did not run.

::: scpn_phase_orchestrator.assurance.run_evidence

## Formal-verification evidence

`scpn_phase_orchestrator.assurance.formal_evidence` maps a serialised
`FormalVerificationPackage.to_audit_record()` manifest — the supervisor formal
exporters' artefact hashes, model-checking property library, and non-executing
checker commands — into a single `formal_verification` evidence item. Like the
run-derived evidence, it consumes the JSON manifest (not the package object), so
the assurance package stays free of the supervisor import chain, and it restates
the manifest verbatim: it records which properties were posed against which
artefacts, never that any external checker accepted them.

::: scpn_phase_orchestrator.assurance.formal_evidence

## Bundle assembly

`scpn_phase_orchestrator.assurance.case` maps each catalogued clause to the
evidence that addresses it, records the conformance status and rationale, and
seals the result into a deterministic, fail-closed bundle.

::: scpn_phase_orchestrator.assurance.case

## Certification evidence package

`scpn_phase_orchestrator.assurance.certification` assembles the review package
around the assurance-case bundle. It keeps package assembly deterministic and
hash-sealed while preserving the same review-only boundary as the underlying
assurance case.

::: scpn_phase_orchestrator.assurance.certification

## Signed certification envelope

`scpn_phase_orchestrator.assurance.envelope` binds a certification package to the
run that produced it. A `SignedCertificationEnvelope` commits, in one deterministic
record, to the package hash, the run's audit-chain tip (the SHA-256 commitment to
the whole audit log, so the package is anchored to a specific, replayable, tamper-
evident execution), and an optional post-quantum seal over that tip
(`scpn_phase_orchestrator.runtime.audit_pqc.AuditChainSeal`, ML-DSA / FIPS 204). It
reuses the audit seal verbatim and performs no signing or log reading itself — the
CLI layer reads the tip and produces the seal — so the assurance leaf only validates
and binds. `verify_signed_certification_envelope` re-derives the envelope hash,
checks the package binding, and verifies any attached seal against a trusted public
key.

::: scpn_phase_orchestrator.assurance.envelope

## Supply-chain provenance (SLSA / DSSE)

The certification envelope attests to a *run*; the provenance layer attests to a
*build* — which release artefacts were produced, from which resolved inputs, by
which builder. `scpn_phase_orchestrator.assurance.provenance` assembles a
deterministic [in-toto Statement v1](https://in-toto.io/Statement/v1) carrying a
[SLSA provenance v1](https://slsa.dev/provenance/v1) predicate: the produced
artefacts as digest-pinned subjects, the build definition (build type, external
parameters, digest-pinned resolved dependencies), and the run details (builder
identity and invocation). The run details also carry the optional `builder.version`
map, the builder's own digest-pinned `builderDependencies`, and the build
`byproducts` (for example a digest-pinned SBOM); each is omitted when empty, so a
minimal statement is byte-identical to one without them. `pypi_resolved_dependency`
turns a hash-pinned lock-file entry into a Package-URL-addressed resolved dependency,
so the `resolvedDependencies` block can carry the full dependency tree. It reads no
wall clock and makes no network call, so the same build inputs always serialise to
the same statement.

`scpn_phase_orchestrator.assurance.dsse` wraps that statement in a
[DSSE](https://github.com/secure-systems-lab/dsse) envelope — the wire format
`cosign attest` produces — and signs its pre-authentication encoding with ML-DSA
(FIPS 204), reusing the single post-quantum primitive in
`scpn_phase_orchestrator.runtime.audit_pqc`. Each signature records its algorithm so
a second scheme can be added without breaking existing envelopes; **SLH-DSA**
(FIPS 205 / SPHINCS+) is the reserved hash-based alternative and is added once the
`cryptography` backend ships it. Verification is offline and self-contained: the
verifier supplies the trusted public key, whose short id must match the signature.

```bash
spo provenance-attest build_provenance.json \
  --signing-seed-file signing.seed > attestation.json

spo provenance-verify attestation.json --public-key-file signer.pub
```

Signing needs the `pqc` extra and an OpenSSL 3.5+ backend. Publishing the envelope
to a Rekor transparency log or verifying it with `cosign` is an optional operator
step that needs network and OIDC, and is left to the operator; the envelope itself
is deterministic and verifiable without either.

The release workflow (`.github/workflows/release.yml`) wires this into the build:
after building the sdist and SBOM it runs `tools/build_release_provenance_spec.py`
to assemble the spec — the sdist as a subject, the SBOM as a byproduct, the
hash-pinned lock files as resolved dependencies, and the tag, commit, and runner
metadata as the build definition and run details — then signs it with `spo
provenance-attest` using the `SPO_PROVENANCE_SIGNING_SEED` repository secret, and
attaches `provenance_attestation.json` and `provenance_signing_key.pub` to the
GitHub Release. The seed is written to a private file, used, and deleted within the
step; it is never committed. When the secret is not configured the step is skipped
and the release still carries GitHub's own keyless build-provenance attestation.
Consumers should obtain the public key from a trusted channel before pinning it.

::: scpn_phase_orchestrator.assurance.provenance

::: scpn_phase_orchestrator.assurance.dsse

## Conformity report

`scpn_phase_orchestrator.assurance.report` renders an assurance-case bundle as a
deterministic Markdown conformity report — the document a regulatory assessor
reads. It restates the sealed bundle verbatim (coverage rollup, per-standard
clause conformance with status, evidence, and rationale, and the evidence
inventory) under the regulatory disclaimer and anchored to the bundle hash. It
adds no claim beyond the bundle and is review-only. The certification evidence
package seals the rendered report as `conformity_report.md`.
`render_conformity_report_pdf` renders the same content as a deterministic,
dependency-free text PDF — the distributable artefact an assessor files — built
on the reusable `scpn_phase_orchestrator.reporting.markdown_to_pdf_bytes` helper.

::: scpn_phase_orchestrator.assurance.report

## Oscillation-Monitoring Evidence (NERC PRC-028-1 / PRC-030-1)

`scpn_phase_orchestrator.assurance.prc_oscillation` is the audit-package end of
the dVOC grid pack. `screen_oscillation_modes` takes the modes recovered by the
[matrix-pencil estimator](monitor_oscillation_modes.md), screens each damping ratio
for PRC-028-1 disturbance-data analysis and PRC-030-1 unexpected IBR event
mitigation workflows, and seals the screening into a content-addressed,
review-only `PRCOscillationEvidence` record. Undamped modes and positive but
poorly damped modes are flagged for operator review. Each finding also carries
the engineering mode family from the matrix-pencil estimator, and the record
aggregates `mode_family_counts`, so inter-area and sub-synchronous oscillation
signals are visible in the same sealed package. The capture timestamp is supplied
by the caller, so the record is deterministic and reproducible. Like the
assurance-case bundle, it is a technical evidence-mapping aid, not a legal
conformity assessment, and it never actuates.

::: scpn_phase_orchestrator.assurance.prc_oscillation

## Ride-Through Evidence (NERC PRC-029-1)

`scpn_phase_orchestrator.assurance.prc_ride_through` screens operator-provided
high-side transformer voltage and frequency samples against the approved NERC
PRC-029-1 ride-through tables. It carries both voltage categories from
Attachment 1 — AC-connected wind IBRs and all other IBRs — plus the Attachment 2
frequency bands. The screener aggregates cumulative duration inside the
standard's voltage and frequency review windows, records the operation region,
minimum ride-through duration, observed value range, and review classification
for each non-nominal band, then seals the record as `PRCRideThroughEvidence`.

The record is review-only. It does not evaluate real/reactive-current
performance, phase-jump exceptions, hardware-limit exemptions, reporting duties,
or legal compliance. Observations outside the review envelope use
`assessor_review_required`, not pass/fail language.

::: scpn_phase_orchestrator.assurance.prc_ride_through

## Power-Grid PRC Assessor Bundle

`scpn_phase_orchestrator.assurance.power_grid_prc_bundle` binds the three
power-grid PRC review artefacts into one deterministic handoff package:

- `scpn_dvoc_oscillation_damping_audit_v1` from the offline dVOC/Koopman-MPC
  damping screen;
- `scpn_pmu_ringdown_prc_audit_v1` from an operator PMU frequency ringdown CSV;
- `scpn_ibr_ride_through_prc029_audit_v1` from an operator voltage/frequency
  ride-through CSV.

The builder verifies the source JSON SHA-256 metadata, exact child schema,
review-only claim boundary, and each child `content_hash` before sealing the
bundle as `scpn_power_grid_prc_audit_bundle_v1`. The bundle keeps the full child
records for assessor replay and carries no live-actuation or conformity claim.

::: scpn_phase_orchestrator.assurance.power_grid_prc_bundle
