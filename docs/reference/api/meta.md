# Meta-Transfer

## Why this subsystem exists

This is a bootstrap surface for policy transfer between domains, not a production
adaptive controller. Its role is to make replay history reusable by giving operators
an interpretable first proposal from prior domains.

In enterprise contexts, this reduces “blank start” risk for new deployments: teams
can start from documented historical baselines rather than writing new policy
defaults from scratch.

## Decision point for operators

Meta-transfer is a **bootstrap, not a controller**. Its value is highest during:

- first-day onboarding of a new domainpack,
- recovery after a topology drift event,
- and cross-domain comparison when handoff teams need a first proposal quickly.

Because every output is review-only, teams preserve human authority over policy promotion while gaining a deterministic starting point from prior audited history.

## Governance rule

All proposals from this surface remain advisory until the normal policy stack approves them. Evidence packages should be treated as hypothesis-generating artefacts and validated through the same deterministic replay channels as any other control-change candidate.

The meta-transfer subsystem provides a deterministic first slice for
cross-domain policy bootstrapping. It reads replay or audit-derived records,
embeds domain metrics into a shared feature vector, and proposes initial
supervisor knobs from nearest historical neighbours.

This is not an online autonomous trainer. Proposals are reviewable starting
points for policy authors, and every proposal exposes neighbour evidence and a
serialisable audit record.

```python
from scpn_phase_orchestrator.meta import CrossDomainMetaTransfer, MetaPolicyRecord

records = (
    MetaPolicyRecord("power_grid", {"R_global": 0.4}, {"K": 0.08}),
    MetaPolicyRecord("cardiac", {"R_global": 0.8}, {"zeta": 0.05}),
)
model = CrossDomainMetaTransfer.fit(records)
proposal = model.propose({"R_global": 0.5})

audit_payload = proposal.to_audit_record()
```

Larger replay corpora can be loaded from explicit audit JSONL file lists with
`CrossDomainMetaTransfer.fit_audit_history()` or from nested audit directories
with `CrossDomainMetaTransfer.fit_audit_directory()`. Directory loading uses
`records_from_audit_directory()` and discovers `**/*.jsonl` by default, so
multi-domain replay corpora can be trained without hand-listing every audit
file. The fitted model exposes an audit-ready `training_summary` with record
count, domain count, feature keys, knob keys, and reward range. Use
`to_json_package()` and `from_json_package()` to save and restore a
deterministic review package for proposal jobs.
Audit JSONL ingestion and JSON package import reject non-finite constants,
duplicate object keys, and non-object package payloads before records enter
the nearest-neighbour proposal surface.
`to_package_manifest()` emits a packaging-readiness manifest for the optional
`scpn-meta` surface: it binds the deterministic JSON package SHA-256, public
import target, console-script name, and training summary while keeping
`execution_permitted=false`. It does not build, install, run, or upload a
package.

The same review-only manifest can be emitted from the CLI for release and
operator review jobs:

```bash
spo meta-transfer-manifest audit_grid.jsonl audit_cardiac.jsonl --min-records 2
spo meta-transfer-manifest --audit-directory audit_history --min-records 10
```

Both forms print manifest JSON to stdout unless `--output` is provided. The
command accepts explicit audit JSONL files or one nested audit directory, never
both, and still keeps `execution_permitted=false`; it does not build, install,
upload, or execute `scpn-meta`.

Installed packages also expose the same review-only command as `scpn-meta`.
This console script is intentionally narrow: it points to the manifest exporter,
not the full SPO runtime CLI, so packaging metadata matches the manifest without
adding a live training or execution surface.

## How teams typically use it

The operational path is usually:

1. Collect comparable replay corpus (or nested history directory),
2. Fit and inspect `training_summary`,
3. Generate proposals and review neighbour evidence,
4. Export a manifest for reproducible transfer handoff.

That sequence keeps transfer evidence, not just transfer parameters, part of the
release documentation.

## Leave-one-domain-out transfer sweep

Proposing knobs is one thing; claiming a detector **generalises across domains** is
a far stronger, and more easily overstated, claim. `meta.leave_one_domain_out` runs
the honest test: hold out each domain in turn, transfer the pooled remainder onto
it, and aggregate the per-fold verdicts under a rule that never upgrades. A single
domain that is detectable within-domain yet receives no transfer skill
(`transfer_negative`) refutes generality decisively — the recorded CHB-MIT
cross-subject negative must surface as `lodo_negative`, never a laundered aggregate
positive. Only an unbroken sweep of positive folds earns `lodo_generalises`; a
sweep with no detectable target at all is `lodo_untestable`; anything in between is
`lodo_inconclusive`.

```python
from scpn_phase_orchestrator.meta import (
    LeaveOneDomainOutFold,
    leave_one_domain_out_transfer,
)

report = leave_one_domain_out_transfer(
    [
        LeaveOneDomainOutFold("grid", transfer, within, controls),
        LeaveOneDomainOutFold("chbmit", chb_transfer, chb_within, chb_controls),
    ]
)
verdict = report.verdict  # e.g. "lodo_negative"
audit_payload = report.to_record()
```

Every arm is scored by the caller and audited through the same honest
`audit_cross_domain_transfer` calibration, so the sweep stays a pure, deterministic
aggregation with no hidden training step.

::: scpn_phase_orchestrator.meta.transfer

::: scpn_phase_orchestrator.meta.leave_one_domain_out
