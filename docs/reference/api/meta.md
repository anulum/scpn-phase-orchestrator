# Meta-Transfer

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

::: scpn_phase_orchestrator.meta.transfer
