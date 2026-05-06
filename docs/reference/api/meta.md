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

Larger replay corpora can be loaded from multiple audit JSONL files with
`CrossDomainMetaTransfer.fit_audit_history()`. The fitted model exposes an
audit-ready `training_summary` with record count, domain count, feature keys,
knob keys, and reward range. Use `to_json_package()` and
`from_json_package()` to save and restore a deterministic review package for
proposal jobs.

::: scpn_phase_orchestrator.meta.transfer
