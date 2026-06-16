# Cardiac Rhythm Domainpack

Maps cardiac electrophysiology to SPO's Kuramoto/UPDE framework.

## Why Kuramoto Fits This Domain

Gap-junction (connexin-43) coupling between cardiac cells is literally
electrical Kuramoto coupling.  Coupling constant is proportional to gap
junction conductance.  The SA node entrains atrial and ventricular tissue
exactly as a high-frequency Kuramoto oscillator entrains slower ones
(Strogatz 2003, "Sync").

## Layers

| Layer | Oscillators | Channel | bpm | Purpose |
|-------|------------|---------|-----|---------|
| sa_node | 3 | P (hilbert) | 35–70 | Primary/secondary/subsidiary pacemakers |
| atrial_conduction | 2 | P (hilbert) | 40–45 | AV node + His bundle |
| ventricular_depolarization | 2 | I (event) | 25–30 | Septum + freewall activation |
| repolarization | 3 | I (event) | ~20 | Apex/base/RVED recovery |

## Boundaries

- **heart_rate**: 40–180 bpm (hard)
- **qt_interval**: < 500 ms (hard) — Torsade de Pointes risk (Roden 2004)
- **st_deviation**: ±0.2/0.3 mV (soft) — ischemia indicator
- **pr_interval**: < 200 ms (soft) — first-degree AV block

## Actuators

| Actuator | Knob | Physical Meaning |
|----------|------|-----------------|
| pacing_rate | zeta | External pacemaker drive |
| drug_coupling | K | Antiarrhythmic drug coupling |
| vagal_stimulation | alpha | Vagal nerve SA lag modulation |
| target_rhythm | Psi | Target pacing phase |

## Value-Alignment Guard

The binding spec includes a `value_alignment` template for review-time
supervisor actuation checks. It limits global pacing drive, drug-coupling
strength, and target phase steps, then forces a safe-hold fallback
(`zeta = 0`) when a proposed action violates those priors or fails the
minimum score.

This is an auditable guard template for simulation and policy review. It is
not a medical-device controller.

## Imprint

Drug accumulation (digoxin t½=36h, amiodarone t½=58d): repeated dosing
builds imprint that modulates coupling, representing pharmacokinetics.

## Scenario

250 steps: normal sinus → PVCs → sustained VT → drug intervention →
overdrive pacing → sinus recovery.

## Operator Runbook

Use this pack as a non-actuating clinical simulation and review artefact only.
The local CLI intentionally refuses an admitted audit run for the `clinical`
safety tier unless an explicit reviewed deployment gate is present.

### 1. Validate the Binding

```bash
spo validate domainpacks/cardiac_rhythm/binding_spec.yaml
```

Expected result: the binding is valid and reports the `clinical` safety tier.

### 2. Run the Pack-Owned Scenario

```bash
PYTHONPATH=src python domainpacks/cardiac_rhythm/run.py
```

Expected result: the deterministic scenario executes end to end and prints a
compact JSON summary. Treat the output as simulation evidence for review, not
as pacing, drug, or vagal-stimulation guidance.

### 3. Confirm the Local Runtime Refusal

```bash
mkdir -p /tmp/spo_cardiac_rhythm_runbook
spo run domainpacks/cardiac_rhythm/binding_spec.yaml \
  --steps 40 \
  --audit /tmp/spo_cardiac_rhythm_runbook/audit.json \
  > /tmp/spo_cardiac_rhythm_runbook/blocked-run.txt 2>&1
```

Expected result: the command is blocked by the clinical safety tier before an
audit log is admitted. This is the correct local posture for a medical-style
domainpack. Keep `blocked-run.txt` with deployment evidence when auditing the
boundary.

Because no audit JSON is admitted in this local path, do not run `spo replay`,
`spo report`, or `spo explain` against this directory. Those commands require
a reviewed, admitted audit log from a gated environment.

### 4. Export the Review Policy Model

```bash
spo formal-export domainpacks/cardiac_rhythm/binding_spec.yaml \
  --export policy \
  --output /tmp/spo_cardiac_rhythm_runbook/policy.prism
```

Expected result: `policy.prism` is written for offline policy review. Full
formal packages and STL monitor exports remain blocked until this binding adds
`protocol_net` and `stl_monitors` evidence.

### 5. Live-Use Readiness Boundary

Do not connect this pack to ECG hardware, pacemakers, pumps, stimulators, or
clinical workflow systems from the local runbook. A live or clinical deployment
needs, at minimum:

- reviewed safety case and clinical governance for the intended use;
- hardware adapter provenance and fail-safe isolation evidence;
- admitted audit-log environment with replay/report/explain verification;
- external formal-review packet for the policy and monitor surfaces;
- operator approval, version pinning, and rollback procedure;
- explicit statement that SPO does not make diagnosis or treatment decisions.

## Hierarchy Sync Demo

`hierarchy_sync_demo.py` demonstrates the transport-neutral edge/cloud summary
path for this domainpack. It wraps pacemaker/atrial and ventricular/recovery
coherence summaries in deterministic sync envelopes, ingests them at a parent
node, and emits the resulting reduced parent plan.

Run the replay with:

```bash
PYTHONPATH=src python domainpacks/cardiac_rhythm/hierarchy_sync_demo.py
```

The emitted JSON is intentionally reduced: envelopes include source node,
sequence, protocol version, `R`, `Psi`, regime, confidence, and metadata, but no
raw conduction phase series, coupling matrices, or actuator targets. It is a
simulation/replay artefact, not a medical-device control path.

## Hierarchy Transport Demo

`hierarchy_transport_demo.py` validates the three hierarchy transport boundaries
for this domainpack: REST payload, WebSocket-style frame, and JSONL replay.

Run the transport replay with:

```bash
PYTHONPATH=src python domainpacks/cardiac_rhythm/hierarchy_transport_demo.py
```

The payload is intentionally reduced and transport-focused. It includes:
- `rest_boundary`, `websocket_frame`, and `jsonl_replay` audit records
- reduced child summaries (no raw phases, coupling matrices, or actuators)
- deterministic source/sequence ordering for transport replay slices

## Causal Attribution Demo

`causal_attribution_demo.py` runs a paired counterfactual for a deterministic
ventricular disturbance:

- baseline: continue from the same phases with no new supervisor action
- intervention: apply a candidate global pacing drive through `zeta`

The demo prints an audit payload containing both trajectories and a compact
attribution record. The attribution record labels the candidate action as
`stabilising`, `neutral`, or `destabilising` from the measured final and mean
order-parameter deltas. It is a replay/simulation review aid, not a live
medical-device decision path.

Run:

```bash
PYTHONPATH=src python domainpacks/cardiac_rhythm/causal_attribution_demo.py
```
