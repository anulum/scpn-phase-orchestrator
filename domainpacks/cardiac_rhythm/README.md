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
