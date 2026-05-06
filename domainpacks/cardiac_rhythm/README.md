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

## Imprint

Drug accumulation (digoxin t½=36h, amiodarone t½=58d): repeated dosing
builds imprint that modulates coupling, representing pharmacokinetics.

## Scenario

250 steps: normal sinus → PVCs → sustained VT → drug intervention →
overdrive pacing → sinus recovery.

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
