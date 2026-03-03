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
