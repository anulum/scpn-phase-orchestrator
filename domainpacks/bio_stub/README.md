# Bio Stub Domainpack

Multi-scale biological oscillator template for SPO's Kuramoto/UPDE framework.

## Why Kuramoto Fits This Domain

Biological systems oscillate at every scale: Ca2+ transients (ms), cardiac
rhythm (s), circadian clocks (24 h), hormonal cycles (days-weeks).  Multi-scale
coupling between cellular, tissue, organ, and systemic layers is inherently a
Kuramoto hierarchy.  Winfree (1967) first modelled biological oscillator
populations; Strogatz (2003) *Sync* Ch. 5 formalised the connection.

## Layers

| Layer | Oscillators | Channel | Timescale | Purpose |
|-------|------------|---------|-----------|---------|
| cellular | 4 | P (hilbert) | ms-s | Ca2+, membrane, metabolic, mitotic |
| tissue | 4 | I (event) | s-min | ECM, vasculature, immune, neural |
| organ | 4 | P (hilbert) | s-min | Cardiac, respiratory, hepatic, renal |
| systemic | 4 | S (symbolic) | hr-day | Circadian, hormonal, autonomic, immune |

## Boundaries

- **heart_rate**: 40-180 bpm (upper: hard, lower: soft) -- clinical range
- **circadian_deviation**: < 3 h (soft) -- circadian disruption threshold

## Actuators

| Actuator | Knob | Physical Meaning |
|----------|------|-----------------|
| entrainment | zeta | External pacemaker drive |
| coupling_global | K | Inter-scale coupling strength |
| reference_phase | Psi | Circadian reference phase |

## Imprint

Chronic exposure accumulation: repeated stimulation builds history-dependent
coupling modulation, representing biological plasticity.

## Scenario

200 steps: baseline homeostasis -> cellular stress -> tissue adaptation ->
organ-level compensation -> systemic recovery.
