# bio_stub

Biological domain stub. SCPN-compatible oscillator template.

## Setup

16 oscillators across 4 hierarchy layers (cellular, tissue, organ, systemic). Uses all 3 channels (P, I, S). Includes imprint model for history-dependent coupling.

## Layers

| Layer | Oscillators | Timescale |
|-------|-------------|-----------|
| cellular (0) | Ca2+ transients, membrane potential, metabolic cycles, mitotic clock | ms-s |
| tissue (1) | ECM remodelling, vasculature tone, immune patrol, neural firing | s-min |
| organ (2) | cardiac cycle, respiratory rhythm, hepatic clearance, renal filtration | s-min |
| systemic (3) | circadian, hormonal, autonomic, global immune | hr-day |

## Objective

Maximise coherence on tissue, organ, and systemic layers. No explicit bad layers.

## Run

```bash
spo run domainpacks/bio_stub/binding_spec.yaml --steps 200
python domainpacks/bio_stub/run.py
```
