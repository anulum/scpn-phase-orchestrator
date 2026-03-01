# geometry_walk

Graph-walk coherence domain. Phase derived from random walk position on a discrete graph.

## Setup

8 oscillators across 2 layers, all Symbolic channel (graph mode). Each oscillator represents a random walker on a 16-node graph. Phase = normalised position on the walk sequence.

## Objective

Maximise coherence across both layers. Tests the Symbolic extractor and graph-walk phase coupling.

## Run

```bash
spo run domainpacks/geometry_walk/binding_spec.yaml --steps 100
python domainpacks/geometry_walk/run.py
```
