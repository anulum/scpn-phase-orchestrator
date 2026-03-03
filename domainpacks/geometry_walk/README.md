# Geometry Walk Domainpack

Graph-walk coherence domain for SPO's Kuramoto/UPDE framework.

## Why Kuramoto Fits This Domain

Random walkers on a graph synchronise when coupling exceeds a critical
threshold related to the graph's spectral gap.  Phase = ring-mapped node
index (theta = 2*pi*s/N).  Clustering and fragmentation transitions map
to Kuramoto order parameter bifurcations.  Jadbabaie, Lin, Morse (IEEE
TAC 2003) proved consensus on graphs via coupled-oscillator dynamics.

## Layers

| Layer | Oscillators | Channel | Purpose |
|-------|------------|---------|---------|
| local | 4 | S (symbolic) | Individual random walkers |
| global | 4 | S (symbolic) | Aggregate cluster coherence |

## Boundaries

- **R_floor**: R >= 0.1 (soft) -- minimum coherence for meaningful dynamics

## Actuators

| Actuator | Knob | Physical Meaning |
|----------|------|-----------------|
| coupling_global | K | Graph adjacency coupling strength |

## Imprint

None. Memoryless random walk dynamics.

## Scenario

100 steps: dispersed walkers -> coupling increase -> cluster emergence ->
fragmentation event -> re-convergence.
