# Traffic Flow Domainpack

Maps urban traffic signal coordination to SPO's Kuramoto/UPDE framework.

## Why Kuramoto Fits This Domain

Traffic signal coordination IS phase synchronisation.  Each signalised
intersection cycles with a period (~90 s), and offset-based green wave
coordination aligns phase differences between adjacent signals.
Gershenson & Rosenblueth (2012) showed self-organising traffic lights
converge via coupled-oscillator dynamics.

## Layers

| Layer | Oscillators | Channel | Cycle | Purpose |
|-------|------------|---------|-------|---------|
| intersection | 3 | P (hilbert) | ~90 s | Signal phase, queue, pedestrian |
| corridor | 3 | P (hilbert) | ~5 min | Green wave, travel time, bus priority |
| network | 2 | I (event) | ~15 min | Area flow, congestion index |
| demand | 2 | I (event) | ~1 h | Commuter, freight patterns |

## Boundaries

- **queue_overflow**: < 50 vehicles (hard) -- intersection capacity
- **travel_time**: < 300 s (soft) -- corridor level of service
- **pedestrian_wait**: < 120 s (soft) -- ADA compliance

## Actuators

| Actuator | Knob | Physical Meaning |
|----------|------|-----------------|
| cycle_length | K | Signal coordination coupling |
| offset | zeta | Green wave drive |
| split | alpha | Phase split at intersections |
| metering | Psi | Ramp metering target |

## Imprint

Signal controller timing drift and road surface degradation accumulate
over weeks-months, modulating both coupling strength and phase lag.

## Scenario

200 steps: free-flow -> morning rush (demand spike) -> queue spillback ->
adaptive signal control -> green wave recovery.

## Topology Adaptation Demo

`topology_adaptation_demo.py` demonstrates transfer-entropy-supported
higher-order topology mutation for signal coordination. The demo builds a
pairwise support matrix from deterministic corridor phase histories, applies a
`TopologyMutationPolicy` with a pairwise support floor, and emits an audit
payload for newly proposed 2-simplices.

Run it with:

```bash
PYTHONPATH=src python domainpacks/traffic_flow/topology_adaptation_demo.py
```

The output includes transfer-entropy support counts, the mutation policy, and
the supervisor topology audit record.
