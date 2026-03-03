# Minimal Domain Domainpack

Minimal 2-layer, 4-oscillator test harness for SPO's Kuramoto/UPDE framework.

## Why Kuramoto Fits This Domain

Reference implementation exercising every pipeline component with the
smallest possible configuration: CouplingBuilder, UPDEEngine,
BoundaryObserver, RegimeManager, SupervisorPolicy, PolicyEngine.
Useful as a template for new domain authors and as a regression test.

## Layers

| Layer | Oscillators | Channel | Purpose |
|-------|------------|---------|---------|
| lower | 2 | P (physical) | Fast oscillators |
| upper | 2 | P (physical) | Slow oscillators |

## Boundaries

- **R_floor**: R >= 0.1 (soft) -- minimum coherence

## Actuators

| Actuator | Knob | Physical Meaning |
|----------|------|-----------------|
| coupling_global | K | Global coupling strength |

## Imprint

None. Stateless minimal dynamics.

## Scenario

100 steps: all oscillators in good_layers, engine converges to high R.

## Run

```bash
spo run domainpacks/minimal_domain/binding_spec.yaml --steps 100
python domainpacks/minimal_domain/run.py
```
