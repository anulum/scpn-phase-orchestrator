# queuewaves

Queue-wave desynchronisation domain. Models cloud retry storms as Kuramoto oscillators.

## Scenario

A microservice system with:

- **Micro layer (layer 0):** queue depths and retry bursts. High coherence here = retry storm (bad).
- **Meso layer (layer 1):** P99 latency. Indicator, not directly controlled.
- **Macro layer (layer 2):** error rate and throughput. High coherence here = coordinated service health (good).

## Objective

- Suppress R on layer 0 (R_bad) to break retry-storm synchronisation.
- Maintain R on layer 2 (R_good) to keep services coordinated.

## Policy

`policy.yaml` defines two rules:
1. If R_bad > 0.7, increase alpha on layer 0 (phase-shift retries).
2. If R_good < 0.3, increase K globally (restore coordination).

## Run

```bash
spo run domainpacks/queuewaves/binding_spec.yaml --steps 200
python domainpacks/queuewaves/run.py
```
