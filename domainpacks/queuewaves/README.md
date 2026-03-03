# Queuewaves Domainpack

Queue-wave desynchronisation domain for SPO's Kuramoto/UPDE framework.

## Why Kuramoto Fits This Domain

Service queues exhibit oscillatory behaviour under periodic demand.
Phase relationships between upstream and downstream queues determine
system throughput.  When queues synchronise (retry storm), correlated
burst traffic overwhelms backends.  Breaking phase-lock (desync) is
the control objective -- inverse of classical Kuramoto.

## Layers

| Layer | Oscillators | Channel | Purpose |
|-------|------------|---------|---------|
| micro | 3 | P (physical) | queue_a, queue_b, retry_burst depths |
| meso | 1 | P (physical) | P99 latency indicator |
| macro | 2 | I (informational) | error_rate, throughput health |

## Boundaries

- **queue_overflow**: queue_depth < 10000 (hard) -- backpressure limit
- **latency_warning**: p99_latency < 500 ms (soft) -- SLA threshold

## Actuators

| Actuator | Knob | Physical Meaning |
|----------|------|-----------------|
| coupling_global | K | Inter-service coupling strength |
| lag_micro | alpha | Retry backoff (phase shift) |
| damping | zeta | Circuit breaker / rate limiting |

## Imprint

Service degradation: memory leaks, connection pool exhaustion, and GC
pressure accumulate and modulate coupling, representing gradual system decay.

## Scenario

200 steps: steady state -> traffic spike (2x load) -> retry storm (micro
sync) -> circuit breaker -> recovery.
