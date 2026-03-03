# Network Security Domainpack

DDoS attack as undesirable phase synchronisation; defensive desynchronisation.

## Why Kuramoto Fits This Domain

A DDoS attack concentrates packet arrival times, effectively synchronising
traffic oscillators to overwhelm a target.  The defense goal is the opposite:
desynchronise attack traffic while maintaining normal traffic coherence.
Kuramoto coupling models both the attacker's synchronisation mechanism and
the firewall's adaptive rate-limiting as counter-coupling.

## Layers

| Layer | Oscillators | Channel | Purpose |
|-------|------------|---------|---------|
| normal_traffic | 3 | I (event) | Legitimate client request patterns |
| attack_vector | 2 | I (event) | Coordinated DDoS packet floods |
| defense_response | 3 | P (hilbert) | Firewall rate-limiting phase |

## Boundaries

- **packet_loss_max**: < 0.1 (hard) -- legitimate packet drop rate
- **latency_max**: < 0.8 (soft) -- end-to-end request latency
- **false_positive_max**: < 0.05 (soft) -- false positive blocking rate

## Actuators

| Actuator | Knob | Physical Meaning |
|----------|------|-----------------|
| firewall_coupling | K | Defense coordination strength |
| defense_drive | zeta | Active traffic shaping drive |

## Scenario

200 steps: normal traffic flow -> DDoS burst at step 80 (attack layer
phases randomised) -> defense desynchronisation and recovery.
