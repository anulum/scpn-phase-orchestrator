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

## Value-Alignment Guard

The binding spec includes a `value_alignment` template for review-time
defence-control checks. It bounds firewall-coupling and active
traffic-shaping drive steps, then falls back to a zero-drive safe hold when a
candidate action exceeds those priors.

This template is for simulation, replay, and policy review. It is not a live
network security appliance policy.

## Scenario

200 steps: normal traffic flow -> DDoS burst at step 80 (attack layer
phases randomised) -> defense desynchronisation and recovery.

## Causal Attribution Demo

`causal_attribution_demo.py` compares a no-action lateral-movement replay
against a bounded firewall-coupling candidate. The output includes paired
baseline/intervention trajectories and an attribution record explaining whether
the candidate improved final and mean coherence.

Run it with:

```bash
PYTHONPATH=src python domainpacks/network_security/causal_attribution_demo.py
```

The demo is replay-only and non-actuating. It is not a live firewall,
rate-limiter, or network security appliance policy.

## Topology Adaptation Demo

`topology_adaptation_demo.py` derives pairwise support from deterministic
transfer-entropy traces, proposes higher-order topology mutations for
traffic/attack/defence cohorts, and records Lyapunov before/after energy
evidence for the candidate mutation.

Run it with:

```bash
PYTHONPATH=src python domainpacks/network_security/topology_adaptation_demo.py
```

The demo is replay-only and non-actuating. It emits an audit payload for
reviewing topology mutation evidence before any live security adapter is
allowed to apply a policy.

## Morphogenetic Field Demo

`morphogenetic_field_demo.py` demonstrates reaction-diffusion-style topology
field shaping for a lateral-movement replay. It grows normal-traffic and
defence-response coupling while shrinking attack-vector coupling, then emits a
field snapshot for audit review.

Run it with:

```bash
PYTHONPATH=src python domainpacks/network_security/morphogenetic_field_demo.py
```

The demo is replay-only and non-actuating. It is intended for topology-field
review, not live network enforcement.

## Sheaf Obstruction Demo

`sheaf_obstruction_demo.py` evaluates normal-traffic, attack-vector, and
defence-response cohorts as a directed cellular sheaf over traffic-rate,
threat-level, defence-phase, and trust-score channels. It compares a nominal
section against a replayed lateral-movement section and emits raw sheaf audit
records plus obstruction severity summaries.

Run the replay with:

```bash
PYTHONPATH=src python domainpacks/network_security/sheaf_obstruction_demo.py
```

The replay is non-actuating. It is intended for obstruction triage and
supervisor validation, not live network enforcement.
