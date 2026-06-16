# Chemical Reactor Domainpack

Maps CSTR (continuous stirred-tank reactor) oscillation dynamics to SPO's
Kuramoto/UPDE framework.

## Why Kuramoto Fits This Domain

CSTR systems undergo Hopf bifurcations where concentration and temperature
oscillate with well-defined phase relationships.  The heat-mass coupling
(Arrhenius rate ↔ heat generation) creates limit cycles that are naturally
modelled as coupled oscillators.  Coolant-jacket dynamics add a second
timescale whose phase relationship to the reactor determines stability.

Fogler (2020) "Elements of Chemical Reaction Engineering" Ch. 12.

## Layers

| Layer | Oscillators | Channel | Purpose |
|-------|------------|---------|---------|
| reaction_kinetics | 3 | P (hilbert) | Concentration oscillations (Hopf) |
| heat_transfer | 3 | P (hilbert) | Reactor/jacket/coolant temperatures |
| pressure_vessel | 2 | I (event) | Headspace pressure + vent |
| feed_flow | 2 | I (event) | Feed and coolant flow rates |

## Boundaries

- **temperature_limit**: < 450°C (hard) — Semenov ignition
- **pressure_limit**: < 15 bar (hard) — ASME Section VIII MAWP
- **concentration_floor**: > 0.1 (soft) — reactant depletion
- **coolant_flow_min**: > 10 LPM (soft) — minimum cooling

## Actuators

| Actuator | Knob | Physical Meaning |
|----------|------|-----------------|
| coolant_flow | K | Coolant valve position |
| feed_rate | zeta | Feed pump speed |
| agitator_speed | alpha | Mixing intensity |
| jacket_setpoint | Psi | Jacket temperature SP |

## Value-Alignment Guard

The binding spec includes a `value_alignment` template for review-time
process-control checks. It bounds coolant-flow, feed-rate, agitator, and
jacket-setpoint steps, then falls back to a zero-feed safe hold when a
candidate action exceeds those priors.

This template is for simulation, replay, and policy review. It is not a live
process safety instrumented system.

## Imprint

Catalyst deactivation / fouling: very slow decay (months timescale)
modulates coupling as active surface area decreases.

## Scenario

250 steps: stable CSTR → Hopf bifurcation onset → coolant failure
transient → emergency quench → feed rate recovery with fouling.

## Operator Runbook

Use this sequence for local review evidence. It validates the binding, runs the
pack-owned non-actuating scenario, records the expected local-runtime refusal
for the production-tier binding, and exports a policy model for formal review.

```bash
mkdir -p /tmp/spo-chemical-reactor
.venv/bin/spo validate domainpacks/chemical_reactor/binding_spec.yaml
.venv/bin/python domainpacks/chemical_reactor/run.py \
  > /tmp/spo-chemical-reactor/scenario.txt
set +e
.venv/bin/spo run domainpacks/chemical_reactor/binding_spec.yaml \
  --steps 500 \
  --seed 42 \
  --audit /tmp/spo-chemical-reactor/run.jsonl \
  > /tmp/spo-chemical-reactor/blocked-run.txt 2>&1
test $? -ne 0
set -e
.venv/bin/spo formal-export domainpacks/chemical_reactor/binding_spec.yaml \
  --export policy \
  --output /tmp/spo-chemical-reactor/policy.prism
```

Expected artefacts:

- `/tmp/spo-chemical-reactor/scenario.txt`: deterministic local scenario output
  from this pack's `run.py`.
- `/tmp/spo-chemical-reactor/blocked-run.txt`: expected refusal showing that the
  local runtime will not execute a production-tier binding through
  `spo run --audit`.
- `/tmp/spo-chemical-reactor/policy.prism`: policy export for formal review.

`spo replay`, `spo report`, and `spo explain` require an audit log from an
admitted execution path. They are intentionally not part of the current local
production-tier command chain. Full formal-package and STL exports also remain
blocked until this binding defines a `protocol_net` and `stl_monitors`.

Do not use this runbook as a live DCS, SIS, or process-safety controller. The
actuators above map to coolant-flow, feed-rate, agitator, and jacket-setpoint
concepts for review only. Any reactor connection remains blocked until a senior
operator provides all of the following evidence:

- read-only historian or laboratory-data mapping for every oscillator and
  boundary in `binding_spec.yaml`;
- proof that the first connector path cannot write DCS setpoints, valve
  positions, pump speeds, agitator speeds, or SIS state;
- independent SIS, relief, interlock, and manual shutdown protection outside SPO
  for temperature, pressure, coolant-flow, and concentration boundaries;
- operator-approved target hashes for the reviewed binding, connector plan, and
  audit-storage location;
- site-specific management-of-change, HAZOP/LOPA review, commissioning window,
  and rollback plan for any advisory workflow.
