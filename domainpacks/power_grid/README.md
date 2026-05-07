# Power Grid Domainpack

Maps interconnected power system dynamics to SPO's Kuramoto/UPDE framework.

## Why Kuramoto Fits This Domain

The swing equation `dδ/dt = ω, dω/dt = (P_m - P_e - Dω)/(2H)` is a
second-order Kuramoto model.  PMU voltage phasor angle IS the oscillator
phase — no extraction needed.  Coupling constant equals line admittance.

Dorfler, Chertkov, Bullo (2013) "Synchronization in complex oscillator
networks and smart grids" proved this equivalence rigorously.

## Layers

| Layer | Oscillators | Channel | Purpose |
|-------|------------|---------|---------|
| generator_rotor | 3 | P | Machine rotor angles |
| area_frequency | 2 | P | Control area deviations |
| tie_line | 2 | I | Inter-area power flows |
| load_demand | 3 | I | Aggregate load fluctuations |
| renewable_intermittency | 2 | I | Solar/wind variability |

## Boundaries

- **frequency_deviation**: ±0.5 Hz (hard) — NERC BAL-003-2
- **voltage_magnitude**: 0.95–1.05 pu (hard) — ANSI C84.1
- **rotor_angle**: < 90° (hard) — transient stability limit

## Actuators

| Actuator | Knob | Physical Meaning |
|----------|------|-----------------|
| governor_droop | K | Governor droop gain |
| agc_bias | zeta | AGC frequency bias |
| load_shedding | alpha | Under-frequency load shed |
| renewable_curtailment | Psi | Curtailment anti-phase signal |

## Value-Alignment Guard

The binding spec includes a `value_alignment` template for review-time grid
actuation checks. It bounds governor-droop, AGC-bias, load-shedding, and
curtailment phase steps, then falls back to a zero-bias safe hold when a
candidate control action exceeds those priors.

This template is for simulation, replay, and policy review. It is not a live
grid protection relay.

## Imprint

Transformer insulation aging: IEEE C57.91 thermal degradation model
accumulates and modulates coupling, representing gradual loss of
transmission capacity over years of operation.

## Scenario

250 steps: steady-state → sudden load step → renewable ramp → generator
trip fault → AGC + policy restore synchronism.

## FEP Hierarchy Demo

`fep_hierarchy_demo.py` provides a deterministic proof that the
`FEPPredictiveSupervisor` can run as a two-level hierarchy for this domainpack.
It evaluates generation and demand/renewable child regions independently, then
feeds their observed coherence into a parent FEP supervisor that emits a
hierarchy-level corrective action record.

Run the proof with:

```bash
PYTHONPATH=src python domainpacks/power_grid/fep_hierarchy_demo.py
```

The emitted JSON is audit-ready: each child record includes free energy,
surprise, observed/predicted `R`, and bounded `zeta` / `Psi` proposals, while
the parent record documents the aggregate N-channel hierarchy response.

## Hierarchy Sync Demo

`hierarchy_sync_demo.py` demonstrates the transport-neutral edge/cloud summary
path for this domainpack. It wraps generation-area and demand/renewable-area
coherence summaries in deterministic sync envelopes, ingests them at a parent
node, and emits the resulting reduced parent plan.

Run the replay with:

```bash
PYTHONPATH=src python domainpacks/power_grid/hierarchy_sync_demo.py
```

The emitted JSON is intentionally reduced: envelopes include source node,
sequence, protocol version, `R`, `Psi`, regime, confidence, and metadata, but no
raw PMU phase series, coupling matrices, or actuator targets.

## Causal Attribution Demo

`causal_attribution_demo.py` compares a no-action load-step rollout against a
bounded governor-droop coupling candidate. The output includes both
counterfactual trajectories and a compact causal attribution record.

Run it with:

```bash
PYTHONPATH=src python domainpacks/power_grid/causal_attribution_demo.py
```

The demo is intentionally replay-only: it proves that a proposed `K` actuation
can be audited against an unchanged baseline before any live grid adapter is
allowed to apply it.

## Morphogenetic Field Demo

`morphogenetic_field_demo.py` demonstrates reaction-diffusion-style topology
field shaping for a stressed grid replay. It keeps generator rotor and area
frequency phases near each other while tie-line, load-demand, and renewable
layers drift, then emits the next reviewable `K_nm` field audit payload.

Run the replay with:

```bash
PYTHONPATH=src python domainpacks/power_grid/morphogenetic_field_demo.py
```

The replay is non-actuating. It validates the domain binding spec, builds the
configured layer coupling, reports grown and shrunk topology-field edges, and
exports dependency-free field snapshot rows for audit or later UI rendering.
