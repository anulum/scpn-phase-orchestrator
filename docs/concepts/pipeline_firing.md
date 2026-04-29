<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Pipeline firing map -->

# How the Pipeline Fires

This page maps a domainpack run from YAML to actuation. Read it when the
system overview is still too abstract and you need to know what happens first,
what is held in memory, and what is emitted at each control step.

The short version:

```text
binding_spec.yaml
  -> BindingSpec validation
  -> oscillator family and layer resolution
  -> phase extraction or seeded initial phase probe
  -> coupling matrix and driver construction
  -> UPDE engine step
  -> monitor and boundary evaluation
  -> regime and policy evaluation
  -> action projection and actuation mapping
  -> audit.jsonl record
  -> replay/report consumers
```

## 1. Load the Domain Contract

The first file is always the binding spec:

```bash
spo validate domainpacks/<domain>/binding_spec.yaml
spo run domainpacks/<domain>/binding_spec.yaml --steps 1000 --audit audit.jsonl
```

Validation turns YAML into a `BindingSpec` object. The object is the run
contract: it does not contain live samples, but it names the oscillator
families, layers, objectives, boundaries, actuators, policy file, driver
parameters, and optional amplitude/imprint settings.

| Binding field | Runtime meaning |
| --- | --- |
| `layers` | Ordered hierarchy and oscillator identifiers. |
| `oscillator_families` | Channel, extractor type, units, and source metadata for each family. |
| `coupling` | Base strength, decay, templates, and topology constraints for `K_nm`. |
| `drivers` | External drive strength `zeta` and target phase `Psi`. |
| `objectives` | Which layers contribute to `R_good` and `R_bad`. |
| `boundaries` | Hard or soft limits checked after each monitor pass. |
| `actuators` | Knobs the supervisor is allowed to change. |
| `policy` | Optional declarative rule file evaluated by the policy engine. |

If the binding fails validation, the run stops before engine construction.

## 2. Resolve Oscillators and Channels

Each oscillator identifier in `layers[*].oscillator_ids` is matched to an
oscillator family. The default profile uses three channels:

| Channel | Source shape | Extractor role | Output |
| --- | --- | --- | --- |
| `P` | Continuous waveform | Hilbert, wavelet, or zero-crossing phase extraction. | `theta`, `omega`, quality, optional amplitude. |
| `I` | Event timestamps | Inter-event interval to ring phase. | `theta`, `omega`, quality. |
| `S` | State trace | State index or transition graph to symbolic phase. | `theta`, `omega`, quality. |

The engine only sees phase vectors, frequencies, coupling, and driver terms.
Domain semantics stay in the extractor and binding layer.

For normal online integrations, source adapters or custom extractors feed
`PhaseState` values into the pipeline. For the built-in CLI simulation path,
the current binding metadata is used to initialise a deterministic seeded phase
probe so a domainpack can be validated, run, audited, and replayed without
requiring live hardware.

## 3. Build Coupling and Drives

Before stepping the engine, the run constructs the control tensors:

```text
oscillator order -> theta[0:N], omega[0:N]
coupling block   -> K_nm[N,N], alpha[N,N]
drivers block    -> zeta, Psi
objectives       -> layer indices for R_good and R_bad
```

`K_nm` controls how strongly oscillator `j` pulls oscillator `i`.
`alpha` adds phase lag. `zeta` and `Psi` define an external phase drive.
The supervisor may later issue control actions that adjust these exposed
knobs, subject to actuator limits.

## 4. Step the Engine

At each integration step, an engine advances phase state:

```text
dtheta_i/dt = omega_i
            + sum_j K_ij * sin(theta_j - theta_i - alpha_ij)
            + zeta * sin(Psi - theta_i)
```

The default dense path uses `UPDEEngine`. Other engines keep the same
run shape but change the dynamics: sparse, inertial, delayed, geometric,
Stuart-Landau amplitude, swarmalator, hypergraph, splitting, and JAX paths.

The important contract is that every engine produces a compatible state for
downstream monitors: updated phases, order parameters, layer coherence, and
engine-specific metrics where enabled.

## 5. Monitor Coherence and Boundaries

The monitor pass converts phase state into decision variables:

| Monitor output | Used by |
| --- | --- |
| Global and layer order parameter `R` | Regime manager, reports, audit. |
| `R_good` and `R_bad` | Objective partition and policy rules. |
| Boundary states | Regime escalation and action projection. |
| Optional metrics such as PAC, Lyapunov, transfer entropy, chimera index | Domain-specific policies and reports. |

Hard boundary violations can escalate the run to `CRITICAL`. Soft boundary
violations are recorded and can drive policy actions without immediately
blocking the run.

## 6. Evaluate Supervisor and Policy

The supervisor layer evaluates the current state in this order:

1. `RegimeManager` maps coherence and boundary state to `NOMINAL`,
   `DEGRADED`, `CRITICAL`, or `RECOVERY`, with hysteresis to avoid
   one-step flapping.
2. `SupervisorPolicy` can emit default regime-driven actions.
3. `PolicyEngine` evaluates `policy.yaml` rules against metrics, regimes,
   cooldowns, and rule limits.
4. Optional protocol state machines, such as Petri nets, constrain when a
   transition or action is legal.

The output is a list of `ControlAction` objects. Actions are still abstract:
they say "adjust `K` globally" or "drive `Psi` for this scope", not "write
register 17 on device X".

## 7. Project and Map Actions

`ActionProjector` and `ActuationMapper` turn abstract actions into safe,
domain-specific commands:

```text
ControlAction
  -> check actuator exists
  -> clip to limits
  -> enforce rate limits and TTL
  -> validate against boundary constraints
  -> emit domain command or audit-only action
```

This is where a pump, qubit controller, queue throttle, grid controller, or
robotic swarm adapter receives commands. In offline runs, the same action is
kept in the audit trace without touching hardware.

## 8. Audit, Replay, and Report

Every step can append a JSONL record:

```json
{
  "step": 42,
  "regime": "DEGRADED",
  "R": [0.71, 0.44],
  "actions": [{"knob": "K", "scope": "global", "value": 0.15}],
  "boundary_violations": [],
  "prev_hash": "...",
  "hash": "..."
}
```

The hash chain makes the trace tamper-evident. Replay consumes the audit log
to verify deterministic state progression, and reports consume the same log to
explain regimes, actions, and boundary events.

## Setup-Time vs Step-Time Work

| Phase | Happens once | Happens every step |
| --- | --- | --- |
| Binding | Load and validate YAML. | None unless dynamic config reload is enabled by a caller. |
| Oscillators | Resolve family order and extractor configuration. | Extract or update `PhaseState` values. |
| Coupling | Build initial `K_nm` and `alpha`. | Apply imprint, plasticity, or supervisor adjustments. |
| Engine | Select engine variant and backend. | Integrate phases. |
| Monitor | Select enabled monitors and thresholds. | Compute metrics and boundary state. |
| Supervisor | Load policy rules and protocol nets. | Evaluate regime and actions. |
| Actuation | Bind actuator names and limits. | Project, clip, and emit commands. |
| Audit | Open trace writer. | Append chained record. |

## Where to Look Next

- [System Overview](system_overview.md) for the full architecture.
- [Oscillators: P / I / S Channels](oscillators_PIS.md) for extractor
  semantics.
- [Knobs: K, alpha, zeta, Psi](knobs_K_alpha_zeta_Psi.md) for control
  meanings.
- [Policy DSL](../specs/policy_dsl.md) for rule syntax.
- [Audit Trace](../specs/audit_trace.md) for replayable records.
