# SPDX-FileCopyrightText: © Concepts 1996–2026 Miroslav Šotek.
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Code 2020–2026 Miroslav Šotek.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Cardiac Rhythm Head-to-Head Benchmark

# Cardiac Rhythm Head-to-Head Benchmark

This is the second deferred v1.0 domain benchmark page comparing the
`cardiac_rhythm` domainpack against a cardiac-specific reference script.

## Domain Mapping

- Cardiac tissue is modelled as coupled phase oscillators with gap-junction
  coupling (approximation from Guyton, Ch. 10, and Strogatz's Kuramoto framing).
- Layer structure maps pacemaking, atrial conduction, ventricular depolarisation,
  and repolarisation anatomy.
- `R_good` targets healthy sinus/atrial synchrony (`sa_node`, `atrial_conduction`).
- `R_bad` targets ventricular desynchronisation (`ventricular_depolarization`).

## Benchmark Setup

- File: `domainpacks/cardiac_rhythm/run.py`
- Scenario:
  - `sinus` (steps `0..49`)
  - `PVCs` (steps `50..99`)
  - `VT` (steps `100..149`)
  - `drug` (step `150..199`)
  - `pacing` (steps `200..249`)
- Runtime: 250 steps, `dt=0.002` (from `domainpacks/cardiac_rhythm/binding_spec.yaml`)
- Metrics:
  - `R_good`: layers `[sa_node, atrial_conduction]` (indices `[0, 1]`)
- `R_bad`: layers `[ventricular_depolarization]` (index `[2]`)

## Head-to-Head Results

### SPO benchmark runs (same scenario)

| Mode | Final `R_good` | Final `R_bad` | Final regime | Notes |
|---|---:|---:|---|---|
| `policy-disabled` | 0.6239 | 0.9999 | nominal | supervisor and policy actions disabled |
| `policy-enabled` | 0.6105 | 0.9994 | nominal | default policy + supervisor |

## Baseline references

- `examples/cardiac_rhythm.py` is a domain-specific 5-oscillator reference
  without policy/supervisor logic:
  - Normal sinus rhythm: `R=0.997`, `0.992`, `0.988`, `0.986` at 0.5 s intervals
  - AV block: `R=0.997`, `0.992`, `0.987`, `0.982`
  - External pacemaker: `R=0.964`, `0.986`, `0.892`, `0.973`
- This baseline differs in network size and state composition from `cardiac_rhythm`,
  but preserves the same clinical interpretation: entrainment quality from pacemaker
  and pathology under conduction stress.

## Repro Steps

```bash
PYTHONPATH=src python examples/cardiac_rhythm.py
```

```bash
PYTHONPATH=src timeout 240s python domainpacks/cardiac_rhythm/run.py
```

To reproduce the `policy-disabled` run for this scenario:

```bash
PYTHONPATH=src python - <<'PY'
import runpy
from scpn_phase_orchestrator.supervisor.policy import SupervisorPolicy
from scpn_phase_orchestrator.supervisor import policy_rules

SupervisorPolicy.decide = lambda self, upde_state, boundary_state: []
policy_rules.PolicyEngine.evaluate = lambda self, regime_id, upde_state, good_layers, bad_layers: []
runpy.run_path("domainpacks/cardiac_rhythm/run.py", run_name="__main__")
PY
```

## Interpretation

- `cardiac_rhythm` remains in `nominal` regime in both settings over this scenario.
- With the default supervisor, the run keeps `R_good` near policy-off execution while
  suppressing `R_bad` marginally.
- Baseline and full-domain setups are complementary; together they cover a small
  closed-loop orchestrator against a domain-focused entrainment toy reference.
