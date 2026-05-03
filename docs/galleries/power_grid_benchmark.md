# SPDX-FileCopyrightText: © Concepts 1996–2026 Miroslav Šotek.
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Code 2020–2026 Miroslav Šotek.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Power Grid Head-to-Head Benchmark

# Power Grid Head-to-Head Benchmark

This is the first deferred v1.0 domain benchmark page comparing the `power_grid`
domainpack workflow against domain-specific baselines.

## Domain Mapping

- The swing equation can be written as a second-order Kuramoto model:
  `dδ/dt = ω`, `dω/dt = (P_m - P_e - Dω)/(2H)`.
- PMU phase angles are the oscillator phases; line admittance is equivalent to
  coupling strength.
- Main source: Dorfler, Chertkov, Bullo (2013), *Synchronization in complex
  oscillator networks and smart grids*.

## Benchmark Setup

- File: `domainpacks/power_grid/run.py`
- Scenario:
  - `steady-state` (steps `0..49`)
  - `load-step` (steps `50..99`)
  - `renewable-ramp` (steps `100..149`)
  - `generator-trip` (step `150`)
  - `AGC-restore` (steps `175..249`)
- Runtime: 250 steps, `dt=0.002` (from `domainpacks/power_grid/binding_spec.yaml`)
- Metrics:
  - `R_good`: layers `[generator_rotor, area_frequency]`
  - `R_bad`: layers `[load_demand, renewable_intermittency]`

## Head-to-Head Results

### SPO benchmark runs (same scenario)

| Mode | Final `R_good` | Final `R_bad` | Final regime | Notes |
|---|---:|---:|---|---|
| `policy-disabled` | 0.4798 | 0.9874 | nominal | supervisor and policy actions disabled |
| `policy-enabled` | 0.4633 | 0.9852 | nominal | default policy + supervisor |

### Baseline references

- `notebooks/17_power_grid_stability.ipynb` uses an inertial reference model (4-bus, no policy) and reports:
  - Balanced baseline: `R≈0.99873`, frequency deviation `0.00150 Hz` after 500 steps
  - Generator trip: `R≈0.99914`, max transient `0.04288 Hz`
  - Weak coupling: `R≈0.20856`, frequency dev `0.10088 Hz`
- This baseline is a domain-specific sanity comparator; its network size and fault
  schedule are different from `domainpacks/power_grid/run.py`.

## Repro Steps

```bash
PYTHONPATH=src python domainpacks/power_grid/run.py
```

To reproduce the `policy-disabled` run for this same scenario:

```bash
PYTHONPATH=src python - <<'PY'
import runpy
from scpn_phase_orchestrator.supervisor.policy import SupervisorPolicy
from scpn_phase_orchestrator.supervisor import policy_rules

SupervisorPolicy.decide = lambda self, upde_state, boundary_state: []
policy_rules.PolicyEngine.evaluate = lambda self, regime_id, upde_state, good_layers, bad_layers: []
runpy.run_path("domainpacks/power_grid/run.py", run_name="__main__")
PY
```

## Interpretation

- `power_grid` remains in `nominal` regime in both settings over this scenario.
- With this short horizon the primary observable difference is a small change in
  final synchrony (`R_good`) around `0.48` vs `0.46`; both runs avoid hard
  boundary violations.
- Ongoing work is to add a production-equivalent grid model with fixed-size baseline
  and equivalent disturbance schedule for stricter statistical comparison.
