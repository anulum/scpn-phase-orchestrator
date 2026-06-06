# Quickstart

Five minutes from install to synchronized oscillators.

## Why Start Here

This quickstart exists to create one verifiable proof point before any broader
integration: a minimal run that is deterministic, auditable, and easy to
reproduce.

For teams evaluating SPO, this page answers three immediate questions:

- Can our environment execute a coupled-oscillator simulation end-to-end?
- Do we get coherent outputs from known initial conditions?
- Can we verify the same output with fixed seeds and immutable input files?

Those three checks are intentionally practical. If they pass, the same domain
can move into CLI workflows, policy review, and production staging using the
later pages.

## 1. Install

```bash
pip install scpn-phase-orchestrator
```

For repository development, use the one-command setup and smoke run:

```bash
make quickstart
```

That target creates `.venv`, installs the development extras, validates
`domainpacks/minimal_domain/binding_spec.yaml`, runs a short audited
simulation, and prints a coherence report.

## 2. Your First Simulation

8 Kuramoto oscillators with uniform coupling, integrated for 500 RK4 steps:

```python
import numpy as np
from scpn_phase_orchestrator import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

engine = UPDEEngine(n_oscillators=8, dt=0.01, method="rk4")

theta = np.random.uniform(0, 2 * np.pi, 8)
omega = np.ones(8) * 1.0
knm = np.full((8, 8), 0.5)
np.fill_diagonal(knm, 0.0)
alpha = np.zeros((8, 8))

for _ in range(500):
    theta = engine.step(theta, omega, knm, 0.0, 0.0, alpha)

R, psi = compute_order_parameter(theta)
print(f"Order parameter R={R:.3f}")  # ~1.0 (synchronized)
```

`R` is the Kuramoto order parameter: 0.0 = incoherent, 1.0 = phase-locked.

The `step` signature: `step(phases, omegas, knm, zeta, psi_drive, alpha)` where `zeta` is external drive strength, `psi_drive` is drive target phase, and `alpha` is the phase-lag matrix.

## 3. Batch Run

`UPDEEngine.run()` executes N steps without Python loop overhead (uses Rust FFI when available):

```python
theta_final = engine.run(theta, omega, knm, 0.0, 0.0, alpha, n_steps=500)
R, _ = compute_order_parameter(theta_final)
print(f"R={R:.3f}")
```

## 4. Using the CLI with Domainpacks

Validate a binding spec:

```bash
spo validate domainpacks/minimal_domain/binding_spec.yaml
```

Inspect resolved runtime defaults before running:

```bash
spo inspect domainpacks/minimal_domain/binding_spec.yaml
```

Run a simulation and write an audit log:

```bash
spo run domainpacks/minimal_domain/binding_spec.yaml --steps 500 --audit audit.jsonl
```

Replay and verify deterministic integrity:

```bash
spo replay audit.jsonl --verify
```

Generate a coherence report:

```bash
spo report audit.jsonl
```

Run the real-data review demo:

```bash
spo demo --dataset heartbeat.csv --target coherence --steps 100
```

This command downloads the PhysioNet heart-rate-belt CSV for subject 3 from
Guy et al. (2024), "Respiratory and heart rate monitoring dataset from
aeration study", DOI `10.13026/e4dt-f689`; converts RR interval and heart-rate
columns into numeric time-series channels; runs the review-only auto-binding
proposal path; and prints the generated binding YAML plus dashboard commands.
It does not write actuator commands or start network services.

## 5. Running a Binding Spec from Python

Application code can use the high-level facade directly:

```python
from scpn import Orchestrator

orch = Orchestrator.from_yaml("domainpacks/minimal_domain/binding_spec.yaml")
state = orch.run(steps=100, seed=42)

print(f"Domain: {state.spec_name}")
print(f"Oscillators: {state.phases.size}")
print(f"R={state.order_parameter:.3f}")
```

The facade runs local research-tier Kuramoto binding specs and returns an
immutable state record with final phases, coupling matrices, natural
frequencies, and the Kuramoto order parameter. Use the lower-level
`StuartLandauEngine` directly for amplitude-mode studies.

## 6. Loading a Binding Spec Programmatically

```python
from scpn_phase_orchestrator.binding import load_binding_spec, validate_binding_spec

spec = load_binding_spec("domainpacks/queuewaves/binding_spec.yaml")
errors = validate_binding_spec(spec)
assert errors == []

n_osc = sum(len(layer.oscillator_ids) for layer in spec.layers)
print(f"Domain: {spec.name}")
print(f"Layers: {len(spec.layers)}")
print(f"Oscillators: {n_osc}")
print(f"Safety tier: {spec.safety_tier}")
```

`BindingSpec` is a dataclass with fields for layers, oscillator families, coupling, drivers, objectives, boundaries, actuators, imprint model, geometry prior, protocol net, and amplitude mode.

## 7. Stuart-Landau Mode

Phase + amplitude dynamics with subcritical bifurcation detection:

```python
from scpn_phase_orchestrator import StuartLandauEngine

sl = StuartLandauEngine(n_oscillators=4, dt=0.01)
# State vector: [theta_0..theta_3, r_0..r_3]
state = np.concatenate([
    np.random.uniform(0, 2 * np.pi, 4),  # phases
    np.ones(4) * 0.5,                      # amplitudes
])
omega = np.ones(4) * 1.0
mu = np.ones(4) * 1.0       # supercritical parameter
knm = np.full((4, 4), 0.3)
np.fill_diagonal(knm, 0.0)
knm_r = np.full((4, 4), 0.1)
np.fill_diagonal(knm_r, 0.0)
alpha = np.zeros((4, 4))

for _ in range(200):
    state = sl.step(state, omega, mu, knm, knm_r, 0.0, 0.0, alpha, epsilon=0.1)

phases = state[:4]
amplitudes = state[4:]
print(f"Amplitudes: {amplitudes.round(3)}")
```

## Next Steps

- [Minimal Domainpack in 5 Minutes](minimal_domainpack_5min.md) -- start from raw P/I/S source data
- [Troubleshooting](troubleshooting.md) -- install, notebooks, docs, FFI, validation, and audit replay fixes
- [Tutorial: Hello World](hello_world.md) -- build a custom 4-oscillator domain from scratch
- [Concepts: System Overview](../concepts/system_overview.md) -- full pipeline architecture
- [API Reference](../reference/api/index.md) -- complete Python API
- [Domainpack Gallery](../galleries/domainpack_gallery.md) -- all 36 domainpacks
- [Notebooks & Demos](../galleries/notebooks_and_demos.md) -- notebooks, examples, and interactive demos
- [Notebook Execution Matrix](../galleries/notebook_execution_matrix.md) -- notebook extras, CI status, and runtime expectations

## What this gives you after this page

After the quickstart, you should have these verifiable outcomes:

- a deterministic simulation seed and at least one bounded run,
- an auditable command trail (`spo validate`, `spo run`, `spo replay`, `spo report`),
- a reproducible import path for Python embedding (`from scpn_phase_orchestrator import ...`).

Treat the quickstart as the production-entry boundary, not the finish line.
Before moving to domain tuning or topology inference, confirm:

- replay output is stable under repeated execution,
- the default safety gates produce non-empty regime and action traces,
- lockfile and optional extras are aligned with your target profile.

## Recommended hardening sequence

1. Extend one use case at a time, not all channels.
2. Keep the first control policy conservative and reversible.
3. Add one monitor family at a time and compare false-positive rates before
   adding additional supervisors.
4. Promote from dashboard-only exploration to policy execution only after
   replay validation succeeds.

## When this path is complete

This page is intended to create three verifiable artifacts before any broader rollout:

- a deterministic minimal-domain run,
- a replay file that can be reverified by a separate operator,
- and a reviewed execution command list for the same environment.

Treat this as the entry gate for simulation-based confidence. If those three
artifacts do not align, keep the workspace in analysis mode and delay any
control-surface demonstration until the next pass.

## How to avoid premature promotion

The most common operational error is treating command success as evidence of
deployment readiness. This page prevents that by making command families
explicit:

- `spo validate`: structural and semantic entry checks,
- `spo run`: bounded execution output,
- `spo replay --verify`: reproducibility check,
- `spo report`: audit summary suitable for peer review.

Promotion is only appropriate when all four commands are repeatable in the same
runtime lane.
