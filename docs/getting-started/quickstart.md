# Quickstart

Five minutes from install to synchronized oscillators.

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

## 5. Loading a Binding Spec Programmatically

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

## 6. Stuart-Landau Mode

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

- [Tutorial: Hello World](hello_world.md) -- build a custom 4-oscillator domain from scratch
- [Concepts: System Overview](../concepts/system_overview.md) -- full pipeline architecture
- [API Reference](../reference/api/index.md) -- complete Python API
- [Domainpack Gallery](../galleries/domainpack_gallery.md) -- all 33 domainpacks
