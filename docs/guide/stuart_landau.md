# Stuart-Landau Amplitude Mode

*Added in v0.4.*

Standard Kuramoto tracks phase only. Stuart-Landau extends each oscillator with
an amplitude variable `r_i`, yielding coupled phase-amplitude ODEs. Use this
when the system's amplitude dynamics are physically meaningful -- cardiac
oscillations, EEG rhythms, laser arrays, or any domain where oscillators can
die (amplitude collapse) or grow.

## Equations

Phase (unchanged from Kuramoto, Acebron et al. 2005, Rev. Mod. Phys. 77, 137):

```
dtheta_i/dt = omega_i + sum_j K_ij sin(theta_j - theta_i - alpha_ij) + zeta sin(Psi - theta_i)
```

Amplitude:

```
dr_i/dt = (mu_i - r_i^2) * r_i + epsilon * sum_j K^r_ij * r_j * cos(theta_j - theta_i)
```

State vector layout: `state[:N]` = phases, `state[N:]` = amplitudes (total 2N).
Post-step invariants: phases wrapped to `[0, 2*pi)`, amplitudes clamped to
`max(0, r_i)`.

Key limits:

- `mu > 0`, uncoupled: `r -> sqrt(mu)` (supercritical limit cycle)
- `mu < 0`, uncoupled: `r -> 0` (subcritical decay)
- `epsilon = 0`: amplitude equation decouples, phase equation reduces to standard Kuramoto

## Enabling via binding_spec.yaml

Add an `amplitude:` block to the binding spec:

```yaml
amplitude:
  mu: 0.1
  epsilon: 0.01
  amp_coupling_strength: 0.3   # optional, defaults to 0.0
  amp_coupling_decay: 0.3      # optional, defaults to 0.3
```

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `mu` | float | required | Hopf bifurcation parameter. Positive = limit cycle, negative = decay |
| `epsilon` | float | required | Amplitude coupling strength multiplier |
| `amp_coupling_strength` | float | 0.0 | Base strength for `K^r` matrix construction |
| `amp_coupling_decay` | float | 0.3 | Exponential decay for `K^r` inter-layer entries |

When `amplitude` is present, the CLI `spo run` command automatically switches
to `StuartLandauEngine` and reports `mean_amplitude` alongside `R_good`/`R_bad`.

## Python API

```python
import numpy as np
from scpn_phase_orchestrator import StuartLandauEngine, CouplingBuilder

n = 8
builder = CouplingBuilder()
coupling = builder.build_with_amplitude(n, 0.45, 0.3, 0.3, 0.3)

engine = StuartLandauEngine(n_oscillators=n, dt=0.01, method="rk4")
mu = np.full(n, 0.1)
omegas = np.ones(n)
r_init = np.sqrt(np.maximum(mu, 0.0))
state = np.concatenate([np.random.uniform(0, 2*np.pi, n), r_init])

for _ in range(500):
    state = engine.step(
        state, omegas, mu,
        coupling.knm, coupling.knm_r,
        zeta=0.0, psi=0.0, alpha=coupling.alpha,
        epsilon=0.05,
    )

R, psi = engine.compute_order_parameter(state)
mean_r = engine.compute_mean_amplitude(state)
```

The order parameter `Z = mean(r_i * exp(i*theta_i))` is amplitude-weighted.
When all amplitudes are equal this reduces to standard Kuramoto `R`.

## Phase-Amplitude Coupling (PAC) Analysis

PAC quantifies how low-frequency phase modulates high-frequency amplitude --
a signature of cross-frequency interaction in neural and physical oscillators.

### modulation_index

Tort et al. 2010, J. Neurophysiol. Bins amplitude by phase, computes KL
divergence from uniform. Returns MI in [0, 1].

```python
from scpn_phase_orchestrator.upde.pac import modulation_index

mi = modulation_index(theta_low, amp_high, n_bins=18)
```

### pac_matrix

N x N matrix where entry `[i, j]` = `modulation_index(phase_i, amplitude_j)`.
Requires `(T, N)` phase and amplitude histories.

```python
from scpn_phase_orchestrator.upde.pac import pac_matrix

mat = pac_matrix(phases_history, amplitudes_history, n_bins=18)
```

### pac_gate

Binary gate: returns `True` when PAC exceeds a threshold.

```python
from scpn_phase_orchestrator.upde.pac import pac_gate

active = pac_gate(mi, threshold=0.3)
```

## Modulation Envelope

`extract_envelope` computes a sliding-window RMS of amplitude history.
`envelope_modulation_depth` returns `(max - min) / (max + min)` in [0, 1].

```python
from scpn_phase_orchestrator.upde.envelope import (
    extract_envelope, envelope_modulation_depth,
)

env = extract_envelope(amplitudes_history, window=10)
depth = envelope_modulation_depth(env)
```

## PAC-Driven Policy Rules

The policy DSL exposes amplitude-mode metrics for boundary and policy
evaluation:

| Metric | Source | Meaning |
|--------|--------|---------|
| `pac_max` | max MI across oscillators (last 20 steps) | Cross-frequency coupling strength |
| `mean_amplitude` | `mean(r)` across all oscillators | System energy level |
| `subcritical_fraction` | fraction of oscillators with `r < 0.1` | Death indicator |

Example `policy.yaml` rule:

```yaml
rules:
  - name: amplitude_collapse
    regime: [nominal, degraded]
    conditions:
      - metric: subcritical_fraction
        op: ">"
        threshold: 0.5
    action:
      knob: K
      scope: global
      value: 0.1
      ttl_s: 10.0
      justification: ">50% oscillators subcritical"
```

## Domainpacks with Amplitude Configs

Several domainpacks ship with `amplitude:` blocks:

| Domainpack | mu | epsilon | Use case |
|------------|-----|---------|----------|
| `neuroscience_eeg` | 0.1 | 0.05 | EEG cross-frequency coupling |
| `cardiac_rhythm` | 0.2 | 0.03 | Cardiac pacemaker cells |
| `plasma_control` | 0.15 | 0.02 | Tokamak MHD mode amplitude |
| `firefly_swarm` | 0.08 | 0.1 | Bioluminescent synchronisation |
| `rotating_machinery` | 0.3 | 0.01 | Rotor vibration monitoring |
| `power_grid` | 0.1 | 0.02 | Generator voltage amplitude |

## Rust Acceleration

When `spo_kernel` is installed, `StuartLandauEngine` delegates to
`PyStuartLandauStepper` transparently. The Rust path uses the same Dormand-Prince
RK45 implementation as the Python path, with identical numerical results.

```python
engine = StuartLandauEngine(n_oscillators=16, dt=0.01, method="rk4")
# Automatically uses Rust if spo_kernel is importable
```

## Integration Methods

The same three integrators available in `UPDEEngine` apply to
`StuartLandauEngine`: `euler`, `rk4`, `rk45`. The 2N state vector doubles
the derivative cost per evaluation relative to phase-only Kuramoto.

See [UPDE Numerics](../specs/upde_numerics.md) for Butcher tableaux and
adaptive step-size details.
