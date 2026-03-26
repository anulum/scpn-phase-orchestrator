# Tutorial: Hello World

Build a custom 4-oscillator domain from scratch, run it, inspect the audit trail, and plot the results.

## 1. Scaffold a New Domainpack

```bash
spo scaffold my_domain
```

This creates `domainpacks/my_domain/` with a template `binding_spec.yaml` and `README.md`.

## 2. Edit the Binding Spec

Replace `domainpacks/my_domain/binding_spec.yaml` with:

```yaml
name: my_domain
version: "0.1.0"
safety_tier: research
sample_period_s: 0.01
control_period_s: 0.1

layers:
  - name: fast
    index: 0
    oscillator_ids: [osc_0, osc_1]
  - name: slow
    index: 1
    oscillator_ids: [osc_2, osc_3]

oscillator_families:
  base:
    channel: P
    extractor_type: physical
    config: {}

coupling:
  base_strength: 0.45
  decay_alpha: 0.3
  templates: {}

drivers:
  physical:
    zeta: 0.0
    psi: 0.0
  informational: {}
  symbolic: {}

objectives:
  good_layers: [0, 1]
  bad_layers: []
  good_weight: 1.0
  bad_weight: 1.0

boundaries:
  - name: coherence_floor
    variable: R
    lower: 0.2
    upper: null
    severity: soft

actuators:
  - name: coupling_global
    knob: K
    scope: global
    limits: [0.0, 3.0]

policy: policy.yaml
```

Two layers (`fast`, `slow`), two oscillators each. A soft boundary triggers when the global order parameter R drops below 0.2.

## 3. Add Policy Rules

Create `domainpacks/my_domain/policy.yaml`:

```yaml
rules:
  - name: boost_coupling
    regime: [DEGRADED, RECOVERY]
    condition:
      metric: R_good
      layer: 0
      op: "<"
      threshold: 0.4
    action:
      knob: K
      scope: global
      value: 0.15
      ttl_s: 5.0

  - name: recover_boundary
    regime: [CRITICAL]
    condition:
      metric: R
      layer: 0
      op: "<"
      threshold: 0.2
    action:
      knob: zeta
      scope: global
      value: 0.3
      ttl_s: 3.0
```

Rule 1: in DEGRADED or RECOVERY regime, if layer 0 R drops below 0.4, bump global coupling by 0.15 for 5 seconds.

Rule 2: in CRITICAL regime, if R drops below 0.2, apply external drive (zeta=0.3) for 3 seconds to force re-synchronization.

## 4. Validate

```bash
spo validate domainpacks/my_domain/binding_spec.yaml
```

Expected output: `Valid`

If there are errors (missing fields, invalid channel names, etc.), each is printed with a description.

## 5. Run

```bash
spo run domainpacks/my_domain/binding_spec.yaml --steps 500 --audit my_audit.jsonl --seed 42
```

Output:

```
R_good=0.9876  R_bad=0.0000  regime=NOMINAL
```

`R_good` near 1.0 means the good-layer oscillators synchronized. `R_bad=0.0` because no bad layers were defined.

## 6. Inspect the Audit Log

`my_audit.jsonl` contains one JSON object per line. The first record is a header:

```json
{"type": "header", "n_oscillators": 4, "dt": 0.01, "seed": 42, "amplitude_mode": false, "hash": "..."}
```

Each step record includes:

```json
{
  "step": 0,
  "layers": [{"R": 0.3412, "psi": 2.718}, {"R": 0.2891, "psi": 1.043}],
  "stability": 0.3151,
  "regime": "NOMINAL",
  "actions": [],
  "phases": [1.23, 4.56, 0.78, 3.14],
  "hash": "sha256:abc123...",
  "prev_hash": "sha256:def456..."
}
```

Every record is SHA256-chained to the previous one. Tampering with any record breaks the chain.

## 7. Replay and Verify

```bash
spo replay my_audit.jsonl --verify
```

Expected output:

```
Steps logged: 500
Events logged: 0
Final regime: NOMINAL
Final stability: 0.9876
Determinism verified: 500 transitions OK
```

The replay engine re-executes every step from the header seed and verifies each step matches the log within tolerance (atol=1e-6). The SHA256 hash chain independently validates log integrity.

## 8. Generate a Report

```bash
spo report my_audit.jsonl
```

Outputs per-layer R statistics, regime distribution, action counts, and hash chain integrity status.

For JSON output:

```bash
spo report my_audit.jsonl --json-out
```

## 9. Visualize

Requires the `plot` extra: `pip install "scpn-phase-orchestrator[plot]"`

```python
import json
from scpn_phase_orchestrator.reporting import CoherencePlot

data = [json.loads(line) for line in open("my_audit.jsonl")]
plot = CoherencePlot(data)
plot.plot_r_timeline()           # R(t) per layer
plot.plot_regime_timeline()      # regime bands over time
plot.plot_cross_layer_heatmap()  # PLV cross-layer matrix
```

Each method returns a `matplotlib.figure.Figure`. Call `fig.savefig("output.png")` or `plt.show()` as needed.

## 10. Run Programmatically

Skip the CLI entirely:

```python
import numpy as np
from scpn_phase_orchestrator import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

engine = UPDEEngine(n_oscillators=4, dt=0.01, method="rk4")

rng = np.random.default_rng(42)
theta = rng.uniform(0, 2 * np.pi, 4)
omega = np.array([1.0, 1.1, 1.0, 1.1])
knm = np.full((4, 4), 0.45)
np.fill_diagonal(knm, 0.0)
alpha = np.zeros((4, 4))

theta = engine.run(theta, omega, knm, 0.0, 0.0, alpha, n_steps=500)

R, psi = compute_order_parameter(theta)
print(f"R={R:.4f}, psi={psi:.4f}")
```

## Next Steps

- [Tutorial 01: New Domain Checklist](../tutorials/01_new_domain_checklist.md) -- systematic guide for mapping any domain onto the SPO pipeline
- [Tutorial 02: Oscillator Hunt Sheet](../tutorials/02_oscillator_hunt_sheet.md) -- how to find the right signals to extract phases from
- [Concepts: Control Knobs](../concepts/knobs_K_alpha_zeta_Psi.md) -- K, alpha, zeta, Psi equations and supervisor use
- [API Reference](../reference/api/index.md) -- full Python API documentation
