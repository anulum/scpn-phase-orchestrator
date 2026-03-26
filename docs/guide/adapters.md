# Adapter Bridges

Adapters translate SPO phase state into domain-specific telemetry and back.
Each adapter works without its target package installed -- methods accept and
return plain dicts and numpy arrays. Domain-specific features (circuit
construction, SNN process creation) require the target package.

## Available Adapters

### FusionCoreBridge

Bidirectional coupling with [scpn-fusion-core](https://github.com/anulum/scpn-fusion-core).
Maps 6 Grad-Shafranov equilibrium observables (q-profile, beta_n, tau_e,
sawtooth count, ELM count, MHD amplitude) to oscillator phases and back.

```python
from scpn_phase_orchestrator.adapters import FusionCoreBridge

bridge = FusionCoreBridge(n_layers=6)
phases = bridge.observables_to_phases({"q_profile": 2.1, "beta_n": 1.5, "tau_e": 2.0})
feedback = bridge.phases_to_feedback(phases, omegas)
violations = bridge.check_stability({"q_min": 0.8, "beta_n": 3.0})
```

### PlasmaControlBridge

Interface to [scpn-control](https://github.com/anulum/scpn-control) plasma
telemetry. Imports Knm specs via Kronecker expansion, converts tick results
to `UPDEState`, exports control actions, and checks physics invariants
(Kruskal-Shafranov, Troyon, Greenwald limits).

```python
from scpn_phase_orchestrator.adapters import PlasmaControlBridge

bridge = PlasmaControlBridge(n_layers=8)
coupling = bridge.import_knm_spec({"matrix": layer_knm, "n_osc_per_layer": 2})
state = bridge.import_snapshot({"phases": phases_array, "regime": "NOMINAL"})
omegas = bridge.import_plasma_omega(n_osc_per_layer=1)
```

### QuantumControlBridge

Interface to [scpn-quantum-control](https://github.com/anulum/scpn-quantum-control).
Imports quantum calibration results (phases, fidelity, layer assignments) into
`UPDEState`. Exports state back. Optionally builds Trotterised XY-Hamiltonian
circuits (requires scpn-quantum-control).

```python
from scpn_phase_orchestrator.adapters import QuantumControlBridge

bridge = QuantumControlBridge(n_oscillators=4, trotter_order=1)
state = bridge.import_artifact({"phases": [0.1, 0.5, 1.2, 2.3], "fidelity": 0.95})
export = bridge.export_artifact(state)
coupling = bridge.import_knm(knm_array)
```

### SCPNControlBridge

Generic bridge for the SCPN ecosystem. Imports Knm matrices and omega vectors,
exports `UPDEState` as a telemetry dict with lock signatures.

```python
from scpn_phase_orchestrator.adapters import SCPNControlBridge

bridge = SCPNControlBridge(scpn_config={"n_layers": 16})
coupling = bridge.import_knm(knm_16x16)
omega = bridge.import_omega(omega_array)
telemetry = bridge.export_state(upde_state)
```

### SNNControllerBridge

Spiking neural network controller with three backends: pure-numpy LIF (always
available), Nengo (optional), Lava (optional). Converts `UPDEState` R-values
to LIF input currents, spike rates to `ControlAction` objects.

```python
from scpn_phase_orchestrator.adapters import SNNControllerBridge

bridge = SNNControllerBridge(n_neurons=100, tau_rc=0.02, tau_ref=0.002)
currents = bridge.upde_state_to_input_current(state, i_scale=2.0)
rates = bridge.lif_rate_estimate(currents)
actions = bridge.spike_rates_to_actions(rates, layer_assignments=[0, 0, 1, 1])
network = bridge.build_numpy_network(n_layers=4, seed=0)
```

LIF rate estimate uses Abbott 1999, Eq. 1:
`rate = 1 / (tau_ref - tau_rc * ln(1 - 1/J))` for J > 1.

### NeurocoreBridge

Live integration with [sc-neurocore](https://github.com/anulum/sc-neurocore)
`StochasticLIFNeuron` ensembles. Maps UPDE layer coherence R to neuron input
currents, runs a stochastic LIF ensemble, converts spike rates to coupling
boost `ControlAction` objects.

Three backends, selected automatically (best available):

| Backend | Implementation | N=10000 × 100 substeps | Speedup |
|---------|---------------|------------------------|---------|
| **Rust** (spo_kernel) | `spo-engine::lif_ensemble` via PyO3 | 0.004 s | 325× |
| **NumPy** | Vectorised Euler-Maruyama | 0.014 s | 93× |
| **Scalar** | Per-neuron sc-neurocore objects | 1.306 s | 1× |

LIF dynamics match sc-neurocore v3.13.3 defaults exactly (Gerstner & Kistler
2002: v_rest=0, v_threshold=1, tau_mem=20ms, R=1, dt=1ms, no noise).

```python
from scpn_phase_orchestrator.adapters import NeurocoreBridge

# Auto-selects Rust if spo_kernel installed, else numpy
bridge = NeurocoreBridge(n_layers=10, neurons_per_layer=1000, current_scale=2.5)
print(bridge.backend)  # "rust" or "numpy"

# Step the ensemble — returns per-layer firing rates (Hz)
rates = bridge.step(upde_state, n_substeps=100)

# Convert rates above threshold to coupling boost actions
actions = bridge.rates_to_actions(rates)

# Or do both in one call
actions = bridge.step_and_act(upde_state, n_substeps=100)

# Force a specific backend
bridge_np = NeurocoreBridge(n_layers=10, neurons_per_layer=1000, backend="numpy")
bridge_sc = NeurocoreBridge(n_layers=10, neurons_per_layer=100, backend="scalar")
```

At N=10000 the Rust backend completes 100 substeps in 4ms, enabling real-time
spiking control loops at ~250 Hz update rate.

### OTelExporter

OpenTelemetry trace and metric export. Records `spo.r_global` and
`spo.stability_proxy` gauges, `spo.steps_total` counter, and creates spans
for UPDE steps and regime transitions. Falls back to no-op when
`opentelemetry-api` is not installed.

```python
from scpn_phase_orchestrator.adapters import OTelExporter

exporter = OTelExporter(service_name="spo")
with exporter.span("upde_step", {"spo.n": 64}):
    phases = engine.step(...)
exporter.record_step(upde_state, step_idx=42)
exporter.record_regime_change("nominal", "degraded")
```

### PrometheusAdapter

Fetches time-series metrics from a Prometheus endpoint. Currently a stub
(planned for full implementation).

## Optional Dependencies

Install extras to pull in adapter-specific packages:

```bash
pip install scpn-phase-orchestrator[fusion]    # scpn-fusion-core
pip install scpn-phase-orchestrator[plasma]    # scpn-control
pip install scpn-phase-orchestrator[quantum]   # scpn-quantum-control
pip install scpn-phase-orchestrator[otel]      # opentelemetry-api + opentelemetry-sdk
```

All adapters work without their extras installed -- they degrade to dict-based
operation and raise `ImportError` only when domain-specific methods are called
(circuit construction, Lava process creation).

## Writing a Custom Adapter

No base class required. An adapter is any class that translates between
domain data and SPO types (`UPDEState`, `CouplingState`, `ControlAction`).
Follow this pattern:

```python
from scpn_phase_orchestrator.upde.metrics import UPDEState
from scpn_phase_orchestrator.coupling.knm import CouplingState

class MyDomainBridge:
    def __init__(self, domain_config: dict):
        self._config = domain_config

    def import_state(self, domain_data: dict) -> UPDEState:
        """Domain telemetry -> SPO state."""
        ...

    def export_actions(self, actions: list) -> dict:
        """SPO control actions -> domain commands."""
        ...
```

Place the module in `src/scpn_phase_orchestrator/adapters/` and add it to
`adapters/__init__.py`.
