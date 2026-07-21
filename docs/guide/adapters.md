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
The bridge is a review and feedback boundary: it rejects non-positive safety
factor bounds, negative beta/confinement/MHD observables, empty phase feedback
vectors, and non-finite values before any equilibrium state is translated into
SPO phase feedback.

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
The bridge rejects boolean numeric aliases, non-zero layer self-coupling,
empty phase snapshots, negative beta/Greenwald ratios, and non-positive
safety-factor minima before any plasma payload is expanded into `K_nm` or
exported as a phase-state review record.

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
compiler_manifest = bridge.build_quantum_compiler_manifest(
    knm_array,
    omega_array,
    dt=0.01,
)
edge_import = bridge.import_scpn_upde_edge(quantum_edge_payload)
```

`build_quantum_compiler_manifest()` emits dependency-free OpenQASM 3 review
text for Qiskit and PennyLane handoff. It records Z-frequency terms,
symmetrised XY coupling terms, co-simulation parity evidence from deterministic
term reconstruction, and SHA-256 hashes for the QASM and manifest payloads. It
also runs the emitted text through `check_openqasm3()` (a dependency-free
structural conformance checker), embedding the result under the manifest's
`openqasm_conformance` key and a `qasm_parse_ok` flag in the parity evidence.
The checker distinguishes gates that `stdgates.inc` defines from the two-qubit
Pauli-rotation extensions (`rxx`/`ryy`) that Qiskit and PennyLane provide as
builtins, so the manifest reports its use of backend extensions honestly rather
than overclaiming pure-standard conformance. The manifest keeps QPU execution
and live actuation disabled until an operator runs external simulator parity and
target handoff checks.

`import_scpn_upde_edge()` accepts the QUANTUM `knm.scpn-upde.v1` payload only
when its `scope_envelope` is `computational-agreement`, its K/omega and edge
SHA-256 digests recompute, and both QPU execution and actuation permissions are
false. It imports the K_nm matrix into `CouplingState` and rebuilds SPO's own
compiler manifest; it does not promote Paper-27 couplings into canonical
physical evidence.

Use `audit_qpu_target_readiness()` to produce a non-executing target-readiness
record for a reviewed manifest:

```python
readiness = bridge.audit_qpu_target_readiness(
    compiler_manifest,
    target_backend="qiskit_openqasm3",
    provider="ibm_quantum",
)
```

The readiness record validates the requested backend against the manifest,
records missing credentials and operator approval as blockers, and keeps
`qpu_execution_permitted=false` plus `actuation_permitted=false` even when all
preconditions are present. It is release/operator evidence, not a QPU submitter.

### SCPNControlBridge

Generic bridge for the SCPN ecosystem. Imports Knm matrices and omega vectors,
exports `UPDEState` as a telemetry dict with lock signatures. Imported
coupling matrices must be non-empty, finite, real-valued, square, and
zero-self-coupled; imported natural-frequency vectors must be non-empty,
finite, real-valued, one-dimensional, and strictly positive.

```python
from scpn_phase_orchestrator.adapters import SCPNControlBridge

bridge = SCPNControlBridge(scpn_config={"n_layers": 16})
coupling = bridge.import_knm(knm_16x16)
omega = bridge.import_omega(omega_array)
telemetry = bridge.export_state(upde_state)
```

### SNNControllerBridge

Spiking neural network controller with three backends: pure-numpy LIF (always
available), Nengo-style pure-numpy schedules, and Lava schedule handoff.
`lava-nc` is intentionally not installed by SPO extras because current
upstream releases pin a vulnerable `asteval` range; live Lava runtimes must be
managed by the external operator environment. Converts `UPDEState` R-values to
LIF input currents and spike rates to `ControlAction` objects. Imported layer
order parameters must be finite magnitudes in `[0, 1]`; current, rate, and
cross-alignment arrays reject boolean and complex aliases before numeric
coercion, and spike rates must remain non-negative.

```python
from scpn_phase_orchestrator.adapters import SNNControllerBridge

bridge = SNNControllerBridge(n_neurons=100, tau_rc=0.02, tau_ref=0.002)
currents = bridge.upde_state_to_input_current(state, i_scale=2.0)
rates = bridge.lif_rate_estimate(currents)
actions = bridge.spike_rates_to_actions(rates, layer_assignments=[0, 0, 1, 1])
network = bridge.build_numpy_network(n_layers=4, seed=0)
schedule = bridge.build_neuromorphic_schedule_manifest(state, i_scale=2.0)
```

LIF rate estimate uses Abbott 1999, Eq. 1:
`rate = 1 / (tau_ref - tau_rc * ln(1 - 1/J))` for J > 1.

`build_neuromorphic_schedule_manifest()` emits a deterministic review artefact
for Lava and PyNN handoff. It records one population per UPDE layer, positive
cross-layer projections, control-action review records, simulator parity
evidence from the numpy LIF rate path, and a SHA-256 schedule hash. It also emits
a portable NIR-structural graph of the populations and projections via
`to_nir_graph()` under the manifest's `neuromorphic_ir` key (with a `nir_sha256`
digest) — an honestly-labelled structural subset of `neuromorphs/NIR` that
carries only the LIF parameters the Abbott-rate model defines and lists the
unmodelled NIR physical parameters rather than fabricating them. The manifest
keeps `actuation_permitted` and `hardware_write_permitted` false; it is a
simulator-parity handoff, not a live neuromorphic target run.

### SynchrophasorFrameCodec

`SynchrophasorFrameCodec` decodes IEEE C37.118.2-2011 synchrophasor CONFIG-2 and
DATA frames from raw bytes with no network dependency. A CONFIG-2 frame yields
each PMU's measurement layout, and a DATA frame is then decoded against that
layout into phasor, frequency, and analog/digital measurements. FREQ is reported
as a deviation from the PMU nominal frequency (millihertz for the integer FORMAT,
hertz for the floating-point FORMAT) and surfaced as an absolute value in hertz.
Every frame is CRC-CCITT validated before its body is read; a truncated frame, a
wrong SYNC word, a framesize mismatch, or a checksum mismatch raises a typed
`SynchrophasorFrameError` subclass rather than returning partial data. The byte
layout and CRC parameters were cross-checked against two independent open-source
implementations of the standard rather than a paywalled clause reference.
`data_frames_to_frequency_series` emits a `(time_s, frequency_hz)` series in the
exact two-column layout the PMU ringdown screener consumes, so a decoded
synchrophasor stream feeds the existing hash-sealed ringdown evidence path. Live
socket ingestion behind an optional `c37118` extra is a separate follow-up; this
codec deliberately handles only bytes already read.

### C37118PhaseBridge

`C37118PhaseBridge` maps decoded PMU phasors (from `SynchrophasorFrameCodec`) to
oscillator phase states. A PMU phasor already carries a magnitude and an angle,
so the bridge reads the phase directly rather than running a waveform extractor:
`theta` is the phasor angle (rectangular `atan2(imag, real)` — scale-independent
— or a floating-point polar angle in radians), `omega` is `2*pi` times the
frame's measured line frequency, and `amplitude` is the phasor magnitude in
engineering units (integer components scaled by the verified PHUNIT `10**-5` V/A
per-bit factor). `quality` derives only from the STAT word's data-error and
time-sync bits. Integer *polar* phasors are an explicit honest boundary: the
standard scales an integer polar angle differently from its magnitude and the
open-source references disagree, so the bridge raises rather than emit a
fabricated angle. Each binding names the PMU and phasor index and the target
oscillator; the bridge is review-only (`non_actuating` / `execution_disabled`)
and never actuates.

### C37118SessionClient

`C37118SessionClient` is a live asynchronous reader for a phasor data
concentrator or PMU over TCP, built on the standard library's `asyncio` with no
third-party dependency. It issues the standard C37.118.2 command frames to drive
the stream — request CONFIG-2, turn data transmission on, read the requested
number of DATA frames, then turn transmission off — and delegates decoding to
`SynchrophasorFrameCodec`. Those command frames are a benign protocol handshake
that controls only the measurement data stream; the client never writes device
setpoints and cannot actuate grid equipment (`non_actuating`). The command-word
values (`0x0001` data-off, `0x0002` data-on, `0x0005` send CONFIG-2, …) were
verified at source against the pypmu `CommandFrame` table and the Wireshark
synchrophasor dissector, which cites the standard's Table 15. `read_frame`
reassembles a single frame from the stream via its SYNC/FRAMESIZE prefix, so a
partial or malformed prefix fails closed with a typed error.

### Hardware I/O

The sample buffer and simulated hardware board provide deterministic
real-valued sensor ingress for local pipeline tests. Samples must be finite real
amplitudes, and custom frequencies must be finite positive real values; boolean
and complex aliases are rejected before buffering or waveform generation so
digital flags and phasors cannot enter oscillator sensor channels.

### GaianMeshNode

Distributed mesh coupling exchanges reduced peer order parameters. Peer and
local phases use circular semantics: finite real negative `psi` values are
wrapped modulo `2*pi`, while non-finite and boolean aliases remain invalid.

### LSLBCIBridge

Live BCI ingress uses Lab Streaming Layer samples as a phase-extraction input.
The buffer accepts only finite real EEG amplitudes with finite non-negative LSL
timestamps; boolean samples are rejected before Hilbert phase extraction so
binary flags cannot enter the oscillator phase channel as amplitudes.

### Hybrid Neuromorphic-Quantum Co-Compiler

The hybrid co-compiler combines a quantum compiler manifest and a neuromorphic
schedule manifest under one shared audit envelope.

```python
from scpn_phase_orchestrator.adapters import build_hybrid_cocompiler_manifest

hybrid = build_hybrid_cocompiler_manifest(
    compiler_manifest,
    schedule,
    n_channel_semantics=("Q_control", "S_spike", "audit"),
)
```

The combined manifest records target backends, component hashes,
co-simulation parity status, and N-channel semantics in a deterministic JSON
payload. It blocks if either component parity check fails or if a component
tries to enable execution. `qpu_execution_permitted`,
`hardware_write_permitted`, and `actuation_permitted` remain false.

### NeurocoreBridge

Live integration with [sc-neurocore](https://github.com/anulum/sc-neurocore)
`StochasticLIFNeuron` ensembles. Maps UPDE layer coherence R to neuron input
currents, runs a stochastic LIF ensemble, converts spike rates to coupling
boost `ControlAction` objects. Seeds must be non-negative when provided,
UPDE layer coherences must remain finite magnitudes in `[0, 1]`, and
backend/action rate vectors reject boolean aliases before float coercion.

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
from scpn_phase_orchestrator.runtime.observability import OTelExporter

exporter = OTelExporter(service_name="spo")
with exporter.span("upde_step", {"spo.n": 64}):
    phases = engine.step(...)
exporter.record_step(upde_state, step_idx=42)
exporter.record_regime_change("nominal", "degraded")
```

### PrometheusAdapter

Fetches range and instant metrics from a Prometheus endpoint with validated
HTTP(S) endpoint, timeout, query parameters, decoded response shape, sample
timestamps, and sample values. Decoded JSON rejects non-finite constants and
duplicate object keys before status, data, result, or sample fields can be
overwritten by last-value-wins parsing. Range and instant samples are accepted
only when their Prometheus timestamps are finite and non-negative and their
metric values are finite real numbers, so malformed telemetry cannot enter
downstream oscillator-control metrics.

```python
from scpn_phase_orchestrator.adapters import PrometheusAdapter

prom = PrometheusAdapter("https://prometheus.internal", timeout=5.0)
series = prom.fetch_metric("up", start=0.0, end=60.0, step=5.0)
instant = prom.fetch_instant("up")
```

## Production Defaults (Auth + Rate Limits)

- Modbus/TLS: use mutual-TLS certificates. `SecureModbusAdapter` always keeps
  server verification enabled; pass `ca_cert_path` for a deployment CA bundle,
  or omit it only when the operating-system trust store contains the server CA.
- Plain Modbus TCP (`ModbusAdapter`) is a local-lab or isolated-network adapter.
  Do not use it for production writes across routable networks.
- QueueWaves network endpoints: use `security.mode: production`,
  `api_key_env`, and positive `security.rate_limit_per_minute`.
- Prometheus access: terminate auth at a reverse proxy and inject
  short-lived bearer credentials upstream; never commit token-bearing URLs.

### Adapter Schema + Rate-Limit Patterns

For network/file-bound adapters, keep transport inputs explicit and bounded:

```python
from pathlib import Path
from scpn_phase_orchestrator.adapters import SecureModbusAdapter
from scpn_phase_orchestrator.adapters import RedisStateStore
from scpn_phase_orchestrator.runtime.network_security import FixedWindowRateLimiter, env_int

# Strictly parse transport schema before constructing adapters.
with SecureModbusAdapter(
    host="plc.internal.example",
    port=802,
    tls_cert_path=Path("/etc/scada/pki/client.pem"),
    tls_key_path=Path("/etc/scada/pki/client.key"),
    ca_cert_path=Path("/etc/scada/pki/ca.pem"),
) as modbus:
    if not modbus.validate_connection():
        raise RuntimeError("Modbus/TLS connection is not active")

store = RedisStateStore(
    host="redis.internal.example",
    port=6379,
    db=0,
    key="spo:sim_state",
)

rate_limit = env_int("SPO_ADAPTER_RATE_LIMIT_PER_MINUTE", 120)
limiter = FixedWindowRateLimiter(rate_limit_per_minute=rate_limit)
if not limiter.allow("modbus-client-01"):
    raise RuntimeError("rate limit exceeded for modbus writes")
```

Keep this pattern in deployment-specific code: validate config first, then
instantiate adapters with sanitised defaults.

## Optional Dependencies

Install extras to pull in adapter-specific packages:

```bash
pip install scpn-phase-orchestrator[plasma]    # scpn-control
pip install scpn-phase-orchestrator[quantum]   # scpn-quantum-control
pip install scpn-phase-orchestrator[otel]      # opentelemetry-api + opentelemetry-sdk
```

The `[fusion]` extra requires `scpn-fusion-core`, a separate product that is not
on public PyPI, so `pip install scpn-phase-orchestrator[fusion]` will not resolve
from a public index; see the *PyPI availability boundary* section of the
[Install Profiles](install_profiles.md) guide.

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
