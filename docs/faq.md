# Frequently Asked Questions

### What is the Kuramoto model?

A system of $N$ coupled phase oscillators:
$\dot{\theta}_i = \omega_i + \frac{K}{N} \sum_j \sin(\theta_j - \theta_i)$.
Above a critical coupling $K_c$, oscillators spontaneously synchronize.

### What does "domain-agnostic" mean?

The orchestrator separates topology from physics. A `binding_spec.yaml` declares
oscillator names, frequencies, coupling templates, and policy rules without
referencing any particular domain. The same solver handles plasma stability,
neural coherence, or microservice health.

### Do I need Rust?

No. Pure Python works out of the box. When `spo_kernel` is installed (compiled
via `maturin develop` in `spo-kernel/`), hot loops auto-delegate to Rust for
5--10x acceleration. The public API is identical in both backends.

### What Python versions are supported?

3.10, 3.11, 3.12, and 3.13. CI tests all four.

### How do I add a new domain?

Run `spo scaffold <name>` to generate starter files, then follow the
[New Domain Checklist](tutorials/01_new_domain_checklist.md) tutorial.

### What is R_good vs R_bad?

Dual-objective coherence control. `R_good` is the order parameter for
oscillators that should synchronize (e.g., EEG gamma band). `R_bad` is the
order parameter for oscillators that should remain desynchronized (e.g.,
pathological seizure coupling). The supervisor tries to maximize `R_good` while
minimizing `R_bad`.

### How does the Petri net relate to regimes?

The Petri net is a formal finite-state machine for multi-phase protocols. Each
Place corresponds to a regime (e.g., BASELINE, ENTRAINMENT, RECOVERY). Guard
expressions on transitions evaluate phase metrics (R_global, boundary
violations) to decide when regime changes fire. `PetriNetAdapter` wraps
`RegimeManager` with this FSM layer.

### What is Stuart-Landau mode?

The standard Kuramoto model tracks only phase. `StuartLandauEngine` extends
this to phase+amplitude coupling via the Stuart-Landau ODE:
$\dot{z}_i = (\mu_i + i\omega_i)z_i - |z_i|^2 z_i + K \sum_j A_{ij}(z_j - z_i)$.
Use it when amplitude dynamics matter (e.g., neural oscillation power,
oscillation death).

### How does deterministic replay work?

`AuditLogger` writes a SHA256-chained JSONL file. Each line hashes the previous
line's hash plus the current payload. `ReplayEngine` rebuilds the full engine
from the header record, re-runs every step, and compares state vectors
bit-for-bit. A single flipped bit breaks the chain.

### What is QueueWaves?

A cascade failure detector that maps microservice queue-depth and latency
metrics onto Kuramoto oscillators. When inter-service phase coherence drops
(measured via R_global), it signals an impending cascade before individual
service alerts trigger. See the
[QueueWaves guide](guide/queuewaves.md).

### How do I integrate with Prometheus?

Two options:

1. `OTelExporter` -- emits OpenTelemetry spans and metrics, which
   the OTel Collector can forward to Prometheus.
2. `PrometheusExporter` -- directly exposes a `/metrics` endpoint with
   `spo_r_global`, `spo_regime`, and per-layer gauges.

Both live in `scpn_phase_orchestrator.adapters`.

### What is PAC?

Phase-Amplitude Coupling, quantified by the Modulation Index (Tort et al.,
2010). `scpn_phase_orchestrator.upde.pac.modulation_index` computes MI between
a low-frequency phase signal and a high-frequency amplitude envelope. High MI
indicates cross-frequency coupling.

### What are the four control knobs?

| Knob | Symbol | Effect |
|------|--------|--------|
| Coupling strength | $K$ | How strongly oscillators pull each other |
| Phase lag | $\alpha$ | Sakaguchi lag shifts the coupling function |
| Driver amplitude | $\zeta$ | Strength of external forcing |
| Target phase | $\Psi$ | Desired phase offset for entrainment |

Policy rules adjust these knobs in response to regime transitions and boundary
violations.

### How do I report a security vulnerability?

Follow the responsible disclosure process in
[SECURITY.md](https://github.com/anulum/scpn-phase-orchestrator/blob/main/SECURITY.md).
Do not open a public issue.

### How do I cite this project?

Use the metadata in
[CITATION.cff](https://github.com/anulum/scpn-phase-orchestrator/blob/main/CITATION.cff),
which is machine-readable by Zenodo, GitHub, and most reference managers.
