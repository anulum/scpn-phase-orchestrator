# SCPN Phase Orchestrator: Capability Boundaries and Use Cases

For the detailed, per-subsystem architecture (inputs/outputs, processing models,
backend wiring, interface contracts, and honest scope boundaries) see
[`docs/architecture/`](docs/architecture/). This document summarises capabilities
and representative use cases. It is not a performance forecast: latency,
throughput, scale, and deployment suitability must be measured on the target
hardware and active backend before being used as claims.

## 1. Core phase-dynamics engine

SPO is a topology manager and phase-dynamics solver for hierarchical oscillator
systems. The same simulation core can evaluate different reviewed domainpacks
when the declared oscillator families, coupling assumptions, and sampling
contracts match the source data. The base dynamics are the Universal Phase
Dynamics Equation (a Kuramoto/Sakaguchi family with extensions):

$$ \frac{d\theta_i}{dt} = \omega_i + \frac{K}{N} \sum_{j=1}^{N} A_{ij} \sin(\theta_j - \theta_i) + \zeta $$

### Numerics and acceleration

- Integration methods: `euler`, `rk4`, and adaptive `rk45` (Dormand–Prince).
- A Rust kernel (`spo-kernel`, PyO3) provides accelerated paths, with a pure-Python
  reference floor. Backend selection is per-kernel and fastest-first
  (Rust → WebGPU → Mojo → Julia → Go → Python); the kernel is optional and is not
  importable in every environment, in which case the Python floor runs. See
  [`docs/architecture/backends.md`](docs/architecture/backends.md).
- Benchmarks live under `bench/` and `benchmarks/` (Python) and
  `spo-kernel/crates/spo-engine/benches/` (Rust Criterion). Measured figures are
  environment-specific and are not quoted here — run the harness on the target
  hardware and active backend before relying on a number.

## 2. The domainpack abstraction

SPO separates *topology* (hierarchy and coupling structure, declared per domain)
from *physics* (the integrator and observers). A `binding_spec.yaml` shapes the
$K_{nm}$ coupling matrix and the per-layer frequencies. 36 domainpacks ship under
`domainpacks/`. Examples:

- **`plasma_control`** — an 8-layer hierarchy with exponential distance decay,
  mapping micro-turbulence down to wall-equilibrium timescales.
- **`bio_stub`** — 4 macro-physiological layers (cellular → systemic) with
  Sakaguchi phase-lags and biological-clock frequencies.

**Ecosystem bridges** (`adapters/`) translate raw phase state ($R$, $\Psi$) into
domain-specific telemetry — for example a coupling-matrix handshake for
`scpn-control`, or an OpenQASM compiler manifest for `scpn-quantum-control`.
These bridges are opt-in and emit wire formats; they do not hard-depend on the
sibling repositories (only `quantum_control_bridge` lazily imports its sibling,
and only when a quantum method is called).

## 3. Representative use cases and boundaries

- **Transition / disruption early warning.** Tracking the cross-layer alignment
  matrix, the global order parameter $R$, the conformal twin-confidence gate, and
  the ordinal-pattern transition-entropy monitor can flag systemic phase
  transitions (e.g. a tokamak disruption or a seizure-onset analogue) ahead of a
  macroscopic change. SPO's posture is review-only: it emits audited proposals,
  not actuation.
- **Decentralised swarm synchronisation.** A phase-consensus substrate for how
  many independent agents (drones, robots) reach collective agreement from local
  Kuramoto coupling, without a central command server.
- **Oscillator Ising / combinatorial optimisation.** The differentiable `nn/oim`
  oscillator Ising machine maps graph colouring, max-cut, and QUBO to phase
  clustering (a research track, see `docs/architecture/subsystems/nn.md`).

Illustrative-only domainpacks (e.g. `financial_markets`, social-dynamics) are
modelling exercises; phase *control* has no actuator in those domains, so they
are not positioned as control use cases.

SPO does not claim global real-time forecasting, hardware actuation authority,
clinical decision support, or production safety certification from these use
cases alone. Promotion from a review-only proposal to an operational workflow
requires domain validation, replay evidence, signed audit records, and an
external safety process appropriate to the deployment.

## 4. Scaling

The Rust kernel uses data-parallel iterators (`rayon`), and selected kernels have
polyglot accelerator paths. The dense $K_{nm}$ matrix is $O(N^2)$ in memory, so
large-$N$ deployments depend on sparse coupling (`SparseUPDEEngine`) or
mean-field reduction (Ott–Antonsen). Concrete cluster-scale throughput has not
been measured at the time of writing; any large-$N$ figure should be established
on the target hardware rather than projected.
