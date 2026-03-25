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

### What is the nn/ module?

A differentiable Kuramoto backend built on JAX and equinox. It exposes
oscillator dynamics as learnable neural network layers: `KuramotoLayer`,
`StuartLandauLayer`, simplicial 3-body coupling, BOLD hemodynamic signal,
reservoir computing, UDE-Kuramoto (physics + neural residual), inverse
coupling inference, and an oscillator Ising machine (OIM) for combinatorial
optimization. All functions are `jax.jit`-compilable and `jax.vmap`-compatible.

Install: `pip install scpn-phase-orchestrator[nn]`

See the [Differentiable Kuramoto guide](guide/differentiable_kuramoto.md).

### What are the 9 UPDE engines?

SPO ships 9 ODE engine variants beyond the standard Kuramoto:

1. **Standard Kuramoto** — first-order phase coupling
2. **Stuart-Landau** — phase + amplitude with Hopf bifurcation
3. **Inertial** — second-order swing equation for power grids
4. **Market** — financial regime detection via Hilbert phase
5. **Swarmalator** — coupled spatial position + phase (robotics, biology)
6. **Stochastic** — Euler-Maruyama with optimal noise D*
7. **Geometric** — torus-preserving symplectic integrator for long simulations
8. **Delay** — time-delayed coupling with circular buffer
9. **Simplicial** — 3-body higher-order interactions (Gambuzza 2023)

Plus Ott-Antonsen mean-field reduction for O(1) MPC prediction.

See the [Advanced Dynamics guide](guide/advanced_dynamics.md).

### Can SPO detect market crashes?

The `upde.market` module extracts instantaneous phase from price/return
time series via Hilbert transform, computes the Kuramoto order parameter
R(t) across assets, and classifies synchronization regimes. R(t) → 1
preceding crashes is documented for Black Monday 1987 and the 2008
financial crisis (arXiv:1109.1167). The `sync_warning()` function flags
when R crosses a threshold from below.

### What is the SSGF?

The Self-Stabilizing Gauge Field is a free energy framework that maps
Kuramoto dynamics to Friston's Free Energy Principle. The `ssgf/` module
implements the carrier field, Langevin noise injection, Boltzmann weighting,
free energy closure, and the TCBO (Topological Consciousness Boundary
Observer) and PGBO (Phase Gradient Boundary Observer) constructs from the
SCPN consciousness model.

### Can SPO solve combinatorial optimization problems?

Yes. The `nn.oim` module implements an Oscillator Ising Machine that maps
graph coloring, max-cut, and QUBO problems to Kuramoto phase clustering.
Oscillators connected by graph edges repel from the same phase cluster.
The dynamics settle into valid colorings. Differentiable via JAX for
gradient-based energy minimization.

### What is inverse Kuramoto?

Given observed phase trajectories (from EEG, sensors, market data), the
`nn.inverse` module infers the coupling matrix K and natural frequencies
ω by backpropagating through the Kuramoto ODE solver. L1 sparsity penalty
discovers network topology (which oscillators are actually coupled).

### How does stochastic resonance work?

Counter-intuitively, adding noise at the optimal level D* = K·R_det/2
*increases* synchronization. The `upde.stochastic` engine implements
Euler-Maruyama integration with automatic D* tuning. The effect is
explained by the modified Bessel equation in the self-consistency
condition (Acebrón et al. 2005).

### What is the Ott-Antonsen reduction?

An exact analytical reduction of the N-oscillator Kuramoto system to a
single complex ODE: dz/dt = -(Δ + iω₀)z + (K/2)(z - |z|²z). Valid for
globally-coupled oscillators with Lorentzian frequency distribution.
Used by the `PredictiveSupervisor` as a fast forward model for MPC
(O(1) computation vs O(N) for full simulation).

### How do I report a security vulnerability?

Follow the responsible disclosure process in
[SECURITY.md](https://github.com/anulum/scpn-phase-orchestrator/blob/main/SECURITY.md).
Do not open a public issue.

### How do I cite this project?

Use the metadata in
[CITATION.cff](https://github.com/anulum/scpn-phase-orchestrator/blob/main/CITATION.cff),
which is machine-readable by Zenodo, GitHub, and most reference managers.
