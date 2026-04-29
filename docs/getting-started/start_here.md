# Start Here — Learning Path

Welcome to SCPN Phase Orchestrator. This page maps your background to
the right entry point.

The system is a domain-agnostic engine for coupled oscillator dynamics.
It does not matter whether your oscillators are neurons, generators,
servers, or financial instruments — the same equations govern
synchronisation in all of them. What differs is how you extract phases
from your signals and how you configure the coupling topology.

---

## I'm an ML Researcher

You want differentiable oscillator layers for neural networks.

1. **Read:** [Kuramoto Theory](../concepts/kuramoto_theory.md) — 5 min
   overview of the Kuramoto model, order parameter, and synchronisation
   transition.
2. **Do:** [Tutorial: Differentiable Kuramoto](../tutorials/04_differentiable_kuramoto.md)
   — build a `KuramotoLayer`, train coupling topology with backprop.
3. **Explore:** [KuramotoLayer API](../reference/api/nn.md) — all JAX
   layer classes, functional API, loss functions.
4. **Try:** Stuart-Landau layer (phase + amplitude dynamics),
   UDE-Kuramoto (physics-informed neural residual), simplicial layer
   (3-body higher-order interactions).

**Key concepts for you:**
- `nn.functional` — pure JAX functions (`kuramoto_step`, `order_parameter`,
  `plv`, `saf_loss`), all JIT-compilable and differentiable.
- `nn.kuramoto_layer` — `KuramotoLayer` wraps the functional API into
  a trainable module with learnable coupling matrix.
- `nn.training` — training utilities: curriculum learning, spectral
  alignment loss, coupling budget regularisation.
- Gradients flow through `order_parameter` and `saf_loss` for topology
  optimisation (which edges should exist and how strong).

**Install:** `pip install scpn-phase-orchestrator[nn]`

---

## I'm a Neuroscientist

You want to simulate brain oscillations, fit models to EEG/fMRI, detect
synchronisation regimes, or build a brain-computer interface.

1. **Read:** [System Overview](../concepts/system_overview.md) — full
   pipeline from signals to control actions.
2. **Do:** Run `notebooks/02_minimal_domain.ipynb` locally — simulates
   8 coupled oscillators with EEG-like dynamics.
3. **Explore:** `neuroscience_eeg` domainpack — binding spec for 10-20
   EEG channels with alpha/beta/gamma band extractors.
4. **Try:** BOLD generator (fMRI from oscillator amplitudes), PAC
   analysis (phase-amplitude coupling), chimera detection (partial
   synchronisation patterns), sleep staging (ultradian phase tracking).

**Key concepts for you:**
- P-channel extraction: Hilbert transform on bandpass-filtered EEG.
- Quality gating: SNR-based quality scores reject noisy channels.
- Coherence monitoring: R per frequency band, cross-band PLV.
- Psychedelic model: entropy-based coupling reduction (Carhart-Harris
  entropic brain hypothesis).
- BOLD generation: `nn.bold` module maps oscillator dynamics to
  hemodynamic response function convolution.

**Install:** `pip install scpn-phase-orchestrator[full]`

---

## I'm a Power Systems Engineer

You want to model grid stability, test control strategies, predict
cascading failures, or simulate generator dynamics.

1. **Read:** [Kuramoto Theory](../concepts/kuramoto_theory.md) — focus
   on the inertial (second-order) sections, which model generator
   rotor dynamics.
2. **Do:** [Advanced Dynamics Guide — Power Grids](../guide/advanced_dynamics.md#second-order-inertial-kuramoto-power-grids)
   — simulates a 5-bus system with generator trip events.
3. **Explore:** `power_grid` domainpack + `InertialKuramotoEngine`.
4. **Try:** Generator trip scenarios, weak coupling desynchronisation
   tests, frequency nadir prediction.

**Key concepts for you:**
- `InertialKuramotoEngine`: second-order model with inertia constant H
  and damping coefficient D. The swing equation is a special case.
- K_nm from admittance matrix: coupling strength proportional to line
  susceptance.
- Boundary observer: frequency deviation limits (49.5-50.5 Hz hard),
  voltage angle limits (soft).
- Regime manager: NOMINAL (stable), DEGRADED (frequency drifting),
  CRITICAL (cascading trip risk).

**Install:** `pip install scpn-phase-orchestrator`

---

## I'm a Quantitative Analyst

You want to detect market regimes via synchronisation of asset returns
or sector correlations.

1. **Read:** [Advanced Dynamics — Financial Markets](../guide/advanced_dynamics.md#financial-market-synchronization)
2. **Explore:** `financial_markets` domainpack.
3. **Try:** Hilbert phase extraction on detrended log-returns, order
   parameter R(t) as a regime indicator, PLV matrix for sector
   correlation structure.

**Key concepts for you:**
- `upde.market` module: `extract_phase()` from price series,
  `market_order_parameter()`, `detect_regimes()` for regime-switching.
- R spikes precede volatility events (empirical observation, not
  guaranteed — use as a signal, not a predictor).
- Boundary observer: VIX threshold (hard), correlation breakdown (soft).
- `sync_warning()`: early warning when R exceeds historical norms.

**Install:** `pip install scpn-phase-orchestrator`

---

## I'm a Roboticist

You want to coordinate swarm formation via phase coupling, or
synchronise multi-agent systems.

1. **Read:** [Advanced Dynamics — Swarmalators](../guide/advanced_dynamics.md#swarmalator-dynamics)
2. **Explore:** `SwarmalatorEngine` — coupled position + phase dynamics.
3. **Try:** Different J/K parameter regimes (static async, static sync,
   active phase wave), 3D formations.

**Key concepts for you:**
- Swarmalator model: spatial attraction/repulsion (J) coupled to phase
  dynamics (K). Spatial and phase order parameters both matter.
- Position-dependent coupling: K_ij decays with physical distance.
- Formation control: target Psi encodes desired formation geometry.

**Install:** `pip install scpn-phase-orchestrator`

---

## I'm a Physicist / Mathematician

You want the full mathematical framework, all engine variants, and
advanced analysis tools.

1. **Read:** [Kuramoto Theory](../concepts/kuramoto_theory.md) — full
   derivations including mean-field, Ott-Antonsen reduction, and
   bifurcation analysis.
2. **Explore:** all 12 engine variants — simplicial (3-body), Hodge
   decomposition, stochastic resonance, Ott-Antonsen reduction,
   geometric integrator, torus topology.
3. **Try:** [Advanced Dynamics Guide](../guide/advanced_dynamics.md) —
   all 9 engines with worked examples.
4. **Deep:** FEP-Kuramoto correspondence (Friston free energy applied
   to oscillator control), spectral alignment function (SAF for
   topology optimisation), basin stability analysis.

**Key concepts for you:**
- `upde.bifurcation`: `trace_sync_transition()` sweeps coupling
  strength, `find_critical_coupling()` locates K_c.
- `upde.reduction`: Ott-Antonsen mean-field reduction for large N.
- `upde.basin_stability`: Monte Carlo basin stability analysis.
- `coupling.hodge`: Hodge decomposition of coupling matrix into
  gradient, curl, and harmonic components.
- `coupling.spectral`: graph Laplacian, Fiedler value/vector,
  spectral gap, sync convergence rate.
- `monitor.lyapunov`: Lyapunov spectrum for chaos characterisation.
- `monitor.dimension`: correlation dimension, Kaplan-Yorke dimension.
- `monitor.recurrence`: recurrence plots and RQA.

**Install:** `pip install scpn-phase-orchestrator[full]`

---

## I'm a DevOps / Platform Engineer

You want to monitor microservice synchronisation, detect retry storms,
or orchestrate distributed systems.

1. **Read:** [System Overview](../concepts/system_overview.md)
2. **Explore:** `queuewaves` domainpack — models service queues as
   oscillators, request arrivals as I-channel events.
3. **Try:** Deploy with Docker Compose, connect Prometheus adapter,
   visualise R(t) on Grafana.

**Key concepts for you:**
- I-channel extraction: request arrival timestamps → phase.
- S-channel extraction: service state (healthy/degraded/down) → phase.
- R_bad objective: suppress retry storm synchronisation.
- Boundary observer: queue depth (hard), p99 latency (soft).
- gRPC server: `spo serve --grpc` for integration with existing
  infrastructure.

**Install:** `pip install scpn-phase-orchestrator[queuewaves]`

---

## Common Next Steps

After your entry point:

- **Concepts:** [Control Knobs K/alpha/zeta/Psi](../concepts/knobs_K_alpha_zeta_Psi.md)
  — the four parameters you can adjust.
- **Concepts:** [Pipeline Execution](../concepts/pipeline_execution.md)
  — how binding YAML resolves into extractors, engines, supervisor
  actions, and audit records.
- **Concepts:** [Phase Contract](../specs/phase_contract.md) — what
  every oscillator must produce.
- **Concepts:** [Oscillators P/I/S](../concepts/oscillators_PIS.md) —
  three extraction channels.
- **Control:** [Control Systems Guide](../guide/control_systems.md) —
  MPC, regime manager, Petri net sequencing.
- **Analysis:** [Analysis Toolkit Guide](../guide/analysis_toolkit.md)
  — 19 monitors (coherence, Lyapunov, chimera, PAC, transfer entropy,
  winding numbers, recurrence, EVS, embedding, dimension, entropy
  production, PID, STL, NPE, Poincare, sleep staging, session start,
  psychedelic, winding).
- **Calibration:** [Knm Calibration](../specs/knm_calibration.md) —
  how to set coupling strengths.
- **Deployment:** [Hardware Guide](../guide/hardware_deployment.md) —
  Rust FFI, FPGA, WASM, GPU, Docker.
- **Domains:** [Domainpack Gallery](../galleries/domainpack_gallery.md)
  — 24 domains.
- **API:** [Full API Reference](../reference/api/index.md)

---

## Quick Start (5 minutes)

```bash
pip install scpn-phase-orchestrator
```

```python
from scpn_phase_orchestrator import UPDEEngine, CouplingBuilder
import numpy as np

# 8 oscillators with random natural frequencies
n = 8
omegas = np.random.uniform(0.8, 1.2, n)
phases = np.random.uniform(0, 2 * np.pi, n)

# Build coupling matrix (uniform, strength 1.5)
builder = CouplingBuilder(n=n)
knm = builder.build_uniform(strength=1.5)

# Integrate 1000 steps
engine = UPDEEngine(n=n)
for _ in range(1000):
    engine.step(phases, omegas, knm, zeta=0.0, psi=0.0)

# Check synchronisation
from scpn_phase_orchestrator.upde import compute_order_parameter
R, psi = compute_order_parameter(phases)
print(f"Order parameter R = {R:.3f}")
# R > 0.8 means the oscillators synchronised
```

---

## Architecture at a Glance

```
scpn_phase_orchestrator/
    oscillators/     P/I/S phase extractors
    coupling/        K_nm construction, spectral analysis, plasticity
    upde/            12 integration engines
    monitor/         19 analysis monitors
    supervisor/      Regime management, policy engine, Petri net
    actuation/       Control action mapping, constraints, HDL compiler
    imprint/         Memory model (exposure accumulation)
    nn/              JAX differentiable layers (Kuramoto, SL, simplicial)
    adapters/        Bridges to external systems (Prometheus, Redis, LSL, ...)
    ssgf/            Stochastic Synthesis of Geometric Fields
    autotune/        Automated calibration pipeline
    visualization/   JSON export for web visualisation
    binding/         Binding spec loader and validator
    audit/           JSONL logger and deterministic replay
    reporting/       Coherence plots
    drivers/         P/I/S driver wrappers
    apps/            Domain applications (queuewaves)
```

## FAQ

**Q: How many oscillators can SPO handle?**
A: The pure Python path step takes ~0.1ms for N=64 (measured 2026-04-04),
which fits within a 256 Hz sample budget (3.9ms). Rust FFI and JAX GPU
scaling have not been measured on the current host. The
SparseUPDEEngine is recommended for N>100 with sparse coupling
topology to avoid O(N^2) dense matrix overhead.

**Q: Do I need to understand Kuramoto theory to use SPO?**
A: No. If you just want to detect synchronisation regimes, you can use
the auto-tune pipeline to configure everything from data. Understanding
the theory helps for advanced tuning and custom engine selection.

**Q: Can I use SPO with my existing monitoring stack?**
A: Yes. The adapters subpackage includes bridges for Prometheus,
OpenTelemetry, Redis, and gRPC. The MetricsExporter pushes R(t) and
regime status to any Prometheus-compatible endpoint.

**Q: Is there GPU support?**
A: Yes, via JAX. Install with `pip install scpn-phase-orchestrator[nn]`
and use `JaxUPDEEngine` or the `nn.functional` API. Requires JAX with
CUDA or ROCm backend.

**Q: Can I run SPO on a Raspberry Pi / embedded?**
A: The Rust library (`spo-engine`) compiles for ARM (not yet tested
on RPi). The FPGA path (`spo-fpga`) generates Verilog for Xilinx
PYNQ-Z2 — latency not yet measured on hardware.

## References

- [System Overview](../concepts/system_overview.md) — full pipeline diagram.
- [Kuramoto Theory](../concepts/kuramoto_theory.md) — mathematical foundations.
- [Installation Guide](installation.md) — detailed install instructions.
- [CHANGELOG](https://github.com/anulum/scpn-phase-orchestrator/blob/main/CHANGELOG.md) — version history and migration notes.
