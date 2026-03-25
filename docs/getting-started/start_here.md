# Start Here — Learning Path

Welcome to SCPN Phase Orchestrator. This page maps your background to
the right entry point.

## I'm an ML Researcher

You want differentiable oscillator layers for neural networks.

1. **Read:** [Kuramoto Theory](../concepts/kuramoto_theory.md) — 5 min overview
2. **Do:** [Tutorial: Differentiable Kuramoto](../tutorials/04_differentiable_kuramoto.md)
3. **Explore:** [KuramotoLayer API](../reference/api/nn.md)
4. **Try:** Stuart-Landau layer (phase + amplitude), UDE-Kuramoto (physics + neural residual)

**Install:** `pip install scpn-phase-orchestrator[nn]`

## I'm a Neuroscientist

You want to simulate brain oscillations, fit models to EEG/fMRI, detect
synchronization regimes.

1. **Read:** [System Overview](../concepts/system_overview.md)
2. **Do:** Run `notebooks/02_minimal_domain.ipynb` locally
3. **Explore:** `neuroscience_eeg` domainpack
4. **Try:** BOLD generator (fMRI from oscillators), PAC analysis, chimera detection

**Install:** `pip install scpn-phase-orchestrator[full]`

## I'm a Power Systems Engineer

You want to model grid stability, test control strategies, predict cascading failures.

1. **Read:** [Kuramoto Theory](../concepts/kuramoto_theory.md) — focus on the inertial (second-order) sections
2. **Do:** [Advanced Dynamics Guide — Power Grids](../guide/advanced_dynamics.md#second-order-inertial-kuramoto-power-grids)
3. **Explore:** `power_grid` domainpack + `InertialKuramotoEngine`
4. **Try:** Generator trip scenarios, weak coupling desynchronization tests

**Install:** `pip install scpn-phase-orchestrator`

## I'm a Quantitative Analyst

You want to detect market regimes via synchronization.

1. **Read:** [Advanced Dynamics — Financial Markets](../guide/advanced_dynamics.md#financial-market-synchronization)
2. **Explore:** `financial_markets` domainpack
3. **Try:** Hilbert phase extraction → order parameter R(t) → regime detection

**Install:** `pip install scpn-phase-orchestrator`

## I'm a Roboticist

You want to coordinate swarm formation via phase coupling.

1. **Read:** [Advanced Dynamics — Swarmalators](../guide/advanced_dynamics.md#swarmalator-dynamics)
2. **Explore:** `SwarmalatorEngine` — position + phase coupling
3. **Try:** Different J/K parameter regimes, 3D formations

**Install:** `pip install scpn-phase-orchestrator`

## I'm a Physicist / Mathematician

You want the full mathematical framework.

1. **Read:** [Kuramoto Theory](../concepts/kuramoto_theory.md) — full derivations
2. **Explore:** Simplicial (3-body), Hodge decomposition, stochastic resonance,
   Ott-Antonsen reduction, geometric integrator
3. **Try:** [Advanced Dynamics Guide](../guide/advanced_dynamics.md) — all 9 engines
4. **Deep:** FEP-Kuramoto correspondence, spectral alignment function

**Install:** `pip install scpn-phase-orchestrator[full]`

## Common Next Steps

After your entry point:

- **Control:** [Control Systems Guide](../guide/control_systems.md) — MPC, regime manager, Petri net
- **Analysis:** [Analysis Toolkit Guide](../guide/analysis_toolkit.md) — 12 monitors
- **Deployment:** [Hardware Guide](../guide/hardware_deployment.md) — Rust FFI, FPGA, WASM, GPU
- **Domains:** [Domainpack Gallery](../galleries/domainpack_gallery.md) — 32 domains
- **API:** [Full API Reference](../reference/api/index.md)
