# Domainpack Gallery

Each domainpack ships a `binding_spec.yaml` that maps a real-world problem onto
SCPN Kuramoto oscillators. The notebooks below demonstrate baseline vs orchestrated
simulations.

## Notebooks

| # | Domainpack | Oscillators | Key Feature | Notebook |
|---|-----------|-------------|-------------|----------|
| 01 | **queuewaves** | 6 (micro/meso/macro) | Retry-storm recovery, supervisor 9x speedup | `01_queuewaves_retry_storm.ipynb` |
| 02 | **minimal_domain** | 4 (lower/upper) | Simplest possible spec, coherence convergence | `02_minimal_domain.ipynb` |
| 03 | **geometry_walk** | 8 (local/global) | Symbolic channel, graph-walk phases, zeta drive | `03_geometry_walk.ipynb` |
| 04 | **bio_stub** | 16 (4 scales) | Multi-scale biology, imprint memory | `04_bio_stub.ipynb` |
| 05 | **manufacturing_spc** | 9 (sensor/machine/line) | Bad-layer suppression, policy rules | `05_manufacturing_spc.ipynb` |
| 06 | **stuart_landau** | 8 | Phase-amplitude ODE, bifurcation tracking, PAC | `06_stuart_landau_amplitude.ipynb` |
| 07 | **policy_petri_net** | 8 | Policy DSL, regime FSM, Petri net sequencing | `07_policy_petri_net.ipynb` |
| 08 | **audit_replay** | 8 | SHA256-chained audit trail, deterministic replay | `08_audit_replay.ipynb` |
| 09 | **binding_spec** | varies | Binding spec schema walkthrough, validation | `09_binding_spec.ipynb` |
| 10 | **adapters** | varies | Bridge adapters (fusion, quantum, neurocore) | `10_reporting_adapters.ipynb` |
| 11 | **identity_coherence** | 30 (6 layers) | SSGF identity model, chimera + plasticity | `11_identity_coherence.ipynb` |
| 12 | **autotune** | varies | Frequency ID, coupling estimation pipeline | `12_autotune_pipeline.ipynb` |
| 13 | **ssgf** | varies | SSGF free energy closure | `13_ssgf_closure.ipynb` |
| 14 | **chimera** | varies | Chimera state identification and visualization | `14_chimera_detection.ipynb` |
| 15 | **spectral** | varies | Spectral Alignment Function (SAF) optimization | `15_spectral_analysis.ipynb` |
| 16 | **sleep_architecture** | 8 (4 layers) | EEG sleep stage classification from R | `16_sleep_staging.ipynb` |
| 17 | **power_grid** | 12 (5 layers) | Inertial Kuramoto, generator trip transients | `17_power_grid_stability.ipynb` |
| 18 | **financial_markets** | 8 (4 layers) | Hilbert phase, order parameter regime detection | `18_market_regime_detection.ipynb` |
| 19 | **swarm_robotics** | 8 (3 layers) | Swarmalator spatial + phase coupling | `19_swarmalator_dynamics.ipynb` |

## All 32 Domainpacks

| Pack | Domain | Layers | Oscillators | Channels |
|------|--------|--------|-------------|----------|
| `autonomous_vehicles` | Vehicle platoons | 3 | 8 | P/I |
| `bio_stub` | Multi-scale biology | 4 | 16 | P/I/S |
| `brain_connectome` | HCP-inspired structural connectivity | 4 | 12 | P/I/S |
| `cardiac_rhythm` | Cardiology | 4 | 10 | P/I |
| `chemical_reactor` | Process control | 4 | 10 | P/I |
| `circadian_biology` | Chronobiology | 4 | 10 | S |
| `epidemic_sir` | Epidemiology | 3 | 8 | P/I |
| `financial_markets` | Stock sync and crash detection | 4 | 8 | P/I/S |
| `firefly_swarm` | Ecology | 2 | 8 | P/I |
| `fusion_equilibrium` | Fusion plasma | 6 | 12 | P/I |
| `gene_oscillator` | Repressilator + quorum sensing | 3 | 6 | P/I/S |
| `geometry_walk` | Graph systems | 2 | 8 | S |
| `identity_coherence` | Consciousness/identity model (SSGF) | 6 | 30 | P/I/S |
| `laser_array` | Photonics | 3 | 8 | P/I |
| `manufacturing_spc` | Manufacturing | 3 | 9 | P/I/S |
| `metaphysics_demo` | P/I/S showcase | 3 | 7 | P/I/S |
| `minimal_domain` | Synthetic baseline | 2 | 4 | P |
| `musical_acoustics` | Consonance and groove via sync | 3 | 9 | P/I/S |
| `network_security` | Cybersecurity | 3 | 8 | I |
| `neuroscience_eeg` | Neuroscience | 6 | 14 | P/I |
| `plasma_control` | Tokamak plasma | 8 | 16 | P/I |
| `pll_clock` | Telecommunications | 3 | 8 | P/I |
| `power_grid` | Power systems | 5 | 12 | P/I |
| `quantum_simulation` | Quantum computing | 3 | 8 | P/I |
| `queuewaves` | Cloud/queues | 3 | 6 | P/I |
| `robotic_cpg` | Quadruped CPG locomotion | 4 | 8 | P/I/S |
| `rotating_machinery` | Vibration | 4 | 10 | P/I |
| `satellite_constellation` | Aerospace | 3 | 8 | P/I |
| `sleep_architecture` | AASM sleep staging from R values | 4 | 8 | P/I/S |
| `swarm_robotics` | Robotics | 3 | 8 | P/I |
| `traffic_flow` | Transportation | 4 | 10 | P/I |
| `vortex_shedding` | Wake dynamics (Stuart-Landau) | 3 | 9 | P/I/S |

## Benchmark Results (Measured 2026-03-28)

All 32 domainpacks tested via `spo demo --steps 20` on Kaggle (Linux, Python 3.12,
NumPy fallback — no Rust kernel). Zero failures.

| Domainpack | Oscillators | Layers | R (20 steps) | Regime |
|------------|-------------|--------|--------------|--------|
| autonomous_vehicles | 8 | 3 | 0.394 | degraded |
| bio_stub | 16 | 4 | 0.274 | degraded |
| brain_connectome | 12 | 4 | 0.375 | degraded |
| cardiac_rhythm | 10 | 4 | 0.130 | nominal |
| chemical_reactor | 10 | 4 | 0.410 | nominal |
| circadian_biology | 10 | 4 | 0.357 | degraded |
| epidemic_sir | 8 | 3 | 0.590 | nominal |
| financial_markets | 8 | 4 | 1.000 | nominal |
| firefly_swarm | 8 | 2 | 0.566 | degraded |
| fusion_equilibrium | 12 | 6 | 0.196 | nominal |
| gene_oscillator | 6 | 3 | 1.000 | nominal |
| geometry_walk | 8 | 2 | 0.795 | nominal |
| identity_coherence | 35 | 6 | 0.352 | degraded |
| laser_array | 8 | 3 | 0.312 | degraded |
| manufacturing_spc | 9 | 3 | 0.530 | nominal |
| metaphysics_demo | 7 | 3 | 0.572 | degraded |
| minimal_domain | 4 | 2 | 0.524 | degraded |
| musical_acoustics | 9 | 3 | 0.998 | nominal |
| network_security | 8 | 3 | 0.575 | nominal |
| neuroscience_eeg | 14 | 6 | 0.114 | nominal |
| plasma_control | 16 | 8 | 0.321 | nominal |
| pll_clock | 8 | 3 | 0.209 | degraded |
| power_grid | 12 | 5 | 0.345 | nominal |
| quantum_simulation | 8 | 2 | 0.608 | degraded |
| queuewaves | 6 | 3 | 0.735 | nominal |
| robotic_cpg | 8 | 4 | 0.999 | nominal |
| rotating_machinery | 10 | 4 | 0.287 | nominal |
| satellite_constellation | 8 | 3 | 0.503 | nominal |
| sleep_architecture | 8 | 4 | 0.229 | nominal |
| swarm_robotics | 8 | 3 | 0.380 | degraded |
| traffic_flow | 10 | 4 | 0.492 | nominal |
| vortex_shedding | 9 | 3 | 0.999 | nominal |

**Notes:** R values at step 20 from random initial phases. "degraded" =
R < 0.6 (not yet synchronised). Longer runs converge for all domainpacks.
Financial markets and gene oscillator reach R=1.0 within 20 steps due to
strong coupling topology.

## Running Locally

```bash
pip install -e ".[dev,plot]"
jupyter lab notebooks/
```

19 notebooks total. Notebooks 01--07 are validated in CI via `jupyter nbconvert --execute`;
the remaining 12 run locally but are excluded from CI due to optional dependencies (JAX,
Qiskit) or nbconvert performance constraints.

## Adding a New Domainpack

1. Create `domainpacks/<name>/binding_spec.yaml` following the
   [binding spec schema](../specs/binding_spec.schema.json)
2. Add `domainpacks/<name>/policy.yaml` with supervisor rules
3. Create a notebook in `notebooks/` following the pattern above
4. Add a row to this table
