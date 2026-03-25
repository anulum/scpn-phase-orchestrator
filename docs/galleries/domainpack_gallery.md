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
