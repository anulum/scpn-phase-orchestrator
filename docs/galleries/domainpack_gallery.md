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

## All 24 Domainpacks

| Pack | Domain | Layers | Oscillators | Channels |
|------|--------|--------|-------------|----------|
| `autonomous_vehicles` | Vehicle platoons | 3 | 8 | P/I |
| `bio_stub` | Multi-scale biology | 4 | 16 | P/I/S |
| `cardiac_rhythm` | Cardiology | 4 | 10 | P/I |
| `chemical_reactor` | Process control | 4 | 10 | P/I |
| `circadian_biology` | Chronobiology | 4 | 10 | S |
| `epidemic_sir` | Epidemiology | 3 | 8 | P/I |
| `firefly_swarm` | Ecology | 2 | 8 | P/I |
| `fusion_equilibrium` | Fusion plasma | 6 | 12 | P/I |
| `geometry_walk` | Graph systems | 2 | 8 | S |
| `laser_array` | Photonics | 3 | 8 | P/I |
| `manufacturing_spc` | Manufacturing | 3 | 9 | P/I/S |
| `metaphysics_demo` | P/I/S showcase | 3 | 7 | P/I/S |
| `minimal_domain` | Synthetic baseline | 2 | 4 | P |
| `network_security` | Cybersecurity | 3 | 8 | I |
| `neuroscience_eeg` | Neuroscience | 6 | 14 | P/I |
| `plasma_control` | Tokamak plasma | 8 | 16 | P/I |
| `pll_clock` | Telecommunications | 3 | 8 | P/I |
| `power_grid` | Power systems | 5 | 12 | P/I |
| `quantum_simulation` | Quantum computing | 3 | 8 | P/I |
| `queuewaves` | Cloud/queues | 3 | 6 | P/I |
| `rotating_machinery` | Vibration | 4 | 10 | P/I |
| `satellite_constellation` | Aerospace | 3 | 8 | P/I |
| `swarm_robotics` | Robotics | 3 | 8 | P/I |
| `traffic_flow` | Transportation | 4 | 10 | P/I |

## Running Locally

```bash
pip install -e ".[dev,plot]"
jupyter lab notebooks/
```

All 7 notebooks are validated in CI via `jupyter nbconvert --execute`.

## Adding a New Domainpack

1. Create `domainpacks/<name>/binding_spec.yaml` following the
   [binding spec schema](../specs/binding_spec.schema.json)
2. Add `domainpacks/<name>/policy.yaml` with supervisor rules
3. Create a notebook in `notebooks/` following the pattern above
4. Add a row to this table
