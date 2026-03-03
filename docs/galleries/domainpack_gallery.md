# Domainpack Gallery

Each domainpack ships a `binding_spec.yaml` that maps a real-world problem onto
SCPN Kuramoto oscillators. The notebooks below demonstrate baseline vs orchestrated
simulations for every built-in domainpack.

## Notebooks

| # | Domainpack | Oscillators | Key Feature | Notebook |
|---|-----------|-------------|-------------|----------|
| 01 | **queuewaves** | 6 (micro/meso/macro) | Retry-storm recovery, supervisor 9x speedup | `01_queuewaves_retry_storm.ipynb` |
| 02 | **minimal_domain** | 4 (lower/upper) | Simplest possible spec, coherence convergence | `02_minimal_domain.ipynb` |
| 03 | **geometry_walk** | 8 (local/global) | Symbolic channel, graph-walk phases, zeta drive | `03_geometry_walk.ipynb` |
| 04 | **bio_stub** | 16 (4 scales) | Multi-scale biology, imprint memory | `04_bio_stub.ipynb` |
| 05 | **manufacturing_spc** | 9 (sensor/machine/line) | Bad-layer suppression, policy rules | `05_manufacturing_spc.ipynb` |

## Running Locally

```bash
pip install -e ".[dev,plot]"
jupyter lab notebooks/
```

All notebooks are validated in CI via `jupyter nbconvert --execute`.

## Adding a New Domainpack

1. Create `domainpacks/<name>/binding_spec.yaml` following the
   [binding spec schema](../specs/binding_spec.schema.json)
2. Add `domainpacks/<name>/policy.yaml` with supervisor rules
3. Create a notebook in `notebooks/` following the pattern above
4. Add a row to this table
