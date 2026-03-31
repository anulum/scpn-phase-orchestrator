# Reporting

Matplotlib-based visualisation for coherence analysis and phase dynamics.
Requires the `plot` optional extra:

```bash
pip install scpn-phase-orchestrator[plot]
```

## Available Plots

The reporting module generates publication-quality figures for
synchronisation analysis:

| Function | Description | Typical use |
|----------|-------------|-------------|
| `plot_order_parameter` | R(t) time series with regime bands | Monitor sync health over time |
| `plot_phase_portrait` | Oscillator phases on the unit circle | Visualise sync state at one timestep |
| `plot_coupling_matrix` | $K_{nm}$ heatmap with spectral summary | Inspect coupling topology |
| `plot_bifurcation_diagram` | R vs K with $K_c$ marker | Characterise sync transition |
| `plot_recurrence` | Binary recurrence matrix | Detect dynamical regimes |
| `plot_sync_dashboard` | Multi-panel: R, PLV, NPE, chimera | Comprehensive sync summary |
| `plot_phase_trajectory` | Phase time series (unwrapped) | Track individual oscillators |
| `plot_spectral_density` | Eigenvalue distribution of $L$ | Assess network connectivity |

## Usage Pattern

All plot functions accept an optional `ax` parameter (matplotlib Axes)
for embedding in custom layouts. When `ax` is omitted, a new figure
is created.

```python
import matplotlib.pyplot as plt
from scpn_phase_orchestrator.reporting.plots import (
    plot_order_parameter,
    plot_phase_portrait,
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: R(t) over simulation
plot_order_parameter(R_history, dt=0.01, ax=axes[0])

# Right: phase snapshot at final timestep
plot_phase_portrait(phases_final, ax=axes[1])

plt.tight_layout()
plt.savefig("sync_analysis.png", dpi=150)
```

## Colour Conventions

- **R(t)** plot: green band for NOMINAL (R > 0.6), yellow for DEGRADED
  (0.3 < R < 0.6), red for CRITICAL (R < 0.3)
- **Phase portrait**: oscillators coloured by channel (P=blue, I=green, S=orange)
- **Coupling heatmap**: sequential colourmap (viridis), zero diagonal masked

## Pipeline integration

```
UPDEEngine ──→ phases, R(t) ──→ plot_order_parameter()
                                 plot_phase_portrait()
                                 plot_sync_dashboard()

CouplingBuilder ──→ K_nm ──→ plot_coupling_matrix()

bifurcation ──→ R(K) curve ──→ plot_bifurcation_diagram()

embedding ──→ recurrence matrix ──→ plot_recurrence()
```

All plots consume engine output directly — no intermediate
transformation. This ensures the visualisation always reflects
the actual pipeline state.

## Plot gallery

### Order parameter time series

```python
plot_order_parameter(R_history, dt=0.01, ax=ax)
```

Shows R(t) with coloured background bands:
- Green: NOMINAL (R > 0.6)
- Yellow: DEGRADED (0.3 < R < 0.6)
- Red: CRITICAL (R < 0.3)

Vertical dashed lines mark regime transitions.

### Phase portrait

```python
plot_phase_portrait(phases, ax=ax)
```

Plots oscillator phases on the unit circle with the mean-field
vector (R, ψ) as a bold arrow. Oscillators coloured by channel
(P=blue, I=green, S=orange).

### Coupling heatmap

```python
plot_coupling_matrix(knm, ax=ax)
```

Heatmap of K_nm with viridis colourmap, zero diagonal masked.
Annotated with Fiedler value λ₂ and mean coupling strength.

### Sync dashboard

```python
plot_sync_dashboard(R_history, phases, knm, plv_matrix, ax=axes)
```

Multi-panel figure combining R(t), phase portrait, K_nm heatmap,
and PLV matrix in a single publication-ready layout.

::: scpn_phase_orchestrator.reporting.plots
