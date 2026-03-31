# Reporting

Matplotlib-based visualisation for coherence analysis and phase dynamics.
Requires the `plot` optional extra:

```bash
pip install scpn-phase-orchestrator[plot]
```

## CoherencePlot

The reporting module provides a single class `CoherencePlot` that
consumes JSONL audit log data and produces diagnostic figures.

```python
CoherencePlot(log_data: list[dict])
```

The constructor accepts a list of parsed audit log records (from
`ReplayEngine.load()` or direct JSON parsing). It filters to step
records containing `"step"` and `"layers"` fields.

### Available plots

| Method | Output | Description |
|--------|--------|-------------|
| `plot_r_timeline(output_path)` | PNG | Per-layer R over simulation steps |
| `plot_regime_timeline(output_path)` | PNG | Regime epochs as coloured horizontal bands |
| `plot_action_audit(output_path)` | PNG | R(t) with actuation event markers |
| `plot_amplitude_timeline(output_path)` | PNG | Mean amplitude and subcritical fraction |
| `plot_pac_heatmap(output_path)` | PNG | Phase-amplitude coupling matrix |

All methods return `Path` to the saved figure.

### Regime colour conventions

| Regime | Colour | Hex |
|--------|--------|-----|
| NOMINAL | Green | #2ecc71 |
| DEGRADED | Orange | #f39c12 |
| CRITICAL | Red | #e74c3c |
| RECOVERY | Blue | #3498db |

## Pipeline integration

```
AuditLogger.log_step() ──→ audit.jsonl
                                │
                                ↓
                       ReplayEngine.load()
                                │
                                ↓
                       CoherencePlot(log_data)
                                │
               ┌────────────────┼────────────────┐
               ↓                ↓                ↓
        plot_r_timeline  plot_regime_timeline  plot_action_audit
               │                │                │
               ↓                ↓                ↓
          r_timeline.png  regime.png        actions.png
```

The reporting module consumes audit log output. It does not connect
to the engine directly — all data passes through the audit trail,
ensuring that plots match the auditable record.

## Usage

```python
from scpn_phase_orchestrator.audit.replay import ReplayEngine
from scpn_phase_orchestrator.reporting.plots import CoherencePlot

# Load audit log
replay = ReplayEngine("audit.jsonl")
entries = replay.load()

# Generate diagnostic plots
plotter = CoherencePlot(entries)
plotter.plot_r_timeline("output/r_timeline.png")
plotter.plot_regime_timeline("output/regime.png")
plotter.plot_action_audit("output/actions.png")
plotter.plot_amplitude_timeline("output/amplitude.png")
plotter.plot_pac_heatmap("output/pac.png")
```

## Internal extraction methods

| Method | Returns | Description |
|--------|---------|-------------|
| `_extract_r_series` | `(steps, n_layers, series)` | Per-layer R arrays |
| `_extract_regime_epochs` | `[(regime, start, end)]` | Regime change boundaries |
| `_extract_actions` | `(steps, r_global, knob_steps)` | Action event indices |
| `_extract_amplitude` | `(steps, amps, sub_frac)` | Amplitude time series |
| `_extract_pac_matrix` | `(n, matrix)` | PAC from last log record |

::: scpn_phase_orchestrator.reporting.plots
