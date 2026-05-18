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

When the audit trail contains a run header, `spo report --json-out` includes
the resolved `binding_summary`. N-channel runs also expose `channel_algebra` at
the top level of the JSON report so downstream tools can read channel groups,
derived channels, runtime evidence, and missing required channel evidence
without re-parsing the binding spec.

The text report also prints a compact channel-algebra line when the audit
header contains one, including required/optional/derived/delayed/uncertain
counts and any missing required channel evidence.

If the audit stream includes passive integrated-information monitor records
with `monitor: integrated_information`, the JSON summary includes an
`integrated_information` block with latest Phi proxy values, normalised Phi
values, series data, record count, and the claim boundary. The text report
prints a compact line with the latest Phi proxy, normalised Phi, total
integration, and number of monitor records.

Programmatic tools can use `build_audit_report_summary()` directly to get the
same JSON-ready report payload as `spo report --json-out`.

## Usage

```python
from scpn_phase_orchestrator.runtime.replay import ReplayEngine
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

## Summary Builder

::: scpn_phase_orchestrator.reporting.summary

## Explainability

Human-readable report helpers that translate audit and supervisor records into
plain diagnostic summaries for notebooks, demos, and operator-facing reports.

::: scpn_phase_orchestrator.reporting.explainability
