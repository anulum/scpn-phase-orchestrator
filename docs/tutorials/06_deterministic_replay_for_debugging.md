# 06 — Deterministic Replay and Audit-First Debugging

Use an existing audit trail to validate determinism, diagnose control-path
differences, and produce reproducible diagnostics for incident review.

The steps below assume you have an audit file from the previous tutorial:
`valve_tune_audit.jsonl`.

## 1. Confirm Audit Integrity First

```bash
spo replay valve_tune_audit.jsonl --verify
```

The verification pass is the core safety check:

- same seed and binding config from the header must reproduce the logged trajectory
- step-by-step phase/metric deltas must stay within SPO tolerances
- hash chain mismatches are rejected immediately

You should see a pass line similar to:

```text
Determinism verified: 180 transitions OK
```

## 2. Generate a Machine-Readable Replay Report

```bash
spo replay valve_tune_audit.jsonl --verify --output valve_tune_replay_report.json
```

The JSON report gives you the per-step failure envelope and any tolerance gaps
without re-running external tooling.

## 3. Parse Reported Divergence for Fast Triage

When replay fails, keep the first mismatch only:

```python
import json

with open("valve_tune_replay_report.json", "r", encoding="utf-8") as fp:
    report = json.load(fp)

if report.get("status") != "ok":
    first = report["divergences"][0]
    print(f"first divergence: step={first['step']} magnitude={first['magnitude']:.3e}")
```

Useful fields in a replay report:

- `step`: failing step index
- `magnitude`: measured numerical divergence
- `max_abs_diff`: max absolute difference across logged versus replayed state
- `status`: `ok` / `mismatch`

## 4. Map Divergence to Supervisor Decisions

Link divergence points with supervisor actions to spot controller-sensitive branches:

```python
import json
from pathlib import Path

log_entries = [json.loads(line) for line in Path("valve_tune_audit.jsonl").read_text().splitlines()]
actions = [e for e in log_entries if e.get("actions")]

for i, step in enumerate(actions[:10], start=1):
    print(f"{i:02d}. step={step['step']} actions={step['actions']}")
```

If mismatch often appears after a specific action, inspect that policy rule and
thresholds first.

## 5. Build Reproducible Plots From the Audit File

This guarantees that figures are generated from the same record that replay checks:

```python
from scpn_phase_orchestrator.audit.replay import ReplayEngine
from scpn_phase_orchestrator.reporting import CoherencePlot

entries = ReplayEngine("valve_tune_audit.jsonl").load()
plotter = CoherencePlot(entries)

plotter.plot_r_timeline("diagnostic_r.png")
plotter.plot_action_audit("diagnostic_actions.png")
plotter.plot_regime_timeline("diagnostic_regimes.png")
```

## 6. Record a Full Audit-First Postmortem Bundle

Create an immutable bundle you can attach to issue trackers:

```bash
spo report valve_tune_audit.jsonl > valve_tune_summary.txt
spo explain valve_tune_audit.jsonl --markdown-out valve_tune_explain.md --max-actions 20
python - <<'PY'
from pathlib import Path
import shutil

for name in [
    "valve_tune_audit.jsonl",
    "valve_tune_summary.txt",
    "valve_tune_explain.md",
    "valve_tune_replay_report.json",
    "diagnostic_r.png",
    "diagnostic_actions.png",
]:
    p = Path(name)
    if p.exists():
        p.rename(Path("artifacts") / p.name)
PY
```

Create `artifacts/` first if needed.

## 7. Add a `run → replay → report` Gate to CI

For long-lived workflows, enforce the audit gate:

```bash
spo run domainpacks/valve_tune/binding_spec.yaml --steps 180 --seed 7 --audit valve_tune_audit_ci.jsonl
spo replay valve_tune_audit_ci.jsonl --verify --output valve_tune_ci_report.json
spo report valve_tune_audit_ci.jsonl --json-out > valve_tune_ci_summary.json
```

This pattern is the deterministic baseline for regression triage:
every successful change must still pass `--verify`.
