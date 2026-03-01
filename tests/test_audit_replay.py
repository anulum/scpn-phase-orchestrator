from __future__ import annotations

import json

import numpy as np
import pytest

from scpn_phase_orchestrator.audit.replay import ReplayEngine
from scpn_phase_orchestrator.upde.engine import UPDEEngine


@pytest.fixture
def log_file(tmp_path):
    log = tmp_path / "audit.jsonl"
    entries = [
        {
            "step": 0,
            "regime": "nominal",
            "stability": 0.85,
            "layers": [{"R": 0.8, "psi": 1.0}, {"R": 0.9, "psi": 0.5}],
        },
        {"event": "transition", "from": "nominal", "to": "degraded"},
        {
            "step": 1,
            "regime": "degraded",
            "stability": 0.6,
            "layers": [{"R": 0.5, "psi": 2.0}, {"R": 0.4, "psi": 1.5}],
        },
    ]
    log.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")
    return log


def test_load_entries(log_file):
    re = ReplayEngine(log_file)
    entries = re.load()
    assert len(entries) == 3


def test_replay_step_reconstructs_state(log_file):
    re = ReplayEngine(log_file)
    entries = re.load()
    step_data = entries[0]
    state = re.replay_step(step_data)
    assert len(state.layers) == 2
    assert pytest.approx(0.8) == state.layers[0].R
    assert state.regime_id == "nominal"
    assert state.stability_proxy == pytest.approx(0.85)


def test_verify_determinism_with_matching_data(tmp_path):
    n = 4
    engine = UPDEEngine(n, dt=0.01)
    rng = np.random.default_rng(0)
    phases = rng.uniform(0, 2 * np.pi, n)
    omegas = np.ones(n)
    knm = 0.3 * np.ones((n, n))
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))

    new_phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)
    r_val, _ = engine.compute_order_parameter(new_phases)

    entry = {
        "phases": phases.tolist(),
        "omegas": omegas.tolist(),
        "knm": knm.tolist(),
        "alpha": alpha.tolist(),
        "zeta": 0.0,
        "psi_drive": 0.0,
        "layers": [{"R": r_val}],
    }
    log = tmp_path / "det.jsonl"
    log.write_text(json.dumps(entry) + "\n", encoding="utf-8")

    re = ReplayEngine(log)
    assert re.verify_determinism(engine, re.load()) is True


def test_verify_determinism_skips_entries_without_phases(log_file):
    engine = UPDEEngine(2, dt=0.01)
    re = ReplayEngine(log_file)
    assert re.verify_determinism(engine, re.load()) is True
