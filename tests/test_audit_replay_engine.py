# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Audit replay engine tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.audit.replay import ReplayEngine
from scpn_phase_orchestrator.upde.engine import UPDEEngine


def test_load_parses_jsonl(tmp_path):
    log = tmp_path / "audit.jsonl"
    log.write_text(
        '{"step": 0, "layers": [{"R": 0.9, "psi": 0.1}]}\n'
        '{"step": 1, "layers": [{"R": 0.8, "psi": 0.2}]}\n',
        encoding="utf-8",
    )
    engine = ReplayEngine(log)
    entries = engine.load()
    assert len(entries) == 2
    assert entries[0]["step"] == 0


def test_replay_step_reconstructs_state():
    engine = ReplayEngine("dummy.jsonl")
    entry = {
        "layers": [{"R": 0.75, "psi": 1.2}],
        "stability": 0.75,
        "regime": "nominal",
    }
    state = engine.replay_step(entry)
    assert len(state.layers) == 1
    assert state.layers[0].R == 0.75
    assert state.stability_proxy == 0.75


def test_replay_step_empty_layers():
    engine = ReplayEngine("dummy.jsonl")
    state = engine.replay_step({"layers": []})
    assert state.layers == []


def test_verify_determinism_matching(tmp_path):
    n = 4
    dt = 0.01
    engine = UPDEEngine(n_oscillators=n, dt=dt)
    phases = np.zeros(n)
    omegas = np.ones(n)
    knm = np.zeros((n, n))
    alpha = np.zeros((n, n))

    new_phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)
    r, _ = engine.compute_order_parameter(new_phases)

    entry = {
        "phases": phases.tolist(),
        "omegas": omegas.tolist(),
        "knm": knm.tolist(),
        "alpha": alpha.tolist(),
        "zeta": 0.0,
        "psi_drive": 0.0,
        "stability_proxy": r,
    }
    replay = ReplayEngine("dummy.jsonl")
    assert replay.verify_determinism(engine, [entry]) is True


def test_verify_determinism_mismatch(tmp_path):
    n = 4
    engine = UPDEEngine(n_oscillators=n, dt=0.01)
    entry = {
        "phases": np.zeros(n).tolist(),
        "omegas": np.ones(n).tolist(),
        "knm": np.zeros((n, n)).tolist(),
        "alpha": np.zeros((n, n)).tolist(),
        "stability_proxy": 999.0,  # intentionally wrong
    }
    replay = ReplayEngine("dummy.jsonl")
    assert replay.verify_determinism(engine, [entry]) is False


def test_verify_determinism_skips_entries_without_phases():
    engine = UPDEEngine(n_oscillators=4, dt=0.01)
    replay = ReplayEngine("dummy.jsonl")
    assert replay.verify_determinism(engine, [{"step": 0}]) is True
