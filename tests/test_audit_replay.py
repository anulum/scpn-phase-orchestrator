from __future__ import annotations

import json

import numpy as np
import pytest
from click.testing import CliRunner

from scpn_phase_orchestrator.audit.logger import AuditLogger
from scpn_phase_orchestrator.audit.replay import ReplayEngine
from scpn_phase_orchestrator.cli import main
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState


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
        "stability_proxy": r_val,
    }
    log = tmp_path / "det.jsonl"
    log.write_text(json.dumps(entry) + "\n", encoding="utf-8")

    re = ReplayEngine(log)
    assert re.verify_determinism(engine, re.load()) is True


def test_verify_determinism_skips_entries_without_phases(log_file):
    engine = UPDEEngine(2, dt=0.01)
    re = ReplayEngine(log_file)
    assert re.verify_determinism(engine, re.load()) is True


def test_verify_determinism_returns_false_on_r_mismatch(tmp_path):
    n = 4
    engine = UPDEEngine(n, dt=0.01)
    rng = np.random.default_rng(1)
    phases = rng.uniform(0, 2 * np.pi, n)
    omegas = np.ones(n)
    knm = 0.3 * np.ones((n, n))
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))

    entry = {
        "phases": phases.tolist(),
        "omegas": omegas.tolist(),
        "knm": knm.tolist(),
        "alpha": alpha.tolist(),
        "zeta": 0.0,
        "psi_drive": 0.0,
        "stability_proxy": 0.0001,
    }
    log = tmp_path / "bad.jsonl"
    log.write_text(json.dumps(entry) + "\n", encoding="utf-8")

    re = ReplayEngine(log)
    assert re.verify_determinism(engine, re.load()) is False


def _make_state(r_val, psi_val):
    return UPDEState(
        layers=[LayerState(R=r_val, psi=psi_val)],
        cross_layer_alignment=np.zeros((1, 1)),
        stability_proxy=r_val,
        regime_id="nominal",
    )


def test_logger_header_and_full_state(tmp_path):
    """log_header + log_step with arrays round-trips correctly."""
    log = tmp_path / "full.jsonl"
    n = 4
    rng = np.random.default_rng(99)
    phases = rng.uniform(0, 2 * np.pi, n)
    omegas = np.ones(n)
    knm = 0.3 * np.ones((n, n))
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))

    with AuditLogger(log) as logger:
        logger.log_header(n_oscillators=n, dt=0.01, seed=99)
        logger.log_step(
            0,
            _make_state(0.8, 1.0),
            [],
            phases=phases,
            omegas=omegas,
            knm=knm,
            alpha=alpha,
        )

    re = ReplayEngine(log)
    entries = re.load()
    assert entries[0]["header"] is True
    assert entries[0]["n_oscillators"] == 4
    assert entries[0]["seed"] == 99
    assert "phases" in entries[1]
    assert len(entries[1]["phases"]) == 4
    assert entries[1]["zeta"] == 0.0


def test_roundtrip_deterministic_replay(tmp_path):
    """Full round-trip: engine -> audit log -> replay -> chained verify."""
    log = tmp_path / "roundtrip.jsonl"
    n = 4
    dt = 0.01
    engine = UPDEEngine(n, dt=dt)
    rng = np.random.default_rng(42)
    phases = rng.uniform(0, 2 * np.pi, n)
    omegas = np.ones(n)
    knm = 0.3 * np.ones((n, n))
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))

    with AuditLogger(log) as logger:
        logger.log_header(n_oscillators=n, dt=dt, seed=42)
        for step_idx in range(10):
            r_val, psi_val = engine.compute_order_parameter(phases)
            logger.log_step(
                step_idx,
                _make_state(r_val, psi_val),
                [],
                phases=phases,
                omegas=omegas,
                knm=knm,
                alpha=alpha,
            )
            phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)

    re = ReplayEngine(log)
    entries = re.load()
    header = re.load_header(entries)
    assert header is not None
    replay_eng = re.build_engine(header)
    passed, n_verified = re.verify_determinism_chained(replay_eng, entries)
    assert passed
    assert n_verified == 9


def test_chained_verification_detects_divergence(tmp_path):
    """Tampered phases are caught by chained verification."""
    n = 4
    dt = 0.01
    engine = UPDEEngine(n, dt=dt)
    rng = np.random.default_rng(7)
    phases = rng.uniform(0, 2 * np.pi, n)
    omegas = np.ones(n)
    knm = 0.3 * np.ones((n, n))
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))

    entries = [{"header": True, "n_oscillators": n, "dt": dt, "method": "euler"}]
    for step_idx in range(5):
        entries.append(
            {
                "step": step_idx,
                "phases": phases.tolist(),
                "omegas": omegas.tolist(),
                "knm": knm.tolist(),
                "alpha": alpha.tolist(),
                "zeta": 0.0,
                "psi_drive": 0.0,
                "stability": float(engine.compute_order_parameter(phases)[0]),
                "regime": "nominal",
                "layers": [{"R": 0.5, "psi": 0.0}],
            }
        )
        phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)

    # Tamper step 2 input phases
    entries[3]["phases"][0] += 1.0

    log = tmp_path / "tamper.jsonl"
    log.write_text(
        "\n".join(json.dumps(e) for e in entries) + "\n",
        encoding="utf-8",
    )

    re = ReplayEngine(log)
    loaded = re.load()
    replay_eng = re.build_engine(re.load_header(loaded))
    passed, n_verified = re.verify_determinism_chained(replay_eng, loaded)
    assert not passed
    assert n_verified == 1


def test_chained_verify_single_entry(tmp_path):
    """Single-entry log returns (True, 0) — nothing to chain."""
    entry = {
        "header": True,
        "n_oscillators": 2,
        "dt": 0.01,
        "method": "euler",
    }
    step = {
        "step": 0,
        "phases": [0.1, 0.2],
        "omegas": [1.0, 1.0],
        "knm": [[0.0, 0.3], [0.3, 0.0]],
        "alpha": [[0.0, 0.0], [0.0, 0.0]],
        "zeta": 0.0,
        "psi_drive": 0.0,
    }
    log = tmp_path / "single.jsonl"
    log.write_text(json.dumps(entry) + "\n" + json.dumps(step) + "\n", encoding="utf-8")

    re = ReplayEngine(log)
    loaded = re.load()
    eng = re.build_engine(re.load_header(loaded))
    passed, n = re.verify_determinism_chained(eng, loaded)
    assert passed
    assert n == 0


def test_cli_replay_verify_roundtrip(tmp_path):
    """CLI: run --audit -> replay --verify succeeds."""
    import yaml

    spec = {
        "name": "replay-test",
        "version": "1.0.0",
        "safety_tier": "research",
        "sample_period_s": 0.01,
        "control_period_s": 0.1,
        "layers": [
            {"name": "L1", "index": 0, "oscillator_ids": ["o0", "o1"]},
            {"name": "L2", "index": 1, "oscillator_ids": ["o2", "o3"]},
        ],
        "oscillator_families": {
            "phys": {"channel": "P", "extractor_type": "hilbert", "config": {}},
        },
        "coupling": {"base_strength": 0.45, "decay_alpha": 0.3, "templates": {}},
        "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
        "objectives": {"good_layers": [0], "bad_layers": [1]},
        "boundaries": [],
        "actuators": [],
    }
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.dump(spec), encoding="utf-8")
    audit_path = str(tmp_path / "audit.jsonl")

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["run", str(spec_path), "--steps", "10", "--audit", audit_path],
    )
    assert result.exit_code == 0

    result = runner.invoke(main, ["replay", audit_path, "--verify"])
    assert result.exit_code == 0
    assert "Determinism verified" in result.output
    assert "9 transitions OK" in result.output


def test_hash_chain_present(tmp_path):
    """Every record gets a _hash field."""
    log = tmp_path / "hash.jsonl"
    n = 4
    rng = np.random.default_rng(0)
    phases = rng.uniform(0, 2 * np.pi, n)
    omegas = np.ones(n)
    knm = 0.3 * np.ones((n, n))
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))

    with AuditLogger(log) as logger:
        logger.log_header(n_oscillators=n, dt=0.01)
        for i in range(5):
            logger.log_step(
                i,
                _make_state(0.8, 1.0),
                [],
                phases=phases,
                omegas=omegas,
                knm=knm,
                alpha=alpha,
            )

    entries = ReplayEngine(log).load()
    assert len(entries) == 6
    for entry in entries:
        assert "_hash" in entry
        assert len(entry["_hash"]) == 64


def test_hash_chain_integrity(tmp_path):
    """verify_integrity passes on an untampered log."""
    log = tmp_path / "intact.jsonl"
    n = 4
    rng = np.random.default_rng(1)
    phases = rng.uniform(0, 2 * np.pi, n)
    omegas = np.ones(n)
    knm = 0.3 * np.ones((n, n))
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))

    with AuditLogger(log) as logger:
        logger.log_header(n_oscillators=n, dt=0.01)
        for i in range(5):
            logger.log_step(
                i,
                _make_state(0.8, 1.0),
                [],
                phases=phases,
                omegas=omegas,
                knm=knm,
                alpha=alpha,
            )

    re = ReplayEngine(log)
    ok, n_verified = re.verify_integrity(re.load())
    assert ok
    assert n_verified == 6


def test_hash_chain_detects_tampering(tmp_path):
    """Mutating a field without recomputing hashes breaks the chain."""
    log = tmp_path / "tampered.jsonl"
    n = 4
    rng = np.random.default_rng(2)
    phases = rng.uniform(0, 2 * np.pi, n)
    omegas = np.ones(n)
    knm = 0.3 * np.ones((n, n))
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))

    with AuditLogger(log) as logger:
        logger.log_header(n_oscillators=n, dt=0.01)
        for i in range(5):
            logger.log_step(
                i,
                _make_state(0.8, 1.0),
                [],
                phases=phases,
                omegas=omegas,
                knm=knm,
                alpha=alpha,
            )

    entries = ReplayEngine(log).load()
    entries[3]["stability"] = 0.0
    ok, _ = ReplayEngine.verify_integrity(entries)
    assert not ok


def test_legacy_log_without_hashes(tmp_path):
    """Logs without _hash fields return (True, 0)."""
    log = tmp_path / "legacy.jsonl"
    entries = [
        {"step": 0, "regime": "nominal", "stability": 0.85, "layers": []},
        {"step": 1, "regime": "nominal", "stability": 0.9, "layers": []},
    ]
    log.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")
    re = ReplayEngine(log)
    ok, n_verified = re.verify_integrity(re.load())
    assert ok
    assert n_verified == 0


def test_sl_chained_replay(tmp_path):
    """Stuart-Landau replay: engine output matches logged next state."""
    from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

    n = 4
    dt = 0.01
    engine = StuartLandauEngine(n, dt=dt)
    rng = np.random.default_rng(42)
    phases = rng.uniform(0, 2 * np.pi, n)
    amps = np.ones(n) * 0.8
    state = np.concatenate([phases, amps])
    omegas = np.ones(n)
    mu = np.full(n, 1.0)
    knm = 0.3 * np.ones((n, n))
    np.fill_diagonal(knm, 0.0)
    knm_r = 0.1 * np.ones((n, n))
    np.fill_diagonal(knm_r, 0.0)
    alpha = np.zeros((n, n))

    entries = [{"header": True, "n_oscillators": n, "dt": dt, "amplitude_mode": True}]
    for step_idx in range(5):
        entries.append(
            {
                "step": step_idx,
                "phases": state.tolist(),
                "omegas": omegas.tolist(),
                "knm": knm.flatten().tolist(),
                "alpha": alpha.flatten().tolist(),
                "mu": mu.tolist(),
                "knm_r": knm_r.flatten().tolist(),
                "epsilon": 1.0,
                "zeta": 0.0,
                "psi_drive": 0.0,
                "stability": 0.5,
                "regime": "nominal",
                "layers": [{"R": 0.5, "psi": 0.0}],
            }
        )
        state = engine.step(state, omegas, mu, knm, knm_r, 0.0, 0.0, alpha)

    log = tmp_path / "sl_replay.jsonl"
    log.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")

    re = ReplayEngine(log)
    loaded = re.load()
    header = re.load_header(loaded)
    replay_eng = re.build_engine(header)
    assert isinstance(replay_eng, StuartLandauEngine)
    passed, n_verified = re.verify_determinism_sl_chained(replay_eng, loaded)
    assert passed
    assert n_verified == 4


def test_cli_replay_verify_no_header(tmp_path):
    """CLI: replay --verify on legacy log (no header) exits 1."""
    log = tmp_path / "legacy.jsonl"
    log.write_text(
        json.dumps({"step": 0, "regime": "nominal", "stability": 0.8, "layers": []})
        + "\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(main, ["replay", str(log), "--verify"])
    assert result.exit_code == 1
    assert "no header" in result.output
