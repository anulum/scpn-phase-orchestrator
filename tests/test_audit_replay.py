# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Audit replay tests

from __future__ import annotations

import hashlib
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


def test_replay_step_ignores_malformed_layer_containers(tmp_path):
    re = ReplayEngine(tmp_path / "unused.jsonl")

    state = re.replay_step(
        {
            "step": 0,
            "regime": "nominal",
            "stability": 0.85,
            "layers": ["not-a-layer", {"R": 0.5, "psi": 0.25}],
        }
    )

    assert len(state.layers) == 1
    assert state.layers[0] == LayerState(R=0.5, psi=0.25)


def test_replay_step_sanitizes_malformed_layer_numeric_values(tmp_path):
    re = ReplayEngine(tmp_path / "unused.jsonl")

    state = re.replay_step(
        {
            "step": 0,
            "regime": "nominal",
            "stability": 0.85,
            "layers": [
                {"R": True, "psi": float("nan")},
                {"R": float("inf")},
                {"psi": 0.25},
                {"R": 0.5, "psi": 0.75},
            ],
        }
    )

    assert state.layers == [
        LayerState(R=0.0, psi=0.0),
        LayerState(R=0.0, psi=0.0),
        LayerState(R=0.0, psi=0.25),
        LayerState(R=0.5, psi=0.75),
    ]


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "0.5"])
def test_replay_step_sanitizes_malformed_stability_proxy(tmp_path, value):
    re = ReplayEngine(tmp_path / "unused.jsonl")

    state = re.replay_step(
        {
            "step": 0,
            "regime": "nominal",
            "stability": value,
            "layers": [{"R": 0.5, "psi": 0.75}],
        }
    )

    assert state.stability_proxy == 0.0


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
        "R": r_val,
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
        "R": 0.0001,
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


@pytest.mark.parametrize("missing", ["n_oscillators", "dt"])
def test_build_engine_rejects_missing_required_header_fields(tmp_path, missing):
    re = ReplayEngine(tmp_path / "unused.jsonl")
    header = {
        "header": True,
        "n_oscillators": 4,
        "dt": 0.01,
        "method": "euler",
    }
    del header[missing]

    with pytest.raises(ValueError, match=f"audit header missing {missing}"):
        re.build_engine(header)


@pytest.mark.parametrize("n_oscillators", [0, -1])
def test_build_engine_rejects_non_positive_oscillator_counts(tmp_path, n_oscillators):
    re = ReplayEngine(tmp_path / "unused.jsonl")

    with pytest.raises(
        ValueError, match="audit header n_oscillators must be a positive integer"
    ):
        re.build_engine(
            {
                "header": True,
                "n_oscillators": n_oscillators,
                "dt": 0.01,
                "method": "euler",
            }
        )


@pytest.mark.parametrize("method", [True, "", "bogus"])
def test_build_engine_rejects_malformed_method_metadata(tmp_path, method):
    re = ReplayEngine(tmp_path / "unused.jsonl")

    with pytest.raises(ValueError, match="audit header method"):
        re.build_engine(
            {
                "header": True,
                "n_oscillators": 4,
                "dt": 0.01,
                "method": method,
            }
        )


@pytest.mark.parametrize("amplitude_mode", [1, "true", [], object()])
def test_build_engine_rejects_malformed_amplitude_mode_metadata(
    tmp_path, amplitude_mode
):
    re = ReplayEngine(tmp_path / "unused.jsonl")

    with pytest.raises(ValueError, match="audit header amplitude_mode"):
        re.build_engine(
            {
                "header": True,
                "n_oscillators": 4,
                "dt": 0.01,
                "method": "euler",
                "amplitude_mode": amplitude_mode,
            }
        )


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


def test_chained_verification_rejects_missing_required_step_arrays(tmp_path):
    engine = UPDEEngine(2, dt=0.01)
    re = ReplayEngine(tmp_path / "unused.jsonl")

    passed, n_verified = re.verify_determinism_chained(
        engine,
        [
            {
                "step": 0,
                "phases": [0.1, 0.2],
                "knm": [[0.0, 0.3], [0.3, 0.0]],
                "alpha": [[0.0, 0.0], [0.0, 0.0]],
            },
            {
                "step": 1,
                "phases": [0.2, 0.3],
                "omegas": [1.0, 1.0],
                "knm": [[0.0, 0.3], [0.3, 0.0]],
                "alpha": [[0.0, 0.0], [0.0, 0.0]],
            },
        ],
    )

    assert not passed
    assert n_verified == 0


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


def test_cli_replay_verify_with_audit_key_rejects_unsigned_log(tmp_path, monkeypatch):
    """CLI replay verification must fail closed when signatures are required."""
    monkeypatch.setenv("SPO_AUDIT_KEY", "unit-test-secret-key")
    log = tmp_path / "unsigned.jsonl"
    entries = [
        {"header": True, "n_oscillators": 2, "dt": 0.01, "method": "euler"},
        {
            "step": 0,
            "phases": [0.1, 0.2],
            "omegas": [1.0, 1.0],
            "knm": [[0.0, 0.3], [0.3, 0.0]],
            "alpha": [[0.0, 0.0], [0.0, 0.0]],
            "zeta": 0.0,
            "psi_drive": 0.0,
        },
    ]
    log.write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(main, ["replay", str(log), "--verify"])
    assert result.exit_code == 1
    assert "audit integrity FAILED" in result.output


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


def test_integrity_with_audit_key_rejects_unsigned_records(tmp_path, monkeypatch):
    """Configured SPO_AUDIT_KEY makes unsigned development logs fail closed."""
    monkeypatch.setenv("SPO_AUDIT_KEY", "unit-test-secret-key")
    log = tmp_path / "unsigned.jsonl"
    log.write_text(
        json.dumps({"step": 0, "regime": "nominal", "stability": 0.85}) + "\n",
        encoding="utf-8",
    )

    ok, n_verified = ReplayEngine.verify_integrity(ReplayEngine(log).load())
    assert not ok
    assert n_verified == 0


def test_integrity_with_audit_key_rejects_signature_tampering(tmp_path, monkeypatch):
    """Hash-preserving payload mutation must still fail HMAC verification."""
    monkeypatch.setenv("SPO_AUDIT_KEY", "unit-test-secret-key")
    log = tmp_path / "signed.jsonl"
    with AuditLogger(log) as logger:
        logger.log_step(0, _make_state(0.8, 1.0), [])

    entries = ReplayEngine(log).load()
    entries[0]["stability"] = 0.1
    clean = {k: v for k, v in entries[0].items() if k != "_hash"}
    json_line = json.dumps(clean, separators=(",", ":"), sort_keys=True)
    entries[0]["_hash"] = hashlib.sha256(("0" * 64 + json_line).encode()).hexdigest()

    ok, n_verified = ReplayEngine.verify_integrity(entries)
    assert not ok
    assert n_verified == 0


def test_integrity_with_audit_key_rejects_audit_mode_tampering(tmp_path, monkeypatch):
    """Signed JSONL records must bind audit mode into HMAC verification."""
    monkeypatch.setenv("SPO_AUDIT_KEY", "unit-test-secret-key")
    log = tmp_path / "signed.jsonl"
    with AuditLogger(log) as logger:
        logger.log_step(0, _make_state(0.8, 1.0), [])

    entries = ReplayEngine(log).load()
    entries[0]["_audit_mode"] = "unsigned-development"
    clean = {k: v for k, v in entries[0].items() if k != "_hash"}
    json_line = json.dumps(clean, separators=(",", ":"), sort_keys=True)
    entries[0]["_hash"] = hashlib.sha256(("0" * 64 + json_line).encode()).hexdigest()

    ok, n_verified = ReplayEngine.verify_integrity(entries)
    assert not ok
    assert n_verified == 0


def test_integrity_verifies_records_across_rotated_audit_keys(tmp_path, monkeypatch):
    """A JSON keyring verifies old and current records after key rotation."""
    old_key = "old-rotation-key"
    new_key = "new-rotation-key"
    old_key_id = hashlib.sha256(old_key.encode()).hexdigest()[:16]
    new_key_id = hashlib.sha256(new_key.encode()).hexdigest()[:16]
    log = tmp_path / "rotated.jsonl"

    monkeypatch.setenv("SPO_AUDIT_KEY", old_key)
    with AuditLogger(log) as logger:
        logger.log_header(n_oscillators=2, dt=0.01)

    monkeypatch.setenv("SPO_AUDIT_KEY", new_key)
    with AuditLogger(log) as logger:
        logger.log_step(0, _make_state(0.8, 1.0), [])

    entries = ReplayEngine(log).load()
    assert [entry["_signature"]["key_id"] for entry in entries] == [
        old_key_id,
        new_key_id,
    ]
    ok, n_verified = ReplayEngine.verify_integrity(entries)
    assert not ok
    assert n_verified == 0

    monkeypatch.setenv(
        "SPO_AUDIT_KEYRING",
        json.dumps({old_key_id: old_key, new_key_id: new_key}),
    )
    ok, n_verified = ReplayEngine.verify_integrity(entries)
    assert ok
    assert n_verified == 2


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
