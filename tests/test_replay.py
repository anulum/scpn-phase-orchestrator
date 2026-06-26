# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — deterministic replay tests

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.runtime.audit_logger import AuditLogger
from scpn_phase_orchestrator.runtime.replay import ReplayEngine
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState
from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine


def _make_state(r_value: float = 0.8, psi_value: float = 1.0) -> UPDEState:
    return UPDEState(
        layers=[LayerState(R=r_value, psi=psi_value)],
        cross_layer_alignment=np.zeros((1, 1), dtype=np.float64),
        stability_proxy=r_value,
        regime_id="nominal",
    )


def _rewrite_first_hash(entry: dict[str, Any]) -> dict[str, Any]:
    clean = {key: value for key, value in entry.items() if key != "_hash"}
    json_line = json.dumps(clean, separators=(",", ":"), sort_keys=True)
    entry["_hash"] = hashlib.sha256((("0" * 64) + json_line).encode()).hexdigest()
    return entry


def _signed_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setenv("SPO_AUDIT_KEY", "unit-test-secret-key")
    path = tmp_path / "signed.jsonl"
    with AuditLogger(path) as logger:
        logger.log_step(0, _make_state(), [])
    return ReplayEngine(path).load()[0]


def _verify_rewritten_signed_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[dict[str, Any]], None],
) -> tuple[bool, int]:
    entry = _signed_entry(tmp_path, monkeypatch)
    mutate(entry)
    return ReplayEngine.verify_integrity([_rewrite_first_hash(entry)])


def _sl_entries(
    engine: StuartLandauEngine,
) -> list[dict[str, object]]:
    n = 2
    omegas = np.ones(n, dtype=np.float64)
    mu = np.ones(n, dtype=np.float64)
    knm = np.array([[0.0, 0.2], [0.2, 0.0]], dtype=np.float64)
    knm_r = np.zeros((n, n), dtype=np.float64)
    alpha = np.zeros((n, n), dtype=np.float64)
    state = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    next_state = engine.step(state, omegas, mu, knm, knm_r, 0.0, 0.0, alpha)
    return [
        {
            "phases": state[:n].tolist(),
            "amplitudes": state[n:].tolist(),
            "omegas": omegas.tolist(),
            "mu": mu.tolist(),
            "knm": knm.ravel().tolist(),
            "knm_r": knm_r.ravel().tolist(),
            "alpha": alpha.ravel().tolist(),
            "zeta": 0.0,
            "psi_drive": 0.0,
        },
        {
            "phases": next_state[:n].tolist(),
            "amplitudes": next_state[n:].tolist(),
            "omegas": omegas.tolist(),
            "mu": mu.tolist(),
            "knm": knm.ravel().tolist(),
            "knm_r": knm_r.ravel().tolist(),
            "alpha": alpha.ravel().tolist(),
            "zeta": 0.0,
            "psi_drive": 0.0,
        },
    ]


def test_replay_load_surfaces_malformed_json_syntax(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    path.write_text('{"event": "operator_note"\n', encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        ReplayEngine(path).load()


def test_replay_load_rejects_non_object_json_lines(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line must be an object"):
        ReplayEngine(path).load()


def test_replay_load_rejects_non_finite_json_constants(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    path.write_text('{"event":"operator_note","value":NaN}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="finite JSON"):
        ReplayEngine(path).load()


def test_replay_load_rejects_duplicate_json_object_keys(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    path.write_text('{"event":"operator_note","event":"tampered"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="canonical finite JSON"):
        ReplayEngine(path).load()


def test_replay_step_treats_non_list_layers_as_empty(tmp_path: Path) -> None:
    state = ReplayEngine(tmp_path / "unused.jsonl").replay_step(
        {"layers": {"R": 0.9, "psi": 0.1}, "stability": 0.25}
    )

    assert state.layers == []
    assert state.stability_proxy == pytest.approx(0.25)


def test_build_engine_rejects_non_positive_dt(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="audit header dt"):
        ReplayEngine(tmp_path / "unused.jsonl").build_engine(
            {"header": True, "n_oscillators": 2, "dt": 0.0}
        )


def test_integrity_rejects_invalid_verification_key_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPO_AUDIT_KEY", "")

    assert ReplayEngine.verify_integrity([]) == (False, 0)


def test_replay_integrity_rejects_non_finite_hashed_entries() -> None:
    entry: dict[str, object] = {"event": "operator_note", "value": float("nan")}
    json_line = json.dumps(entry, separators=(",", ":"), sort_keys=True)
    entry["_hash"] = hashlib.sha256((("0" * 64) + json_line).encode()).hexdigest()

    assert ReplayEngine.verify_integrity([entry]) == (False, 0)


def test_stuart_landau_chained_replay_accepts_single_entry(tmp_path: Path) -> None:
    engine = StuartLandauEngine(2, dt=0.01)
    replay = ReplayEngine(tmp_path / "unused.jsonl")

    assert replay.verify_determinism_sl_chained(engine, _sl_entries(engine)[:1]) == (
        True,
        0,
    )


def test_stuart_landau_chained_replay_detects_next_state_mismatch(
    tmp_path: Path,
) -> None:
    engine = StuartLandauEngine(2, dt=0.01)
    entries = _sl_entries(engine)
    phases = entries[1]["phases"]
    assert isinstance(phases, list)
    phases[0] = 99.0

    assert ReplayEngine(tmp_path / "unused.jsonl").verify_determinism_sl_chained(
        engine,
        entries,
    ) == (False, 0)


def test_verify_determinism_compares_zero_logged_order_parameter(
    tmp_path: Path,
) -> None:
    from scpn_phase_orchestrator.upde.engine import UPDEEngine

    engine = UPDEEngine(2, dt=0.01)
    entry = {
        "phases": [0.0, np.pi],
        "omegas": [0.0, 0.0],
        "knm": [[0.0, 0.0], [0.0, 0.0]],
        "alpha": [[0.0, 0.0], [0.0, 0.0]],
        "zeta": 0.0,
        "psi_drive": 0.0,
        "R": 0.0,
    }

    assert ReplayEngine(tmp_path / "unused.jsonl").verify_determinism(
        engine,
        [entry],
    )


def test_signed_integrity_rejects_missing_signature_dict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ok, verified = _verify_rewritten_signed_entry(
        tmp_path,
        monkeypatch,
        lambda entry: entry.pop("_signature", None),
    )

    assert (ok, verified) == (False, 0)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda entry: entry["_signature"].update({"algorithm": "HMAC-BLAKE2"}),
        lambda entry: entry["_signature"].update({"value": "bad"}),
        lambda entry: entry["_signature"].update({"key_id": 42}),
        lambda entry: entry.update({"_audit_schema_version": 2}),
        lambda entry: entry.update({"_audit_sequence": 2}),
        lambda entry: entry.update({"_previous_hash": "f" * 64}),
        lambda entry: entry.update({"_audit_stream_id": ""}),
        lambda entry: entry.update({"_audit_timestamp_unix_ns": -1}),
        lambda entry: entry.update({"_payload_hash": "bad"}),
    ],
)
def test_signed_integrity_rejects_malformed_signature_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    ok, verified = _verify_rewritten_signed_entry(tmp_path, monkeypatch, mutate)

    assert (ok, verified) == (False, 0)
