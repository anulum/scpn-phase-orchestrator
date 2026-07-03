# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — PHA-C handoff tests

"""Behavioural tests for ``upde.pha_c_handoff``."""

from __future__ import annotations

import importlib
import json
import runpy
import sys
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from benchmarks import pha_c_handoff_benchmark
from benchmarks.pha_c_handoff_benchmark import (
    benchmark_pha_c_handoff_polyglot_parity_gate,
)
from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _pha_c_handoff_go,
    _pha_c_handoff_julia,
    _pha_c_handoff_mojo,
    _pha_c_handoff_rust,
    _pha_c_handoff_validation,
)
from scpn_phase_orchestrator.upde import PHACHandoffRecord as ExportedRecord
from scpn_phase_orchestrator.upde import verify_pha_c_handoff_record as exported_verify
from scpn_phase_orchestrator.upde.pha_c_handoff import (
    PHA_C_HANDOFF_CLAIM_BOUNDARY,
    PHA_C_HANDOFF_MARGIN_REPLAY_TOLERANCE,
    PHACHandoffRecord,
    build_pha_c_handoff_record,
    pha_c_handoff_record_to_dict,
    verify_pha_c_handoff_record,
)

MODULE_LINKAGE_PATHS = (
    "scpn_phase_orchestrator.upde.pha_c_handoff",
    "scpn_phase_orchestrator.experimental.accelerators.upde._pha_c_handoff_go",
    "scpn_phase_orchestrator.experimental.accelerators.upde._pha_c_handoff_julia",
    "scpn_phase_orchestrator.experimental.accelerators.upde._pha_c_handoff_mojo",
    "scpn_phase_orchestrator.experimental.accelerators.upde._pha_c_handoff_rust",
    "scpn_phase_orchestrator.experimental.accelerators.upde._pha_c_handoff_validation",
)


def test_module_linkage_paths_cover_pha_c_handoff_chain() -> None:
    for import_path in MODULE_LINKAGE_PATHS:
        assert importlib.import_module(import_path).__name__ == import_path


def test_handoff_record_is_hash_stable_and_review_only() -> None:
    phases = np.array([0.0, 0.003, -0.004], dtype=np.float64)
    positions = np.array([0.0, 0.0005, -0.0008], dtype=np.float64)

    record = build_pha_c_handoff_record(
        phases,
        positions,
        t=4.0,
        phase_tol_rad=0.01,
        spatial_tol_m=0.002,
        required_consecutive_samples=3,
        prior_consecutive_lock_samples=2,
        tolerance_profile="baseline_1x",
    )
    repeated = build_pha_c_handoff_record(
        phases,
        positions,
        t=4.0,
        phase_tol_rad=0.01,
        spatial_tol_m=0.002,
        required_consecutive_samples=3,
        prior_consecutive_lock_samples=2,
        tolerance_profile="baseline_1x",
    )

    assert record.lock_achieved
    assert record.consecutive_lock_samples == 3
    assert record.phase_locked
    assert record.spatial_locked
    assert record.claim_boundary == PHA_C_HANDOFF_CLAIM_BOUNDARY
    assert record.execution_disabled
    assert not record.actuating
    assert record.tolerance_profile_name == "baseline_1x"
    assert record.tolerance_profile_multiplier == pytest.approx(1.0)
    assert len(record.phase_state_sha256) == 64
    assert len(record.position_state_sha256) == 64
    assert len(record.merge_report_sha256) == 64
    assert len(record.source_chain_sha256) == 64
    assert len(record.record_sha256) == 64
    assert record.record_sha256 == repeated.record_sha256
    assert pha_c_handoff_record_to_dict(record) == record.to_dict()
    assert ExportedRecord is PHACHandoffRecord
    assert exported_verify is verify_pha_c_handoff_record
    assert verify_pha_c_handoff_record(record) is record


def test_handoff_replay_verifier_rejects_tampered_evidence() -> None:
    record = build_pha_c_handoff_record(
        np.array([0.0, 0.003, -0.004], dtype=np.float64),
        np.array([0.0, 0.0005, -0.0008], dtype=np.float64),
        phase_tol_rad=0.01,
        spatial_tol_m=0.002,
        required_consecutive_samples=3,
        prior_consecutive_lock_samples=2,
        tolerance_profile="baseline_1x",
    )

    with pytest.raises(ValueError, match="record_sha256"):
        verify_pha_c_handoff_record(replace(record, record_sha256="0" * 64))
    with pytest.raises(ValueError, match="claim_boundary"):
        verify_pha_c_handoff_record(replace(record, claim_boundary="actuating"))
    with pytest.raises(ValueError, match="phase_order_parameter"):
        verify_pha_c_handoff_record(
            replace(
                record,
                phase_order_parameter=1.2,
                record_sha256=record.record_sha256,
            ),
        )


def test_handoff_replay_verifier_rejects_forged_signed_margins() -> None:
    record = build_pha_c_handoff_record(
        np.array([0.0, 0.003, -0.004], dtype=np.float64),
        np.array([0.0, 0.0005, -0.0008], dtype=np.float64),
        phase_tol_rad=0.01,
        spatial_tol_m=0.002,
        required_consecutive_samples=3,
        prior_consecutive_lock_samples=2,
    )

    assert record.phase_margin_rad == pytest.approx(
        record.phase_tol_rad - record.phase_dispersion_rad,
        abs=PHA_C_HANDOFF_MARGIN_REPLAY_TOLERANCE,
    )
    assert record.spatial_margin_m == pytest.approx(
        record.spatial_tol_m - record.spatial_dispersion_m,
        abs=PHA_C_HANDOFF_MARGIN_REPLAY_TOLERANCE,
    )
    with pytest.raises(ValueError, match="phase_margin_rad"):
        verify_pha_c_handoff_record(
            replace(
                record,
                phase_margin_rad=(
                    record.phase_margin_rad
                    + 100.0 * PHA_C_HANDOFF_MARGIN_REPLAY_TOLERANCE
                ),
                record_sha256=record.record_sha256,
            ),
        )
    with pytest.raises(ValueError, match="spatial_margin_m"):
        verify_pha_c_handoff_record(
            replace(
                record,
                spatial_margin_m=(
                    record.spatial_margin_m
                    + 100.0 * PHA_C_HANDOFF_MARGIN_REPLAY_TOLERANCE
                ),
                record_sha256=record.record_sha256,
            ),
        )


def test_handoff_reports_phase_and_spatial_failures_without_actuation() -> None:
    record = build_pha_c_handoff_record(
        np.array([0.0, 0.03], dtype=np.float64),
        np.array([0.0, 0.003], dtype=np.float64),
        phase_tol_rad=0.01,
        spatial_tol_m=0.002,
        required_consecutive_samples=1,
        prior_consecutive_lock_samples=9,
    )

    assert not record.phase_locked
    assert not record.spatial_locked
    assert not record.lock_achieved
    assert record.consecutive_lock_samples == 0
    assert record.execution_disabled
    assert not record.actuating
    assert record.tolerance_profile_name == "explicit"


def test_handoff_tolerance_profile_records_review_boundary() -> None:
    record = build_pha_c_handoff_record(
        np.array([0.0, 0.024], dtype=np.float64),
        np.array([0.0, 0.0045], dtype=np.float64),
        phase_tol_rad=0.01,
        spatial_tol_m=0.002,
        required_consecutive_samples=1,
        tolerance_profile="buffer_3x",
    )

    assert record.lock_achieved
    assert record.phase_tol_rad == pytest.approx(0.03)
    assert record.spatial_tol_m == pytest.approx(0.006)
    assert record.tolerance_profile_name == "buffer_3x"
    assert record.tolerance_profile_multiplier == pytest.approx(3.0)


def test_invalid_handoff_inputs_fail_closed() -> None:
    invalid_cases: tuple[tuple[object, object, str], ...] = (
        ([], [], "phases"),
        ([0.0], [0.0, 1.0], "same one-dimensional shape"),
        ([[0.0]], [[0.0]], "one-dimensional"),
        ([np.nan], [0.0], "finite"),
        ([True, False], [0.0, 0.0], "real-valued"),
        ([0.0 + 1.0j], [0.0], "real-valued"),
        (np.array([0.0], dtype=object), [0.0], "finite real-valued"),
    )
    for phases, positions, match in invalid_cases:
        with pytest.raises(ValueError, match=match):
            build_pha_c_handoff_record(phases, positions)

    with pytest.raises(ValueError, match="phase_tol_rad"):
        build_pha_c_handoff_record([0.0], [0.0], phase_tol_rad=-0.1)
    with pytest.raises(ValueError, match="spatial_tol_m"):
        build_pha_c_handoff_record([0.0], [0.0], spatial_tol_m=-0.1)
    with pytest.raises(ValueError, match="required_consecutive_samples"):
        build_pha_c_handoff_record([0.0], [0.0], required_consecutive_samples=0)
    with pytest.raises(ValueError, match="prior_consecutive_lock_samples"):
        build_pha_c_handoff_record(
            [0.0],
            [0.0],
            prior_consecutive_lock_samples=-1,
        )
    with pytest.raises(ValueError, match="tolerance_profile"):
        build_pha_c_handoff_record([0.0], [0.0], tolerance_profile="unknown")


@st.composite
def _locked_phase_position_vectors(
    draw: st.DrawFn,
) -> tuple[np.ndarray, np.ndarray, float]:
    n = draw(st.integers(min_value=2, max_value=8))
    shift = draw(
        st.floats(
            min_value=-np.pi,
            max_value=np.pi,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    phase_values = draw(
        st.lists(
            st.floats(
                min_value=-0.01,
                max_value=0.01,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=n,
            max_size=n,
        )
    )
    position_values = draw(
        st.lists(
            st.floats(
                min_value=-0.002,
                max_value=0.002,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=n,
            max_size=n,
        )
    )
    return (
        np.array(phase_values, dtype=np.float64),
        np.array(position_values, dtype=np.float64),
        float(shift),
    )


@given(vectors=_locked_phase_position_vectors())
@settings(max_examples=32, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_phase_handoff_invariants_survive_global_phase_shift(
    vectors: tuple[np.ndarray, np.ndarray, float],
) -> None:
    phases, positions, shift = vectors
    reference = build_pha_c_handoff_record(
        phases,
        positions,
        reference_phase=0.0,
        phase_tol_rad=0.02,
        spatial_tol_m=0.003,
        required_consecutive_samples=1,
    )
    shifted = build_pha_c_handoff_record(
        phases + shift,
        positions,
        reference_phase=shift,
        phase_tol_rad=0.02,
        spatial_tol_m=0.003,
        required_consecutive_samples=1,
    )

    assert shifted.phase_dispersion_rad == pytest.approx(
        reference.phase_dispersion_rad,
        abs=1.0e-12,
    )
    assert shifted.phase_order_parameter == pytest.approx(
        reference.phase_order_parameter,
        abs=1.0e-12,
    )
    assert shifted.spatial_dispersion_m == pytest.approx(
        reference.spatial_dispersion_m,
        abs=1.0e-12,
    )
    assert shifted.lock_achieved is reference.lock_achieved


def test_polyglot_handoff_adapter_contracts_match_reference() -> None:
    phases = np.array([0.0, 0.003, -0.004], dtype=np.float64)
    positions = np.array([0.0, 0.0005, -0.0008], dtype=np.float64)
    expected = build_pha_c_handoff_record(
        phases,
        positions,
        t=3.0,
        phase_tol_rad=0.01,
        spatial_tol_m=0.002,
        required_consecutive_samples=2,
        prior_consecutive_lock_samples=1,
        tolerance_profile="buffer_3x",
    )
    adapters = (
        _pha_c_handoff_rust.build_pha_c_handoff_record_rust,
        _pha_c_handoff_mojo.build_pha_c_handoff_record_mojo,
        _pha_c_handoff_julia.build_pha_c_handoff_record_julia,
        _pha_c_handoff_go.build_pha_c_handoff_record_go,
    )
    for adapter in adapters:
        got = adapter(
            phases,
            positions,
            t=3.0,
            phase_tol_rad=0.01,
            spatial_tol_m=0.002,
            required_consecutive_samples=2,
            prior_consecutive_lock_samples=1,
            tolerance_profile="buffer_3x",
        )
        assert (
            _pha_c_handoff_validation.validate_pha_c_handoff_record(got, expected)
            is got
        )


def test_pha_c_handoff_benchmark_gate_accepts_declared_backends() -> None:
    result = benchmark_pha_c_handoff_polyglot_parity_gate(n=8, calls=1)

    assert result["suite"] == "pha_c_handoff_polyglot_parity_gate"
    assert result["backend_count"] == 5
    assert result["parity_pass_count"] == 5
    assert result["source_contract_backend_count"] == 4
    assert result["native_kernel_count"] == 0
    assert result["polyglot_claim_boundary"] == "source_contract_not_native_kernel"
    assert result["phase_margin_equation_validated"] == 1
    assert result["spatial_margin_equation_validated"] == 1
    assert result["signed_margin_equations_validated"] == 1
    assert result["margin_replay_tolerance"] == PHA_C_HANDOFF_MARGIN_REPLAY_TOLERANCE
    for record in json.loads(str(result["backend_records_json"])):
        assert int(record["signed_margin_equations_validated"]) == 1
    assert result["acceptance_passed"] == 1
    assert result["hash_replay_validated"] == 1
    assert result["phase_margin_positive"] == 1
    assert result["spatial_margin_positive"] == 1
    assert result["non_actuating"] == 1
    assert result["execution_disabled"] == 1
    assert result["tolerance_profile_name"] == "buffer_3x"
    assert result["tolerance_profile_multiplier"] == 3.0
    assert result["benchmark_evidence_kind"] == "local_regression_non_isolated"
    backend_records = json.loads(str(result["backend_records_json"]))
    assert {record["execution_mode"] for record in backend_records} == {
        "python_reference",
        "source_contract_reference_validation",
    }
    assert sum(int(record["native_kernel_present"]) for record in backend_records) == 0
    assert all(int(record["hash_replay_validated"]) == 1 for record in backend_records)


def test_handoff_signed_margins_are_hash_replayed() -> None:
    from dataclasses import replace

    record = build_pha_c_handoff_record(
        np.array([0.0, 0.003, -0.004], dtype=np.float64),
        np.array([0.0, 0.0005, -0.0008], dtype=np.float64),
        phase_tol_rad=0.01,
        spatial_tol_m=0.002,
        required_consecutive_samples=1,
    )

    assert record.phase_margin_rad == pytest.approx(
        record.phase_tol_rad - record.phase_dispersion_rad,
        abs=1.0e-12,
    )
    assert record.spatial_margin_m == pytest.approx(
        record.spatial_tol_m - record.spatial_dispersion_m,
        abs=1.0e-12,
    )
    assert record.phase_margin_rad >= 0.0
    assert record.spatial_margin_m >= 0.0
    assert record.to_dict()["phase_margin_rad"] == pytest.approx(
        record.phase_margin_rad,
        abs=1.0e-12,
    )
    assert verify_pha_c_handoff_record(record) is record

    forged = replace(record, phase_margin_rad=record.phase_margin_rad + 1.0e-3)
    with pytest.raises(ValueError, match="phase_margin_rad"):
        verify_pha_c_handoff_record(forged)


def test_handoff_failed_lock_exposes_negative_margins() -> None:
    record = build_pha_c_handoff_record(
        np.array([0.0, 0.02], dtype=np.float64),
        np.array([0.0, 0.003], dtype=np.float64),
        phase_tol_rad=0.01,
        spatial_tol_m=0.002,
        required_consecutive_samples=1,
    )

    assert not record.phase_locked
    assert not record.spatial_locked
    assert record.phase_margin_rad < 0.0
    assert record.spatial_margin_m < 0.0
    assert record.consecutive_lock_samples == 0


def _valid_handoff_record() -> PHACHandoffRecord:
    return build_pha_c_handoff_record(
        np.array([0.0, 0.003, -0.004], dtype=np.float64),
        np.array([0.0, 0.0005, -0.0008], dtype=np.float64),
        t=4.0,
        phase_tol_rad=0.01,
        spatial_tol_m=0.002,
        required_consecutive_samples=3,
        prior_consecutive_lock_samples=2,
        tolerance_profile="baseline_1x",
    )


@pytest.mark.parametrize("tolerance", [True, -1.0e-12, float("nan"), float("inf")])
def test_handoff_validation_rejects_invalid_tolerance(tolerance: float) -> None:
    record = _valid_handoff_record()
    with pytest.raises(ValueError, match="tolerance"):
        _pha_c_handoff_validation.validate_pha_c_handoff_record(
            record,
            record,
            tolerance=tolerance,
        )


def test_handoff_validation_rejects_numeric_divergence() -> None:
    record = _valid_handoff_record()
    expected = build_pha_c_handoff_record(
        np.array([0.0, 0.003, -0.004], dtype=np.float64),
        np.array([0.0, 0.0005, -0.0008], dtype=np.float64),
        t=4.1,
        phase_tol_rad=0.01,
        spatial_tol_m=0.002,
        required_consecutive_samples=3,
        prior_consecutive_lock_samples=2,
        tolerance_profile="baseline_1x",
    )

    with pytest.raises(ValueError, match="'t'"):
        _pha_c_handoff_validation.validate_pha_c_handoff_record(
            record,
            expected,
            tolerance=0.0,
        )


def test_handoff_validation_rejects_numeric_string_record_field() -> None:
    record = _valid_handoff_record()
    forged = replace(
        record,
        phase_margin_rad=cast(float, str(record.phase_margin_rad)),
    )

    with pytest.raises(ValueError, match="phase_margin_rad"):
        _pha_c_handoff_validation.validate_pha_c_handoff_record(
            forged,
            record,
        )


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"phase_dispersion_rad": float("nan")}, "phase_dispersion_rad.*finite"),
        ({"oscillator_count": "3"}, "oscillator_count.*integer"),
        ({"phase_locked": np.bool_(True)}, "phase_locked.*bool"),
        ({"claim_boundary": b"claim"}, "claim_boundary.*string"),
    ],
)
def test_handoff_validation_rejects_strict_record_field_type_drift(
    changes: dict[str, object],
    match: str,
) -> None:
    record = _valid_handoff_record()

    with pytest.raises(ValueError, match=match):
        _pha_c_handoff_validation.validate_pha_c_handoff_record(
            replace(record, **changes),
            record,
        )


def test_handoff_benchmark_error_rejects_numeric_string_record_field() -> None:
    from benchmarks.pha_c_handoff_benchmark import _record_max_abs_error

    record = _valid_handoff_record()
    forged = replace(
        record,
        phase_margin_rad=cast(float, str(record.phase_margin_rad)),
    )

    with pytest.raises(ValueError, match="phase_margin_rad"):
        _record_max_abs_error(forged, record)


def test_handoff_validation_rejects_integer_divergence() -> None:
    record = _valid_handoff_record()
    forged = replace(
        record,
        consecutive_lock_samples=record.consecutive_lock_samples + 1,
    )

    with pytest.raises(ValueError, match="consecutive_lock_samples"):
        _pha_c_handoff_validation.validate_pha_c_handoff_record(
            forged,
            record,
        )


def test_handoff_validation_rejects_boolean_divergence() -> None:
    record = _valid_handoff_record()
    forged = replace(record, phase_locked=not record.phase_locked)

    with pytest.raises(ValueError, match="phase_locked"):
        _pha_c_handoff_validation.validate_pha_c_handoff_record(
            forged,
            record,
        )


def test_handoff_validation_rejects_discrete_divergence() -> None:
    record = _valid_handoff_record()
    expected = build_pha_c_handoff_record(
        np.array([0.0, 0.003, -0.004], dtype=np.float64),
        np.array([0.0, -0.0005, 0.0008], dtype=np.float64),
        t=4.0,
        phase_tol_rad=0.01,
        spatial_tol_m=0.002,
        required_consecutive_samples=3,
        prior_consecutive_lock_samples=2,
        tolerance_profile="baseline_1x",
    )

    with pytest.raises(ValueError, match="position_state_sha256"):
        _pha_c_handoff_validation.validate_pha_c_handoff_record(record, expected)


def test_verify_rejects_a_non_record() -> None:
    with pytest.raises(ValueError, match="must be a PHACHandoffRecord"):
        verify_pha_c_handoff_record("not a record")


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"phase_state_sha256": "not-a-digest"}, "SHA-256 hex digest"),
        ({"evidence_kind": "bogus"}, "evidence_kind must be"),
        ({"execution_disabled": "yes"}, "must be a boolean"),
        ({"execution_disabled": False}, "execution_disabled must be true"),
        ({"actuating": True}, "actuating must be false"),
        ({"oscillator_count": 2.5}, "must be an integer"),
        ({"oscillator_count": 0}, "must be at least"),
        ({"phase_locked": False}, "requires phase and spatial locks"),
        ({"consecutive_lock_samples": 1}, "consecutive-sample threshold"),
        (
            {
                "phase_locked": False,
                "lock_achieved": False,
                "consecutive_lock_samples": 2,
            },
            "must reset consecutive_lock_samples",
        ),
        ({"phase_dispersion_rad": -1.0}, "must be non-negative"),
        ({"phase_margin_rad": "x"}, "finite real scalar"),
        ({"phase_margin_rad": True}, "finite real scalar"),
        ({"phase_margin_rad": float("inf")}, "must be finite"),
        (
            {
                "phase_dispersion_rad": 0.1,
                "phase_tol_rad": 0.01,
                "phase_margin_rad": -0.09,
            },
            "phase_locked requires a non-negative phase_margin",
        ),
        (
            {
                "phase_locked": False,
                "lock_achieved": False,
                "consecutive_lock_samples": 0,
            },
            "phase-unlocked records require a negative phase_margin",
        ),
        (
            {
                "spatial_dispersion_m": 0.1,
                "spatial_tol_m": 0.002,
                "spatial_margin_m": -0.098,
            },
            "spatial_locked requires a non-negative spatial_margin",
        ),
        (
            {
                "spatial_locked": False,
                "lock_achieved": False,
                "consecutive_lock_samples": 0,
            },
            "spatial-unlocked records require a negative spatial_margin",
        ),
        ({"tolerance_profile_multiplier": 0.0}, "multiplier must be positive"),
        ({"tolerance_profile_name": ""}, "must be a non-empty string"),
    ],
)
def test_verify_rejects_tampered_handoff_record(
    changes: dict[str, object],
    match: str,
) -> None:
    record = replace(_valid_handoff_record(), **changes)
    with pytest.raises(ValueError, match=match):
        verify_pha_c_handoff_record(record)


def test_build_rejects_non_integer_sample_count() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        build_pha_c_handoff_record(
            np.array([0.0, 0.003, -0.004]),
            np.array([0.0, 0.0005, -0.0008]),
            required_consecutive_samples=2.5,
        )


def test_build_rejects_non_numeric_phase_vector() -> None:
    with pytest.raises(ValueError, match="must be numeric"):
        build_pha_c_handoff_record(
            np.array(["a", "b", "c"]),
            np.array([0.0, 0.0005, -0.0008]),
        )


def test_pha_c_handoff_benchmark_rejects_invalid_int_controls() -> None:
    with pytest.raises(ValueError, match="n must be an integer"):
        pha_c_handoff_benchmark._validate_int_control(
            True,
            name="n",
            minimum=2,
        )
    with pytest.raises(ValueError, match="calls must be at least 1"):
        pha_c_handoff_benchmark._validate_int_control(
            0,
            name="calls",
            minimum=1,
        )
    with pytest.raises(ValueError, match="flag must be an integer"):
        pha_c_handoff_benchmark._payload_int(True, name="flag")


def test_pha_c_handoff_benchmark_main_writes_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = {"suite": "stub", "acceptance_passed": 1}
    output_path = tmp_path / "handoff.json"

    def fake_gate(*, n: int, calls: int) -> dict[str, object]:
        assert n == 3
        assert calls == 1
        return payload

    monkeypatch.setattr(
        pha_c_handoff_benchmark,
        "benchmark_pha_c_handoff_polyglot_parity_gate",
        fake_gate,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pha-c-handoff",
            "--n",
            "3",
            "--calls",
            "1",
            "--parity-gate",
            "--output",
            str(output_path),
        ],
    )

    assert pha_c_handoff_benchmark._main() == 0
    assert json.loads(output_path.read_text(encoding="utf-8")) == payload
    assert json.loads(capsys.readouterr().out) == payload


def test_pha_c_handoff_benchmark_main_fails_parity_gate(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = {"suite": "stub", "acceptance_passed": 0}

    def fake_gate(*, n: int, calls: int) -> dict[str, object]:
        assert n == 8
        assert calls == 3
        return payload

    monkeypatch.setattr(
        pha_c_handoff_benchmark,
        "benchmark_pha_c_handoff_polyglot_parity_gate",
        fake_gate,
    )
    monkeypatch.setattr(sys, "argv", ["pha-c-handoff", "--parity-gate"])

    assert pha_c_handoff_benchmark._main() == 1
    assert json.loads(capsys.readouterr().out) == payload


def test_pha_c_handoff_benchmark_module_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["pha-c-handoff", "--n", "2", "--calls", "1"])

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(
            str(Path(pha_c_handoff_benchmark.__file__).resolve()),
            run_name="__main__",
        )

    assert exc_info.value.code == 0
    assert json.loads(capsys.readouterr().out)["suite"] == (
        "pha_c_handoff_polyglot_parity_gate"
    )
