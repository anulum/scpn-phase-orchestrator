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

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

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
from scpn_phase_orchestrator.upde.pha_c_handoff import (
    PHA_C_HANDOFF_CLAIM_BOUNDARY,
    PHACHandoffRecord,
    build_pha_c_handoff_record,
    pha_c_handoff_record_to_dict,
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
    invalid_cases = (
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
    assert result["acceptance_passed"] == 1
    assert result["non_actuating"] == 1
    assert result["execution_disabled"] == 1
    assert result["tolerance_profile_name"] == "buffer_3x"
    assert result["tolerance_profile_multiplier"] == 3.0
    assert result["benchmark_evidence_kind"] == "local_regression_non_isolated"
