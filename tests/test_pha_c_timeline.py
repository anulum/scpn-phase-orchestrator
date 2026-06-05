# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — PHA-C timeline tests

"""Behavioural tests for ``upde.pha_c_timeline``."""

from __future__ import annotations

import importlib
import json

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from benchmarks.pha_c_timeline_benchmark import (
    benchmark_pha_c_timeline_polyglot_parity_gate,
)
from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _pha_c_timeline_go,
    _pha_c_timeline_julia,
    _pha_c_timeline_mojo,
    _pha_c_timeline_rust,
    _pha_c_timeline_validation,
)
from scpn_phase_orchestrator.upde import PHACTimelineRecord as ExportedRecord
from scpn_phase_orchestrator.upde.pha_c_timeline import (
    PHA_C_TIMELINE_CLAIM_BOUNDARY,
    PHACTimelineRecord,
    build_pha_c_event_timeline,
    pha_c_event_timeline_to_dict,
)

MODULE_LINKAGE_PATHS = (
    "scpn_phase_orchestrator.upde.pha_c_timeline",
    "scpn_phase_orchestrator.experimental.accelerators.upde._pha_c_timeline_go",
    "scpn_phase_orchestrator.experimental.accelerators.upde._pha_c_timeline_julia",
    "scpn_phase_orchestrator.experimental.accelerators.upde._pha_c_timeline_mojo",
    "scpn_phase_orchestrator.experimental.accelerators.upde._pha_c_timeline_rust",
    "scpn_phase_orchestrator.experimental.accelerators.upde._pha_c_timeline_validation",
)


def _trajectory() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phases = np.array(
        [
            [-0.02, 0.0, 0.02],
            [-0.002, 0.0, 0.002],
            [-0.0015, 0.0, 0.0015],
            [-0.001, 0.0, 0.001],
            [-0.02, 0.0, 0.02],
        ],
        dtype=np.float64,
    )
    positions = np.array(
        [
            [-0.003, 0.0, 0.003],
            [-0.0005, 0.0, 0.0005],
            [-0.0004, 0.0, 0.0004],
            [-0.0003, 0.0, 0.0003],
            [-0.003, 0.0, 0.003],
        ],
        dtype=np.float64,
    )
    times = np.arange(phases.shape[0], dtype=np.float64) * 0.5
    return phases, positions, times


def test_module_linkage_paths_cover_pha_c_timeline_chain() -> None:
    for import_path in MODULE_LINKAGE_PATHS:
        assert importlib.import_module(import_path).__name__ == import_path


def test_timeline_records_lock_loss_reset_and_review_only_hashes() -> None:
    phases, positions, times = _trajectory()

    timeline = build_pha_c_event_timeline(
        phases,
        positions,
        times=times,
        phase_tol_rad=0.01,
        spatial_tol_m=0.002,
        required_consecutive_samples=3,
        tolerance_profile="baseline_1x",
    )
    repeated = build_pha_c_event_timeline(
        phases,
        positions,
        times=times,
        phase_tol_rad=0.01,
        spatial_tol_m=0.002,
        required_consecutive_samples=3,
        tolerance_profile="baseline_1x",
    )

    assert timeline.sample_count == 5
    assert timeline.oscillator_count == 3
    assert timeline.first_lock_observed
    assert timeline.first_lock_index == 3
    assert timeline.first_lock_time == pytest.approx(1.5)
    assert not timeline.final_lock_achieved
    assert timeline.lock_sample_count == 1
    assert timeline.phase_lock_sample_count == 3
    assert timeline.spatial_lock_sample_count == 3
    assert timeline.lock_loss_count == 1
    assert timeline.reset_count == 1
    assert timeline.max_consecutive_lock_samples == 3
    assert timeline.claim_boundary == PHA_C_TIMELINE_CLAIM_BOUNDARY
    assert timeline.execution_disabled
    assert not timeline.actuating
    assert timeline.tolerance_profile_name == "baseline_1x"
    assert timeline.tolerance_profile_multiplier == pytest.approx(1.0)
    assert len(timeline.time_state_sha256) == 64
    assert len(timeline.sample_records_sha256) == 64
    assert len(timeline.transition_table_sha256) == 64
    assert len(timeline.timeline_sha256) == 64
    assert timeline.timeline_sha256 == repeated.timeline_sha256
    assert pha_c_event_timeline_to_dict(timeline) == timeline.to_dict()
    assert ExportedRecord is PHACTimelineRecord


def test_timeline_tolerance_profile_records_boundary() -> None:
    phases, positions, times = _trajectory()

    timeline = build_pha_c_event_timeline(
        phases,
        positions,
        times=times,
        phase_tol_rad=0.01,
        spatial_tol_m=0.002,
        required_consecutive_samples=1,
        tolerance_profile="buffer_3x",
    )

    assert timeline.first_lock_observed
    assert timeline.final_lock_achieved
    assert timeline.phase_tol_rad == pytest.approx(0.03)
    assert timeline.spatial_tol_m == pytest.approx(0.006)
    assert timeline.tolerance_profile_name == "buffer_3x"
    assert timeline.tolerance_profile_multiplier == pytest.approx(3.0)


def test_invalid_timeline_inputs_fail_closed() -> None:
    phases, positions, times = _trajectory()
    invalid_cases = (
        ([], [], None, "two-dimensional"),
        ([[0.0]], [[0.0], [1.0]], None, "same shape"),
        ([0.0], [0.0], None, "two-dimensional"),
        ([[np.nan]], [[0.0]], None, "finite"),
        ([[True, False]], [[0.0, 0.0]], None, "real-valued"),
        ([[0.0 + 1.0j]], [[0.0]], None, "real-valued"),
        (np.array([[0.0]], dtype=object), [[0.0]], None, "finite real-valued"),
        (phases, positions, [0.0, 0.5], "one entry per trajectory sample"),
        (phases, positions, [0.0, 0.5, 1.0, 1.0, 2.0], "strictly increasing"),
        (phases, positions, [0.0, 0.5, np.inf, 1.5, 2.0], "finite"),
    )
    for phase_rows, position_rows, time_rows, match in invalid_cases:
        with pytest.raises(ValueError, match=match):
            build_pha_c_event_timeline(phase_rows, position_rows, times=time_rows)

    with pytest.raises(ValueError, match="phase_tol_rad"):
        build_pha_c_event_timeline(phases, positions, times=times, phase_tol_rad=-0.1)
    with pytest.raises(ValueError, match="required_consecutive_samples"):
        build_pha_c_event_timeline(
            phases,
            positions,
            times=times,
            required_consecutive_samples=0,
        )
    with pytest.raises(ValueError, match="tolerance_profile"):
        build_pha_c_event_timeline(
            phases,
            positions,
            times=times,
            tolerance_profile="unknown",
        )


@st.composite
def _locked_trajectories(
    draw: st.DrawFn,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    steps = draw(st.integers(min_value=3, max_value=6))
    oscillators = draw(st.integers(min_value=2, max_value=6))
    shift = draw(
        st.floats(
            min_value=-np.pi,
            max_value=np.pi,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    phase_rows = draw(
        st.lists(
            st.lists(
                st.floats(
                    min_value=-0.004,
                    max_value=0.004,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=oscillators,
                max_size=oscillators,
            ),
            min_size=steps,
            max_size=steps,
        ),
    )
    position_rows = draw(
        st.lists(
            st.lists(
                st.floats(
                    min_value=-0.0008,
                    max_value=0.0008,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=oscillators,
                max_size=oscillators,
            ),
            min_size=steps,
            max_size=steps,
        ),
    )
    times = np.arange(steps, dtype=np.float64) * 0.25
    return (
        np.array(phase_rows, dtype=np.float64),
        np.array(position_rows, dtype=np.float64),
        times,
        float(shift),
    )


@given(trajectory=_locked_trajectories())
@settings(max_examples=24, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_timeline_invariants_survive_global_phase_shift(
    trajectory: tuple[np.ndarray, np.ndarray, np.ndarray, float],
) -> None:
    phases, positions, times, shift = trajectory
    reference = build_pha_c_event_timeline(
        phases,
        positions,
        times=times,
        reference_phase=0.0,
        phase_tol_rad=0.01,
        spatial_tol_m=0.002,
        required_consecutive_samples=2,
    )
    shifted = build_pha_c_event_timeline(
        phases + shift,
        positions,
        times=times,
        reference_phase=shift,
        phase_tol_rad=0.01,
        spatial_tol_m=0.002,
        required_consecutive_samples=2,
    )

    assert shifted.max_phase_dispersion_rad == pytest.approx(
        reference.max_phase_dispersion_rad,
        abs=1.0e-12,
    )
    assert shifted.min_phase_order_parameter == pytest.approx(
        reference.min_phase_order_parameter,
        abs=1.0e-12,
    )
    assert shifted.max_spatial_dispersion_m == pytest.approx(
        reference.max_spatial_dispersion_m,
        abs=1.0e-12,
    )
    assert shifted.first_lock_index == reference.first_lock_index
    assert shifted.lock_sample_count == reference.lock_sample_count
    assert shifted.lock_loss_count == reference.lock_loss_count
    assert shifted.reset_count == reference.reset_count


def test_polyglot_timeline_adapter_contracts_match_reference() -> None:
    phases, positions, times = _trajectory()
    expected = build_pha_c_event_timeline(
        phases,
        positions,
        times=times,
        phase_tol_rad=0.01,
        spatial_tol_m=0.002,
        required_consecutive_samples=3,
        tolerance_profile="baseline_1x",
    )
    adapters = (
        _pha_c_timeline_rust.build_pha_c_event_timeline_rust,
        _pha_c_timeline_mojo.build_pha_c_event_timeline_mojo,
        _pha_c_timeline_julia.build_pha_c_event_timeline_julia,
        _pha_c_timeline_go.build_pha_c_event_timeline_go,
    )
    for adapter in adapters:
        got = adapter(
            phases,
            positions,
            times=times,
            phase_tol_rad=0.01,
            spatial_tol_m=0.002,
            required_consecutive_samples=3,
            tolerance_profile="baseline_1x",
        )
        assert (
            _pha_c_timeline_validation.validate_pha_c_event_timeline(got, expected)
            is got
        )


def test_pha_c_timeline_benchmark_gate_accepts_declared_backends() -> None:
    result = benchmark_pha_c_timeline_polyglot_parity_gate(n=8, calls=1)

    assert result["suite"] == "pha_c_timeline_polyglot_parity_gate"
    assert result["backend_count"] == 5
    assert result["parity_pass_count"] == 5
    assert result["source_contract_backend_count"] == 4
    assert result["native_kernel_count"] == 0
    assert result["polyglot_claim_boundary"] == "source_contract_not_native_kernel"
    assert result["acceptance_passed"] == 1
    assert result["first_lock_observed"] == 1
    assert result["first_lock_index"] == 3
    assert result["lock_loss_count"] == 1
    assert result["reset_count"] == 1
    assert result["non_actuating"] == 1
    assert result["execution_disabled"] == 1
    assert result["benchmark_evidence_kind"] == "local_regression_non_isolated"
    backend_records = json.loads(str(result["backend_records_json"]))
    assert {record["execution_mode"] for record in backend_records} == {
        "python_reference",
        "source_contract_reference_validation",
    }
    assert sum(int(record["native_kernel_present"]) for record in backend_records) == 0
