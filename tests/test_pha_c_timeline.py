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
import runpy
import sys
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from benchmarks import pha_c_timeline_benchmark
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
from scpn_phase_orchestrator.upde import verify_pha_c_event_timeline as exported_verify
from scpn_phase_orchestrator.upde.pha_c_timeline import (
    PHA_C_TIMELINE_CLAIM_BOUNDARY,
    PHA_C_TIMELINE_MARGIN_REPLAY_TOLERANCE,
    PHACTimelineRecord,
    build_pha_c_event_timeline,
    pha_c_event_timeline_to_dict,
    verify_pha_c_event_timeline,
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
    assert exported_verify is verify_pha_c_event_timeline
    assert verify_pha_c_event_timeline(timeline) is timeline


def test_timeline_replay_verifier_rejects_tampered_evidence() -> None:
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

    with pytest.raises(ValueError, match="timeline_sha256"):
        verify_pha_c_event_timeline(replace(timeline, timeline_sha256="0" * 64))
    with pytest.raises(ValueError, match="sample_count"):
        verify_pha_c_event_timeline(replace(timeline, lock_sample_count=99))
    with pytest.raises(ValueError, match="claim_boundary"):
        verify_pha_c_event_timeline(replace(timeline, claim_boundary="actuating"))


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
    invalid_cases: tuple[tuple[object, object, object | None, str], ...] = (
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
    assert result["hash_replay_validated"] == 1
    assert result["first_lock_observed"] == 1
    assert result["first_lock_index"] == 3
    assert result["lock_loss_count"] == 1
    assert result["reset_count"] == 1
    assert result["phase_margin_loss_observed"] == 1
    assert result["spatial_margin_loss_observed"] == 1
    assert result["phase_margin_equation_validated"] == 1
    assert result["spatial_margin_equation_validated"] == 1
    assert result["signed_margin_equations_validated"] == 1
    assert result["margin_replay_tolerance"] == (PHA_C_TIMELINE_MARGIN_REPLAY_TOLERANCE)
    assert result["non_actuating"] == 1
    assert result["execution_disabled"] == 1
    assert result["benchmark_evidence_kind"] == "local_regression_non_isolated"
    backend_records = json.loads(str(result["backend_records_json"]))
    assert {record["execution_mode"] for record in backend_records} == {
        "python_reference",
        "source_contract_reference_validation",
    }
    assert sum(int(record["native_kernel_present"]) for record in backend_records) == 0
    assert all(int(record["hash_replay_validated"]) == 1 for record in backend_records)
    assert all(
        int(record["signed_margin_equations_validated"]) == 1
        for record in backend_records
    )


def test_timeline_min_signed_margins_are_hash_replayed() -> None:
    from dataclasses import replace

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
    timeline = build_pha_c_event_timeline(
        phases,
        positions,
        times=np.arange(phases.shape[0], dtype=np.float64) * 0.5,
        phase_tol_rad=0.01,
        spatial_tol_m=0.002,
        required_consecutive_samples=3,
    )

    assert timeline.min_phase_margin_rad == pytest.approx(
        timeline.phase_tol_rad - timeline.max_phase_dispersion_rad,
        abs=PHA_C_TIMELINE_MARGIN_REPLAY_TOLERANCE,
    )
    assert timeline.min_spatial_margin_m == pytest.approx(
        timeline.spatial_tol_m - timeline.max_spatial_dispersion_m,
        abs=PHA_C_TIMELINE_MARGIN_REPLAY_TOLERANCE,
    )
    assert timeline.min_phase_margin_rad < 0.0
    assert timeline.min_spatial_margin_m < 0.0
    assert verify_pha_c_event_timeline(timeline) is timeline

    forged = replace(timeline, min_spatial_margin_m=0.0)
    with pytest.raises(ValueError, match="min_spatial_margin_m"):
        verify_pha_c_event_timeline(forged)

    forged_phase = replace(
        timeline,
        min_phase_margin_rad=timeline.min_phase_margin_rad
        + 100.0 * PHA_C_TIMELINE_MARGIN_REPLAY_TOLERANCE,
    )
    with pytest.raises(ValueError, match="min_phase_margin_rad"):
        verify_pha_c_event_timeline(forged_phase)

    forged_spatial = replace(
        timeline,
        min_spatial_margin_m=timeline.min_spatial_margin_m
        + 100.0 * PHA_C_TIMELINE_MARGIN_REPLAY_TOLERANCE,
    )
    with pytest.raises(ValueError, match="min_spatial_margin_m"):
        verify_pha_c_event_timeline(forged_spatial)


def _valid_timeline() -> PHACTimelineRecord:
    phases, positions, times = _trajectory()
    return build_pha_c_event_timeline(
        phases,
        positions,
        times=times,
        phase_tol_rad=0.01,
        spatial_tol_m=0.002,
        required_consecutive_samples=3,
        tolerance_profile="baseline_1x",
    )


@pytest.mark.parametrize("tolerance", [True, -1.0e-12, float("nan"), float("inf")])
def test_timeline_validation_rejects_invalid_tolerance(tolerance: float) -> None:
    timeline = _valid_timeline()
    with pytest.raises(ValueError, match="tolerance"):
        _pha_c_timeline_validation.validate_pha_c_event_timeline(
            timeline,
            timeline,
            tolerance=tolerance,
        )


def test_timeline_validation_rejects_numeric_divergence() -> None:
    phases, positions, times = _trajectory()
    timeline = _valid_timeline()
    expected = build_pha_c_event_timeline(
        phases,
        positions,
        times=times + 1.0,
        phase_tol_rad=0.01,
        spatial_tol_m=0.002,
        required_consecutive_samples=3,
        tolerance_profile="baseline_1x",
    )

    with pytest.raises(ValueError, match="start_time"):
        _pha_c_timeline_validation.validate_pha_c_event_timeline(
            timeline,
            expected,
            tolerance=0.0,
        )


def test_timeline_validation_rejects_numeric_string_record_field() -> None:
    timeline = _valid_timeline()
    forged = replace(
        timeline,
        min_phase_margin_rad=cast(float, str(timeline.min_phase_margin_rad)),
    )

    with pytest.raises(ValueError, match="min_phase_margin_rad"):
        _pha_c_timeline_validation.validate_pha_c_event_timeline(
            forged,
            timeline,
        )


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        (
            {"max_phase_dispersion_rad": float("nan")},
            "max_phase_dispersion_rad.*finite",
        ),
        ({"sample_count": "5"}, "sample_count.*integer"),
        ({"first_lock_observed": np.bool_(True)}, "first_lock_observed.*bool"),
        ({"claim_boundary": b"claim"}, "claim_boundary.*string"),
    ],
)
def test_timeline_validation_rejects_strict_record_field_type_drift(
    changes: dict[str, object],
    match: str,
) -> None:
    timeline = _valid_timeline()

    with pytest.raises(ValueError, match=match):
        _pha_c_timeline_validation.validate_pha_c_event_timeline(
            replace(timeline, **changes),
            timeline,
        )


def test_timeline_benchmark_error_rejects_numeric_string_record_field() -> None:
    from benchmarks.pha_c_timeline_benchmark import _record_max_abs_error

    timeline = _valid_timeline()
    forged = replace(
        timeline,
        min_phase_margin_rad=cast(float, str(timeline.min_phase_margin_rad)),
    )

    with pytest.raises(ValueError, match="min_phase_margin_rad"):
        _record_max_abs_error(forged, timeline)


def test_timeline_validation_rejects_integer_divergence() -> None:
    timeline = _valid_timeline()
    forged = replace(timeline, reset_count=timeline.reset_count + 1)

    with pytest.raises(ValueError, match="reset_count"):
        _pha_c_timeline_validation.validate_pha_c_event_timeline(
            forged,
            timeline,
        )


def test_timeline_validation_rejects_boolean_divergence() -> None:
    timeline = _valid_timeline()
    forged = replace(timeline, first_lock_observed=not timeline.first_lock_observed)

    with pytest.raises(ValueError, match="first_lock_observed"):
        _pha_c_timeline_validation.validate_pha_c_event_timeline(
            forged,
            timeline,
        )


def test_timeline_validation_rejects_discrete_divergence() -> None:
    phases, positions, times = _trajectory()
    timeline = _valid_timeline()
    expected = build_pha_c_event_timeline(
        phases,
        -positions,
        times=times,
        phase_tol_rad=0.01,
        spatial_tol_m=0.002,
        required_consecutive_samples=3,
        tolerance_profile="baseline_1x",
    )

    with pytest.raises(ValueError, match="sample_records_sha256"):
        _pha_c_timeline_validation.validate_pha_c_event_timeline(timeline, expected)


def test_verify_rejects_a_non_timeline() -> None:
    with pytest.raises(ValueError, match="must be a PHACTimelineRecord"):
        verify_pha_c_event_timeline("not a timeline")


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"time_state_sha256": "nope"}, "SHA-256 hex digest"),
        ({"evidence_kind": "bogus"}, "evidence_kind must be"),
        ({"execution_disabled": "yes"}, "must be a boolean"),
        ({"execution_disabled": False}, "execution_disabled must be true"),
        ({"actuating": True}, "actuating must be false"),
        ({"sample_count": 2.5}, "must be an integer"),
        ({"sample_count": 0}, "must be at least"),
        ({"lock_loss_count": 5}, "cannot exceed the transition count"),
        (
            {"max_consecutive_lock_samples": 0, "final_lock_achieved": True},
            "final_lock_achieved requires the consecutive threshold",
        ),
        ({"start_time": True}, "finite real scalar"),
        ({"start_time": "x"}, "finite real scalar"),
        ({"start_time": float("inf")}, "must be finite"),
        ({"duration_s": -1.0}, "must be non-negative"),
        ({"end_time": -1.0}, "end_time must be greater than or equal"),
        ({"duration_s": 10.0}, "duration_s must equal"),
        ({"first_lock_index": 999}, "first_lock_index must refer to an observed"),
        (
            {
                "first_lock_observed": True,
                "first_lock_index": 2,
                "first_lock_time": 999.0,
            },
            "first_lock_time must be inside",
        ),
        (
            {"first_lock_observed": False, "first_lock_index": 3},
            "first_lock_index must be -1 when no lock",
        ),
        (
            {
                "first_lock_observed": False,
                "first_lock_index": -1,
                "first_lock_time": 5.0,
            },
            "first_lock_time must be 0.0 when no lock",
        ),
        ({"min_phase_order_parameter": 1.5}, "must be inside"),
        ({"tolerance_profile_multiplier": 0.0}, "multiplier must be positive"),
        ({"tolerance_profile_name": ""}, "must be a non-empty string"),
    ],
)
def test_verify_rejects_tampered_timeline(
    changes: dict[str, object],
    match: str,
) -> None:
    record = replace(_valid_timeline(), **changes)
    with pytest.raises(ValueError, match=match):
        verify_pha_c_event_timeline(record)


def test_build_uses_default_integer_times_when_none() -> None:
    phases, positions, _ = _trajectory()
    timeline = build_pha_c_event_timeline(
        phases,
        positions,
        times=None,
        phase_tol_rad=0.01,
        spatial_tol_m=0.002,
        required_consecutive_samples=3,
        tolerance_profile="baseline_1x",
    )
    assert timeline.sample_count == phases.shape[0]


@pytest.mark.parametrize(
    ("times", "match"),
    [
        (np.array([0, 1, 2, 3, 4], dtype=object), "finite real-valued vector"),
        (np.array([False, True, False, True, False]), "boolean or complex"),
        (np.zeros((5, 1)), "one-dimensional"),
        (np.array(["a", "b", "c", "d", "e"]), "times must be numeric"),
    ],
)
def test_build_rejects_invalid_times(times: object, match: str) -> None:
    phases, positions, _ = _trajectory()
    with pytest.raises(ValueError, match=match):
        build_pha_c_event_timeline(
            phases,
            positions,
            times=times,
            phase_tol_rad=0.01,
            spatial_tol_m=0.002,
            required_consecutive_samples=3,
            tolerance_profile="baseline_1x",
        )


def test_build_rejects_an_empty_phase_matrix() -> None:
    with pytest.raises(ValueError, match="at least one time sample"):
        build_pha_c_event_timeline(
            np.zeros((0, 3)),
            np.zeros((0, 3)),
            phase_tol_rad=0.01,
            spatial_tol_m=0.002,
            required_consecutive_samples=3,
            tolerance_profile="baseline_1x",
        )


def test_build_rejects_a_non_numeric_phase_matrix() -> None:
    with pytest.raises(ValueError, match="must be numeric"):
        build_pha_c_event_timeline(
            np.array([["a", "b", "c"]] * 5),
            np.zeros((5, 3)),
            phase_tol_rad=0.01,
            spatial_tol_m=0.002,
            required_consecutive_samples=3,
            tolerance_profile="baseline_1x",
        )


def test_verify_accepts_a_never_locked_timeline() -> None:
    samples = 5
    phases = np.tile(np.array([-1.0, 0.0, 1.0]), (samples, 1))
    positions = np.tile(np.array([-0.5, 0.0, 0.5]), (samples, 1))
    times: np.ndarray = np.arange(samples, dtype=np.float64)
    timeline = build_pha_c_event_timeline(
        phases,
        positions,
        times=times,
        phase_tol_rad=0.01,
        spatial_tol_m=0.002,
        required_consecutive_samples=3,
        tolerance_profile="baseline_1x",
    )
    assert not timeline.first_lock_observed
    assert timeline.first_lock_index == -1
    assert timeline.first_lock_time == 0.0
    assert verify_pha_c_event_timeline(timeline) is timeline


def test_pha_c_timeline_benchmark_rejects_invalid_int_controls() -> None:
    with pytest.raises(ValueError, match="n must be an integer"):
        pha_c_timeline_benchmark._validate_int_control(
            True,
            name="n",
            minimum=2,
        )
    with pytest.raises(ValueError, match="calls must be at least 1"):
        pha_c_timeline_benchmark._validate_int_control(
            0,
            name="calls",
            minimum=1,
        )
    with pytest.raises(ValueError, match="flag must be an integer"):
        pha_c_timeline_benchmark._payload_int(True, name="flag")


def test_pha_c_timeline_benchmark_main_writes_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = {"suite": "stub", "acceptance_passed": 1}
    output_path = tmp_path / "timeline.json"

    def fake_gate(*, n: int, calls: int) -> dict[str, object]:
        assert n == 3
        assert calls == 1
        return payload

    monkeypatch.setattr(
        pha_c_timeline_benchmark,
        "benchmark_pha_c_timeline_polyglot_parity_gate",
        fake_gate,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pha-c-timeline",
            "--n",
            "3",
            "--calls",
            "1",
            "--parity-gate",
            "--output",
            str(output_path),
        ],
    )

    assert pha_c_timeline_benchmark._main() == 0
    assert json.loads(output_path.read_text(encoding="utf-8")) == payload
    assert json.loads(capsys.readouterr().out) == payload


def test_pha_c_timeline_benchmark_main_fails_parity_gate(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = {"suite": "stub", "acceptance_passed": 0}

    def fake_gate(*, n: int, calls: int) -> dict[str, object]:
        assert n == 8
        assert calls == 3
        return payload

    monkeypatch.setattr(
        pha_c_timeline_benchmark,
        "benchmark_pha_c_timeline_polyglot_parity_gate",
        fake_gate,
    )
    monkeypatch.setattr(sys, "argv", ["pha-c-timeline", "--parity-gate"])

    assert pha_c_timeline_benchmark._main() == 1
    assert json.loads(capsys.readouterr().out) == payload


def test_pha_c_timeline_benchmark_module_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["pha-c-timeline", "--n", "2", "--calls", "1"])

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(
            str(Path(pha_c_timeline_benchmark.__file__).resolve()),
            run_name="__main__",
        )

    assert exc_info.value.code == 0
    assert json.loads(capsys.readouterr().out)["suite"] == (
        "pha_c_timeline_polyglot_parity_gate"
    )
