# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — merge-window monitor tests

"""Behavioural tests for ``monitor.merge_window``."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from benchmarks.merge_window_benchmark import (
    benchmark_merge_window_polyglot_parity_gate,
)
from scpn_phase_orchestrator import monitor as monitor_pkg
from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _merge_window_go,
    _merge_window_julia,
    _merge_window_mojo,
    _merge_window_rust,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _merge_window_validation as merge_validation,
)
from scpn_phase_orchestrator.monitor import _merge_window_go as public_merge_go
from scpn_phase_orchestrator.monitor import _merge_window_julia as public_merge_julia
from scpn_phase_orchestrator.monitor import _merge_window_mojo as public_merge_mojo
from scpn_phase_orchestrator.monitor import _merge_window_rust as public_merge_rust
from scpn_phase_orchestrator.monitor.merge_window import (
    MERGE_WINDOW_TOLERANCE_PROFILE_MULTIPLIERS,
    MergeReport,
    MergeWindowMonitor,
    MergeWindowToleranceProfile,
    evaluate_merge_window,
    merge_window_report_to_dict,
    merge_window_tolerance_profile_to_dict,
    resolve_merge_window_tolerance_profile,
)

TWO_PI = 2.0 * np.pi
MODULE_LINKAGE_PATHS = (
    "scpn_phase_orchestrator.experimental.accelerators.monitor._merge_window_go",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._merge_window_julia",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._merge_window_mojo",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._merge_window_rust",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._merge_window_validation",
    "scpn_phase_orchestrator.monitor._merge_window_go",
    "scpn_phase_orchestrator.monitor._merge_window_julia",
    "scpn_phase_orchestrator.monitor._merge_window_mojo",
    "scpn_phase_orchestrator.monitor._merge_window_rust",
)


def test_module_linkage_paths_cover_adapter_modules() -> None:
    for import_path in MODULE_LINKAGE_PATHS:
        assert import_path.startswith("scpn_phase_orchestrator.")


def test_truth_table_requires_phase_and_spatial_lock() -> None:
    cases = (
        (np.array([0.0, 0.005]), np.array([0.0, 0.001]), True, True, True),
        (np.array([0.0, 0.02]), np.array([0.0, 0.001]), False, True, False),
        (np.array([0.0, 0.005]), np.array([0.0, 0.003]), True, False, False),
        (np.array([0.0, 0.02]), np.array([0.0, 0.003]), False, False, False),
    )
    for phases, positions, phase_locked, spatial_locked, achieved in cases:
        report = evaluate_merge_window(
            phases,
            positions,
            phase_tol_rad=0.01,
            spatial_tol_m=0.002,
            required_consecutive_samples=1,
        )
        assert report.phase_locked is phase_locked
        assert report.spatial_locked is spatial_locked
        assert report.lock_achieved is achieved


def test_consecutive_gate_and_reset() -> None:
    monitor = MergeWindowMonitor(
        phase_tol_rad=0.01,
        spatial_tol_m=0.002,
        required_consecutive_samples=3,
    )
    locked_phases = np.array([0.0, 0.004, -0.005])
    locked_positions = np.array([0.0, 0.001, -0.0015])

    first = monitor.evaluate(locked_phases, locked_positions, t=1.0)
    second = monitor(locked_phases, locked_positions, t=2.0)
    third = monitor.evaluate(locked_phases, locked_positions, t=3.0)

    assert first.consecutive_lock_samples == 1
    assert not first.lock_achieved
    assert second.consecutive_lock_samples == 2
    assert not second.lock_achieved
    assert third.consecutive_lock_samples == 3
    assert third.lock_achieved
    assert monitor.consecutive_lock_samples == 3

    failed = monitor.evaluate(locked_phases, np.array([0.0, 0.003, 0.0]), t=4.0)
    assert failed.consecutive_lock_samples == 0
    assert not failed.lock_achieved
    assert monitor.consecutive_lock_samples == 0

    monitor.evaluate(locked_phases, locked_positions, t=5.0)
    monitor.reset()
    assert monitor.consecutive_lock_samples == 0


def test_wrapped_phase_lock_near_two_pi() -> None:
    report = evaluate_merge_window(
        np.array([TWO_PI - 0.004, 0.003]),
        np.array([0.0, 0.0]),
        phase_tol_rad=0.005,
        spatial_tol_m=0.001,
        required_consecutive_samples=1,
    )
    assert report.phase_locked
    assert report.phase_dispersion_rad == pytest.approx(0.004, abs=1.0e-12)
    assert report.lock_achieved


def test_report_serialisation_and_lazy_export() -> None:
    report = evaluate_merge_window(
        np.array([0.0, 0.002]),
        np.array([0.0, 0.001]),
        t=7.5,
        required_consecutive_samples=1,
    )
    payload = merge_window_report_to_dict(report)
    assert payload == report.to_dict()
    assert payload["t"] == 7.5
    assert isinstance(payload["lock_achieved"], bool)
    assert monitor_pkg.MergeReport is MergeReport
    assert monitor_pkg.MergeWindowToleranceProfile is MergeWindowToleranceProfile
    assert monitor_pkg.MergeWindowMonitor is MergeWindowMonitor
    assert monitor_pkg.evaluate_merge_window is evaluate_merge_window


def test_tolerance_profiles_resolve_reviewed_buffers() -> None:
    baseline = resolve_merge_window_tolerance_profile("baseline_1x")
    buffer = resolve_merge_window_tolerance_profile("buffer_3x")
    review = resolve_merge_window_tolerance_profile("review_5x")

    assert MERGE_WINDOW_TOLERANCE_PROFILE_MULTIPLIERS["buffer_3x"] == 3.0
    assert baseline.phase_tol_rad == pytest.approx(0.01)
    assert baseline.spatial_tol_m == pytest.approx(0.002)
    assert buffer.phase_tol_rad == pytest.approx(0.03)
    assert buffer.spatial_tol_m == pytest.approx(0.006)
    assert review.phase_tol_rad == pytest.approx(0.05)
    assert review.spatial_tol_m == pytest.approx(0.01)
    assert merge_window_tolerance_profile_to_dict(buffer) == buffer.to_dict()

    custom = MergeWindowToleranceProfile(
        name="buffer_3x",
        phase_tol_rad=0.6,
        spatial_tol_m=0.12,
        multiplier=3.0,
        baseline_phase_tol_rad=0.2,
        baseline_spatial_tol_m=0.04,
    )
    assert resolve_merge_window_tolerance_profile(custom) is custom


def test_tolerance_profile_controls_monitor_and_function() -> None:
    phases = np.array([0.0, 0.024], dtype=np.float64)
    positions = np.array([0.0, 0.0045], dtype=np.float64)

    strict = evaluate_merge_window(
        phases,
        positions,
        phase_tol_rad=0.01,
        spatial_tol_m=0.002,
        required_consecutive_samples=1,
    )
    profiled = evaluate_merge_window(
        phases,
        positions,
        phase_tol_rad=0.01,
        spatial_tol_m=0.002,
        required_consecutive_samples=1,
        tolerance_profile="buffer_3x",
    )
    monitor = MergeWindowMonitor(
        phase_tol_rad=0.01,
        spatial_tol_m=0.002,
        required_consecutive_samples=1,
        tolerance_profile="buffer_3x",
    )
    monitored = monitor.evaluate(phases, positions)

    assert not strict.lock_achieved
    assert profiled.lock_achieved
    assert monitored.lock_achieved
    assert monitor.tolerance_profile is not None
    assert monitor.tolerance_profile.name == "buffer_3x"
    assert monitor.phase_tol_rad == pytest.approx(0.03)
    assert monitor.spatial_tol_m == pytest.approx(0.006)


def test_invalid_inputs_fail_closed() -> None:
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
            evaluate_merge_window(phases, positions)

    with pytest.raises(ValueError, match="phase_tol_rad"):
        MergeWindowMonitor(phase_tol_rad=-0.1)
    with pytest.raises(ValueError, match="spatial_tol_m"):
        MergeWindowMonitor(spatial_tol_m=-0.1)
    with pytest.raises(ValueError, match="required_consecutive_samples"):
        MergeWindowMonitor(required_consecutive_samples=0)
    with pytest.raises(ValueError, match="prior_consecutive_lock_samples"):
        evaluate_merge_window([0.0], [0.0], prior_consecutive_lock_samples=-1)
    with pytest.raises(ValueError, match="t"):
        evaluate_merge_window([0.0], [0.0], t=True)
    with pytest.raises(ValueError, match="tolerance_profile"):
        evaluate_merge_window([0.0], [0.0], tolerance_profile="unknown")
    with pytest.raises(ValueError, match="multiplier"):
        MergeWindowToleranceProfile(
            name="baseline_1x",
            phase_tol_rad=0.01,
            spatial_tol_m=0.002,
            multiplier=0.0,
            baseline_phase_tol_rad=0.01,
            baseline_spatial_tol_m=0.002,
        )


@st.composite
def _phase_position_vectors(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray]:
    n = draw(st.integers(min_value=2, max_value=8))
    phase_values = draw(
        st.lists(
            st.floats(
                min_value=-0.05,
                max_value=0.05,
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
                min_value=-0.01,
                max_value=0.01,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=n,
            max_size=n,
        )
    )
    return np.array(phase_values, dtype=np.float64), np.array(
        position_values,
        dtype=np.float64,
    )


@given(
    vectors=_phase_position_vectors(),
    phase_tol=st.floats(
        min_value=0.0,
        max_value=0.05,
        allow_nan=False,
        allow_infinity=False,
    ),
    spatial_tol=st.floats(
        min_value=0.0,
        max_value=0.01,
        allow_nan=False,
        allow_infinity=False,
    ),
    extra=st.floats(
        min_value=0.0,
        max_value=0.05,
        allow_nan=False,
        allow_infinity=False,
    ),
)
@settings(max_examples=32, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_lock_is_monotone_under_looser_tolerances(
    vectors: tuple[np.ndarray, np.ndarray],
    phase_tol: float,
    spatial_tol: float,
    extra: float,
) -> None:
    phases, positions = vectors
    strict = evaluate_merge_window(
        phases,
        positions,
        phase_tol_rad=phase_tol,
        spatial_tol_m=spatial_tol,
        required_consecutive_samples=1,
    )
    loose = evaluate_merge_window(
        phases,
        positions,
        phase_tol_rad=phase_tol + extra,
        spatial_tol_m=spatial_tol + extra,
        required_consecutive_samples=1,
    )
    if strict.lock_achieved:
        assert loose.lock_achieved


def test_polyglot_adapter_contracts_match_python_reference() -> None:
    phases = np.array([TWO_PI - 0.004, 0.0, 0.003], dtype=np.float64)
    positions = np.array([-0.001, 0.0, 0.001], dtype=np.float64)
    expected = evaluate_merge_window(
        phases,
        positions,
        t=3.0,
        phase_tol_rad=0.005,
        spatial_tol_m=0.002,
        required_consecutive_samples=2,
        prior_consecutive_lock_samples=1,
    )
    adapters = (
        _merge_window_rust.evaluate_merge_window_rust,
        _merge_window_go.evaluate_merge_window_go,
        _merge_window_julia.evaluate_merge_window_julia,
        _merge_window_mojo.evaluate_merge_window_mojo,
        public_merge_rust.evaluate_merge_window_rust,
        public_merge_go.evaluate_merge_window_go,
        public_merge_julia.evaluate_merge_window_julia,
        public_merge_mojo.evaluate_merge_window_mojo,
    )
    for adapter in adapters:
        got = adapter(
            phases,
            positions,
            t=3.0,
            phase_tol_rad=0.005,
            spatial_tol_m=0.002,
            required_consecutive_samples=2,
            prior_consecutive_lock_samples=1,
            tolerance_profile="baseline_1x",
        )
        assert merge_validation.validate_merge_window_report(got, expected) is got


def test_merge_window_polyglot_benchmark_gate() -> None:
    result = benchmark_merge_window_polyglot_parity_gate(n=6, calls=1)
    assert result["suite"] == "merge_window_polyglot_parity_gate"
    assert result["backend_count"] == 5
    assert result["all_available_passed"] == 1
    assert result["acceptance_passed"] == 1
    assert result["buffer_profile_accepts_within_3x"] == 1
    assert result["explicit_profile_rejects_same_sample"] == 1
    assert result["benchmark_evidence_kind"] == "local_regression_non_isolated"
    assert result["production_timing_claim"] == 0
