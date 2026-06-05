# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — PHA-C acceptance tests

"""Behavioural tests for ``upde.pha_c_acceptance``."""

from __future__ import annotations

import importlib
import json
from dataclasses import replace

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from benchmarks.pha_c_acceptance_benchmark import (
    benchmark_pha_c_acceptance_polyglot_gate,
)
from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _pha_c_acceptance_go,
    _pha_c_acceptance_julia,
    _pha_c_acceptance_mojo,
    _pha_c_acceptance_rust,
    _pha_c_acceptance_validation,
)
from scpn_phase_orchestrator.upde import PHACAcceptanceRecord as ExportedRecord
from scpn_phase_orchestrator.upde import (
    verify_pha_c_acceptance_record as exported_verify,
)
from scpn_phase_orchestrator.upde.pha_c_acceptance import (
    PHA_C_ACCEPTANCE_CLAIM_BOUNDARY,
    PHA_C_ACCEPTANCE_KINEMATIC_SUMMARY_REPLAY_TOLERANCE,
    PHA_C_ACCEPTANCE_MARGIN_REPLAY_TOLERANCE,
    PHACAcceptanceRecord,
    build_pha_c_acceptance_record,
    pha_c_acceptance_record_to_dict,
    verify_pha_c_acceptance_record,
)

MODULE_LINKAGE_PATHS = (
    "scpn_phase_orchestrator.upde.pha_c_acceptance",
    "scpn_phase_orchestrator.experimental.accelerators.upde._pha_c_acceptance_go",
    "scpn_phase_orchestrator.experimental.accelerators.upde._pha_c_acceptance_julia",
    "scpn_phase_orchestrator.experimental.accelerators.upde._pha_c_acceptance_mojo",
    "scpn_phase_orchestrator.experimental.accelerators.upde._pha_c_acceptance_rust",
    "scpn_phase_orchestrator.experimental.accelerators.upde._pha_c_acceptance_validation",
    "scpn_phase_orchestrator.upde.pha_c_formal_obligation",
)


def _problem(
    n: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    phases = np.linspace(-0.002, 0.002, n, dtype=np.float64)
    positions = np.linspace(-0.0006, 0.0006, n, dtype=np.float64)
    omega = np.zeros((4, n), dtype=np.float64)
    knm = np.full((n, n), 0.04, dtype=np.float64)
    np.fill_diagonal(knm, 0.0)
    velocity_base = np.linspace(0.10, 0.12, n, dtype=np.float64)
    velocities = np.vstack(
        [velocity_base + 1.0e-3 * step for step in range(4)],
    ).astype(np.float64, copy=False)
    return phases, positions, omega, knm, velocities


def test_module_linkage_paths_cover_pha_c_acceptance_chain() -> None:
    for import_path in MODULE_LINKAGE_PATHS:
        assert importlib.import_module(import_path).__name__ == import_path


def test_acceptance_record_spans_complete_review_only_chain() -> None:
    phases, positions, omega, knm, velocities = _problem()

    record = build_pha_c_acceptance_record(
        phases,
        positions,
        omega,
        knm,
        velocities,
        dt=1.0e-3,
        required_consecutive_samples=3,
        tolerance_profile="baseline_1x",
        backend="python",
    )
    repeated = build_pha_c_acceptance_record(
        phases,
        positions,
        omega,
        knm,
        velocities,
        dt=1.0e-3,
        required_consecutive_samples=3,
        tolerance_profile="baseline_1x",
        backend="python",
    )

    assert record.sample_count == 5
    assert record.step_count == 4
    assert record.oscillator_count == 5
    assert record.first_lock_observed
    assert record.first_lock_index == 2
    assert record.final_lock_achieved
    assert record.lock_loss_count == 0
    assert record.reset_count == 0
    assert record.max_consecutive_lock_samples == 5
    assert record.max_abs_doppler_term > 0.0
    assert record.max_abs_spatial_coupling > 0.0
    assert record.kinematic_residual_max_m <= 1.0e-12
    assert record.max_abs_velocity_m_per_s == pytest.approx(0.123)
    assert record.path_length_max_m > 0.0
    assert record.final_position_equation_validated
    assert record.max_abs_velocity_equation_validated
    assert record.path_length_equation_validated
    assert record.kinematic_equations_validated
    assert record.kinematic_summary_replay_tolerance == (
        PHA_C_ACCEPTANCE_KINEMATIC_SUMMARY_REPLAY_TOLERANCE
    )
    assert record.claim_boundary == PHA_C_ACCEPTANCE_CLAIM_BOUNDARY
    assert record.execution_disabled
    assert not record.actuating
    assert record.tolerance_profile_name == "baseline_1x"
    assert record.tolerance_profile_multiplier == pytest.approx(1.0)
    assert len(record.phase_trajectory_sha256) == 64
    assert len(record.position_trajectory_sha256) == 64
    assert len(record.timeline_sha256) == 64
    assert len(record.acceptance_sha256) == 64
    assert record.acceptance_sha256 == repeated.acceptance_sha256
    assert pha_c_acceptance_record_to_dict(record) == record.to_dict()
    assert ExportedRecord is PHACAcceptanceRecord
    assert exported_verify is verify_pha_c_acceptance_record
    assert verify_pha_c_acceptance_record(record) is record


def test_acceptance_replay_verifier_rejects_tampered_evidence() -> None:
    phases, positions, omega, knm, velocities = _problem()
    record = build_pha_c_acceptance_record(
        phases,
        positions,
        omega,
        knm,
        velocities,
        dt=1.0e-3,
        required_consecutive_samples=3,
        tolerance_profile="baseline_1x",
        backend="python",
    )

    with pytest.raises(ValueError, match="acceptance_sha256"):
        verify_pha_c_acceptance_record(
            replace(record, acceptance_sha256="0" * 64),
        )
    with pytest.raises(ValueError, match="sample_count"):
        verify_pha_c_acceptance_record(replace(record, sample_count=record.step_count))
    with pytest.raises(ValueError, match="claim_boundary"):
        verify_pha_c_acceptance_record(replace(record, claim_boundary="actuating"))


def test_acceptance_tolerance_profile_records_review_boundary() -> None:
    phases, positions, omega, knm, velocities = _problem()

    record = build_pha_c_acceptance_record(
        phases,
        positions,
        omega,
        knm,
        velocities,
        phase_tol_rad=0.01,
        spatial_tol_m=0.002,
        tolerance_profile="buffer_3x",
        required_consecutive_samples=1,
        backend="python",
    )

    assert record.first_lock_observed
    assert record.final_lock_achieved
    assert record.phase_tol_rad == pytest.approx(0.03)
    assert record.spatial_tol_m == pytest.approx(0.006)
    assert record.tolerance_profile_name == "buffer_3x"
    assert record.tolerance_profile_multiplier == pytest.approx(3.0)


def test_invalid_acceptance_inputs_fail_closed() -> None:
    phases, positions, omega, knm, velocities = _problem()
    invalid_cases = (
        ([], positions, omega, knm, velocities, "phases_t0"),
        (phases, positions[:-1], omega, knm, velocities, "same shape"),
        ([[0.0]], positions, omega, knm, velocities, "one-dimensional"),
        ([np.nan] * phases.size, positions, omega, knm, velocities, "finite"),
        ([True] * phases.size, positions, omega, knm, velocities, "real-valued"),
        (phases, positions, omega[:, :-1], knm, velocities, "omega_schedule"),
        (phases, positions, omega, knm[:-1], velocities, "knm"),
        (phases, positions, omega, knm, velocities[:-1], "step count"),
    )
    for case in invalid_cases:
        (
            phase_values,
            position_values,
            omega_values,
            knm_values,
            velocity_values,
            match,
        ) = case
        with pytest.raises(ValueError, match=match):
            build_pha_c_acceptance_record(
                phase_values,
                position_values,
                omega_values,
                knm_values,
                velocity_values,
            )

    with pytest.raises(ValueError, match="dt"):
        build_pha_c_acceptance_record(phases, positions, omega, knm, velocities, dt=0.0)
    with pytest.raises(ValueError, match="n_substeps"):
        build_pha_c_acceptance_record(
            phases,
            positions,
            omega,
            knm,
            velocities,
            n_substeps=0,
        )
    with pytest.raises(ValueError, match="tolerance_profile"):
        build_pha_c_acceptance_record(
            phases,
            positions,
            omega,
            knm,
            velocities,
            tolerance_profile="unknown",
        )


@st.composite
def _shifted_acceptance_inputs(
    draw: st.DrawFn,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    n = draw(st.integers(min_value=3, max_value=6))
    shift = draw(
        st.floats(
            min_value=-np.pi,
            max_value=np.pi,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    phase_values = draw(
        st.lists(
            st.floats(
                min_value=-0.002,
                max_value=0.002,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=n,
            max_size=n,
        ),
    )
    position_values = draw(
        st.lists(
            st.floats(
                min_value=-0.0006,
                max_value=0.0006,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=n,
            max_size=n,
        ),
    )
    phases = np.array(phase_values, dtype=np.float64)
    positions = np.array(position_values, dtype=np.float64)
    omega = np.zeros((3, n), dtype=np.float64)
    knm = np.full((n, n), 0.03, dtype=np.float64)
    np.fill_diagonal(knm, 0.0)
    velocity_base = np.linspace(0.10, 0.12, n, dtype=np.float64)
    velocities = np.vstack(
        [velocity_base + 1.0e-3 * step for step in range(3)],
    ).astype(np.float64, copy=False)
    return phases, positions, omega, knm, velocities, float(shift)


@given(inputs=_shifted_acceptance_inputs())
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_acceptance_lock_counts_survive_global_phase_shift(
    inputs: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float],
) -> None:
    phases, positions, omega, knm, velocities, shift = inputs
    reference = build_pha_c_acceptance_record(
        phases,
        positions,
        omega,
        knm,
        velocities,
        reference_phase=0.0,
        required_consecutive_samples=2,
        backend="python",
    )
    shifted = build_pha_c_acceptance_record(
        phases + shift,
        positions,
        omega,
        knm,
        velocities,
        reference_phase=shift,
        required_consecutive_samples=2,
        backend="python",
    )

    assert shifted.first_lock_index == reference.first_lock_index
    assert shifted.final_lock_achieved is reference.final_lock_achieved
    assert shifted.lock_sample_count == reference.lock_sample_count
    assert shifted.lock_loss_count == reference.lock_loss_count
    assert shifted.reset_count == reference.reset_count
    assert shifted.min_phase_order_parameter == pytest.approx(
        reference.min_phase_order_parameter,
        abs=1.0e-12,
    )
    assert shifted.max_distance_to_reference_m == pytest.approx(
        reference.max_distance_to_reference_m,
        abs=1.0e-12,
    )


def test_polyglot_acceptance_adapter_contracts_match_reference() -> None:
    phases, positions, omega, knm, velocities = _problem()
    expected = build_pha_c_acceptance_record(
        phases,
        positions,
        omega,
        knm,
        velocities,
        dt=1.0e-3,
        required_consecutive_samples=3,
        backend="python",
    )
    adapters = (
        _pha_c_acceptance_rust.build_pha_c_acceptance_record_rust,
        _pha_c_acceptance_mojo.build_pha_c_acceptance_record_mojo,
        _pha_c_acceptance_julia.build_pha_c_acceptance_record_julia,
        _pha_c_acceptance_go.build_pha_c_acceptance_record_go,
    )
    for adapter in adapters:
        got = adapter(
            phases,
            positions,
            omega,
            knm,
            velocities,
            dt=1.0e-3,
            required_consecutive_samples=3,
            backend="python",
        )
        assert (
            _pha_c_acceptance_validation.validate_pha_c_acceptance_record(
                got,
                expected,
            )
            is got
        )


def test_pha_c_acceptance_benchmark_gate_accepts_declared_backends() -> None:
    result = benchmark_pha_c_acceptance_polyglot_gate(
        n=5,
        calls=1,
        include_subgates=False,
    )

    assert result["suite"] == "pha_c_acceptance_polyglot_gate"
    assert result["backend_count"] == 5
    assert result["parity_pass_count"] == 5
    assert result["source_contract_backend_count"] == 4
    assert result["native_kernel_count"] == 0
    assert result["polyglot_claim_boundary"] == "source_contract_not_native_kernel"
    assert result["acceptance_passed"] == 1
    assert result["hash_replay_validated"] == 1
    assert result["first_lock_observed"] == 1
    assert result["first_lock_index"] == 2
    assert result["final_lock_achieved"] == 1
    assert result["lock_loss_count"] == 0
    assert result["reset_count"] == 0
    assert result["phase_margin_positive"] == 1
    assert result["spatial_margin_positive"] == 1
    assert result["phase_margin_equation_validated"] == 1
    assert result["spatial_margin_equation_validated"] == 1
    assert result["signed_margin_equations_validated"] == 1
    assert result["margin_replay_tolerance"] == (
        PHA_C_ACCEPTANCE_MARGIN_REPLAY_TOLERANCE
    )
    assert result["kinematic_residual_contract_passed"] == 1
    assert result["kinematic_residual_max_m"] <= 1.0e-12
    assert result["final_position_equation_validated"] == 1
    assert result["max_abs_velocity_equation_validated"] == 1
    assert result["path_length_equation_validated"] == 1
    assert result["kinematic_equations_validated"] == 1
    assert result["kinematic_summary_replay_tolerance"] == (
        PHA_C_ACCEPTANCE_KINEMATIC_SUMMARY_REPLAY_TOLERANCE
    )
    assert result["formal_obligation_discharged"] == 1
    assert result["formal_obligation_margin_units"] >= 0
    assert result["formal_obligation_time_step_units"] > 0
    assert result["formal_obligation_horizon_time_units"] == (
        result["formal_obligation_time_step_units"] * 4
    )
    assert result["formal_obligation_relative_velocity_rate_units_per_second"] == 0
    assert result["formal_obligation_configured_residual_step_units"] == 0
    assert result["formal_obligation_residual_rate_units_per_second"] == 0
    assert result["formal_obligation_configured_phase_drift_units"] == 0
    assert result["formal_obligation_phase_budget_units"] >= 0
    assert result["formal_obligation_continuous_drive_rate_units_per_second"] == 0
    assert result["formal_obligation_continuous_horizon_drive_units"] == 0
    assert result["formal_obligation_continuous_linear_budget_units"] >= 0
    assert result["formal_obligation_continuous_margin_units"] >= 0
    assert result["formal_obligation_continuous_envelope_discharged"] == 1
    assert result["formal_obligation_gronwall_margin_units"] == (
        result["formal_obligation_margin_units"]
    )
    assert len(str(result["formal_obligation_trace_sha256"])) == 64
    assert result["formal_obligation_phase_margin_units"] >= 0
    assert result["formal_obligation_phase_margin_units"] == (
        result["formal_obligation_phase_tolerance_units"]
        - result["formal_obligation_phase_budget_units"]
    )
    assert result["formal_obligation_phase_budget_discharged"] == 1
    assert result["formal_obligation_phase_theorem"] == (
        "phase_budget_certificate_discharges_phase_lock"
    )
    assert result["formal_obligation_theorem"] == (
        "budget_certificate_discharges_budget"
    )
    assert result["formal_obligation_continuous_theorem"] == (
        "continuous_envelope_certificate_discharges_horizon"
    )
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
    assert all(
        int(record["formal_obligation_discharged"]) == 1
        for record in backend_records
    )
    assert all(
        record["formal_obligation_gronwall_margin_units"]
        == record["formal_obligation_margin_units"]
        for record in backend_records
    )
    assert all(
        record["formal_obligation_horizon_time_units"]
        == record["formal_obligation_time_step_units"] * 4
        for record in backend_records
    )
    assert all(
        record["formal_obligation_relative_velocity_rate_units_per_second"] == 0
        for record in backend_records
    )
    assert all(
        record["formal_obligation_continuous_drive_rate_units_per_second"] == 0
        for record in backend_records
    )
    assert all(
        record["formal_obligation_continuous_horizon_drive_units"] == 0
        for record in backend_records
    )
    assert all(
        int(record["formal_obligation_continuous_envelope_discharged"]) == 1
        for record in backend_records
    )
    assert all(
        len(str(record["formal_obligation_trace_sha256"])) == 64
        for record in backend_records
    )
    assert all(
        record["kinematic_residual_max_m"] <= 1.0e-12
        for record in backend_records
    )
    assert all(
        int(record["kinematic_equations_validated"]) == 1
        for record in backend_records
    )


def test_acceptance_signed_margins_are_hash_replayed() -> None:
    from dataclasses import replace

    n = 5
    phases = np.linspace(-0.002, 0.002, n, dtype=np.float64)
    positions = np.linspace(-0.0006, 0.0006, n, dtype=np.float64)
    omega = np.zeros((4, n), dtype=np.float64)
    knm = np.full((n, n), 0.04, dtype=np.float64)
    np.fill_diagonal(knm, 0.0)
    velocities = np.vstack(
        [np.linspace(0.10, 0.12, n, dtype=np.float64) for _ in range(4)]
    )
    record = build_pha_c_acceptance_record(
        phases,
        positions,
        omega,
        knm,
        velocities,
        dt=1.0e-3,
        required_consecutive_samples=3,
        backend="python",
    )

    assert record.min_phase_margin_rad == pytest.approx(
        record.phase_tol_rad - record.max_phase_dispersion_rad,
        abs=PHA_C_ACCEPTANCE_MARGIN_REPLAY_TOLERANCE,
    )
    assert record.min_spatial_margin_m == pytest.approx(
        record.spatial_tol_m - record.max_spatial_dispersion_m,
        abs=PHA_C_ACCEPTANCE_MARGIN_REPLAY_TOLERANCE,
    )
    assert record.kinematic_residual_max_m <= 1.0e-12
    assert record.kinematic_equations_validated
    assert record.min_phase_margin_rad >= 0.0
    assert record.min_spatial_margin_m >= 0.0
    assert verify_pha_c_acceptance_record(record) is record

    forged = replace(record, min_phase_margin_rad=-1.0)
    with pytest.raises(ValueError, match="min_phase_margin_rad"):
        verify_pha_c_acceptance_record(forged)

    forged_positive_phase = replace(
        record,
        min_phase_margin_rad=record.min_phase_margin_rad
        + 100.0 * PHA_C_ACCEPTANCE_MARGIN_REPLAY_TOLERANCE,
    )
    with pytest.raises(ValueError, match="min_phase_margin_rad"):
        verify_pha_c_acceptance_record(forged_positive_phase)

    forged_positive_spatial = replace(
        record,
        min_spatial_margin_m=record.min_spatial_margin_m
        + 100.0 * PHA_C_ACCEPTANCE_MARGIN_REPLAY_TOLERANCE,
    )
    with pytest.raises(ValueError, match="min_spatial_margin_m"):
        verify_pha_c_acceptance_record(forged_positive_spatial)

    forged_residual = replace(record, kinematic_residual_max_m=1.0e-3)
    with pytest.raises(ValueError, match="kinematic_residual_max_m"):
        verify_pha_c_acceptance_record(forged_residual)

    forged_kinematic_equation = replace(
        record,
        path_length_equation_validated=False,
    )
    with pytest.raises(ValueError, match="path_length_equation_validated"):
        verify_pha_c_acceptance_record(forged_kinematic_equation)
