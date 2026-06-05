# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — PHA-C Lean obligation tests

"""Behavioural tests for ``upde.pha_c_formal_obligation``."""

from __future__ import annotations

from dataclasses import replace
from decimal import ROUND_CEILING, Decimal
from math import ceil

import numpy as np
import pytest

from scpn_phase_orchestrator.upde import (
    PHACKinematicProofObligation as ExportedObligation,
)
from scpn_phase_orchestrator.upde import (
    build_pha_c_kinematic_proof_obligation as exported_build,
)
from scpn_phase_orchestrator.upde import (
    verify_pha_c_kinematic_proof_obligation as exported_verify,
)
from scpn_phase_orchestrator.upde.pha_c_acceptance import (
    PHA_C_ACCEPTANCE_KINEMATIC_SUMMARY_REPLAY_TOLERANCE,
    build_pha_c_acceptance_record,
)
from scpn_phase_orchestrator.upde.pha_c_formal_obligation import (
    PHA_C_FORMAL_ACCEPTANCE_CERTIFICATE_PREDICATE,
    PHA_C_FORMAL_ACCEPTANCE_CERTIFICATE_THEOREM,
    PHA_C_FORMAL_CERTIFICATE_PREDICATE,
    PHA_C_FORMAL_CERTIFICATE_THEOREM,
    PHA_C_FORMAL_CONTINUOUS_CERTIFICATE_PREDICATE,
    PHA_C_FORMAL_CONTINUOUS_CERTIFICATE_THEOREM,
    PHA_C_FORMAL_CONTINUOUS_LEAN_MODULE,
    PHA_C_FORMAL_DEFAULT_TIME_SCALE_S,
    PHA_C_FORMAL_LEAN_MODULE,
    PHA_C_FORMAL_OBLIGATION_CLAIM_BOUNDARY,
    PHA_C_FORMAL_OBLIGATION_SCHEMA,
    PHA_C_FORMAL_PHASE_CERTIFICATE_PREDICATE,
    PHA_C_FORMAL_PHASE_CERTIFICATE_THEOREM,
    PHA_C_FORMAL_PHASE_LEAN_MODULE,
    PHACKinematicProofObligation,
    build_pha_c_kinematic_proof_obligation,
    pha_c_kinematic_proof_obligation_to_dict,
    verify_pha_c_kinematic_proof_obligation,
)


def _ceil_units(value: float, scale: float) -> int:
    return int(
        (Decimal(str(value)) / Decimal(str(scale))).to_integral_value(
            rounding=ROUND_CEILING,
        )
    )


def _record(*, spatial_tol_m: float | None = None):
    n = 5
    phases = np.linspace(-0.002, 0.002, n, dtype=np.float64)
    positions = np.linspace(-0.0006, 0.0006, n, dtype=np.float64)
    omega = np.zeros((4, n), dtype=np.float64)
    knm = np.full((n, n), 0.04, dtype=np.float64)
    np.fill_diagonal(knm, 0.0)
    velocity_base = np.linspace(0.10, 0.12, n, dtype=np.float64)
    velocities = np.vstack(
        [velocity_base + 1.0e-3 * step for step in range(4)],
    ).astype(np.float64, copy=False)
    kwargs = {}
    if spatial_tol_m is not None:
        kwargs["spatial_tol_m"] = spatial_tol_m
    return build_pha_c_acceptance_record(
        phases,
        positions,
        omega,
        knm,
        velocities,
        dt=1.0e-3,
        required_consecutive_samples=3,
        tolerance_profile="baseline_1x",
        backend="python",
        **kwargs,
    )


def test_kinematic_obligation_maps_acceptance_record_to_lean_bounds() -> None:
    record = _record()
    obligation = build_pha_c_kinematic_proof_obligation(record)

    assert obligation.schema_version == PHA_C_FORMAL_OBLIGATION_SCHEMA
    assert obligation.claim_boundary == PHA_C_FORMAL_OBLIGATION_CLAIM_BOUNDARY
    assert obligation.execution_disabled
    assert not obligation.actuating
    assert obligation.lean_module == PHA_C_FORMAL_LEAN_MODULE
    assert obligation.lean_certificate_predicate == PHA_C_FORMAL_CERTIFICATE_PREDICATE
    assert obligation.lean_theorem == PHA_C_FORMAL_CERTIFICATE_THEOREM
    assert obligation.continuous_lean_module == PHA_C_FORMAL_CONTINUOUS_LEAN_MODULE
    assert obligation.continuous_certificate_predicate == (
        PHA_C_FORMAL_CONTINUOUS_CERTIFICATE_PREDICATE
    )
    assert obligation.continuous_theorem == PHA_C_FORMAL_CONTINUOUS_CERTIFICATE_THEOREM
    assert obligation.phase_lean_module == PHA_C_FORMAL_PHASE_LEAN_MODULE
    assert obligation.phase_certificate_predicate == (
        PHA_C_FORMAL_PHASE_CERTIFICATE_PREDICATE
    )
    assert obligation.phase_theorem == PHA_C_FORMAL_PHASE_CERTIFICATE_THEOREM
    assert obligation.acceptance_certificate_predicate == (
        PHA_C_FORMAL_ACCEPTANCE_CERTIFICATE_PREDICATE
    )
    assert obligation.acceptance_certificate_theorem == (
        PHA_C_FORMAL_ACCEPTANCE_CERTIFICATE_THEOREM
    )
    assert obligation.fixed_point_time_scale_s == PHA_C_FORMAL_DEFAULT_TIME_SCALE_S
    assert obligation.time_step_s == pytest.approx(record.dt)
    assert obligation.time_scale_units_per_second == 1_000_000
    assert obligation.time_step_units == 1000
    assert obligation.horizon_time_units == obligation.horizon_steps * 1000
    assert obligation.lipschitz_step_gain_units == 0
    assert obligation.relative_velocity_rate_bound_units_per_second == 0
    assert obligation.relative_velocity_step_bound_units == 0
    assert obligation.configured_coupling_residual_step_bound_units == 0
    assert obligation.coupling_residual_rate_bound_units_per_second == 0
    assert obligation.coupling_residual_step_bound_units == 0
    assert obligation.continuous_drive_rate_bound_units_per_second == 0
    assert obligation.continuous_horizon_drive_bound_units == 0
    assert obligation.continuous_linear_budget_units == (
        obligation.initial_tolerance_units
    )
    assert obligation.continuous_margin_units == (
        obligation.merge_window_tolerance_units
        - obligation.continuous_linear_budget_units
    )
    assert obligation.continuous_envelope_discharged
    assert obligation.drive_bound_units == 0
    assert obligation.horizon_steps == record.step_count
    assert obligation.initial_tolerance_units == _ceil_units(
        record.max_spatial_dispersion_m,
        obligation.fixed_point_scale_m,
    )
    assert obligation.merge_window_tolerance_units == _ceil_units(
        record.spatial_tol_m,
        obligation.fixed_point_scale_m,
    )
    assert obligation.linear_budget_units == obligation.initial_tolerance_units
    assert obligation.gronwall_budget_units == obligation.linear_budget_units
    assert obligation.gronwall_budget_margin_units == (
        obligation.window_budget_margin_units
    )
    assert len(obligation.gronwall_budget_trace_sha256) == 64
    assert obligation.window_budget_margin_units >= 0
    assert obligation.configured_phase_drift_bound_units == 0
    assert obligation.phase_budget_units == obligation.max_phase_dispersion_units
    assert obligation.phase_margin_units == (
        obligation.phase_tolerance_units - obligation.phase_budget_units
    )
    assert obligation.phase_budget_discharged
    assert obligation.phase_margin_units >= 0
    assert obligation.acceptance_kinematic_equations_validated
    assert obligation.acceptance_kinematic_summary_replay_tolerance == (
        PHA_C_ACCEPTANCE_KINEMATIC_SUMMARY_REPLAY_TOLERANCE
    )
    assert obligation.acceptance_kinematic_summary_replay_tolerance_units == 1
    assert obligation.acceptance_kinematic_summary_replay_tolerance_limit_units == 1
    assert obligation.acceptance_replay_certificate_discharged
    assert obligation.acceptance_certificate_discharged
    assert obligation.observed_velocity_step_units == _ceil_units(
        record.max_abs_velocity_m_per_s * record.dt,
        obligation.fixed_point_scale_m,
    )
    assert obligation.path_length_units == _ceil_units(
        record.path_length_max_m,
        obligation.fixed_point_scale_m,
    )
    assert obligation.proof_obligations_discharged
    assert len(obligation.record_sha256) == 64
    assert len(obligation.acceptance_sha256) == 64
    assert len(obligation.timeline_sha256) == 64
    assert pha_c_kinematic_proof_obligation_to_dict(obligation) == (
        obligation.to_dict()
    )
    assert ExportedObligation is PHACKinematicProofObligation
    assert exported_build is build_pha_c_kinematic_proof_obligation
    assert exported_verify is verify_pha_c_kinematic_proof_obligation
    assert verify_pha_c_kinematic_proof_obligation(obligation) is obligation


def test_kinematic_obligation_supports_predictive_residual_slack() -> None:
    record = _record(spatial_tol_m=0.1)
    obligation = build_pha_c_kinematic_proof_obligation(
        record,
        coupling_residual_step_bound_m=2.0e-5,
    )

    expected_residual_units = _ceil_units(2.0e-5, obligation.fixed_point_scale_m)
    expected_residual_rate_units = _ceil_units(
        2.0e-5 / record.dt,
        obligation.fixed_point_scale_m,
    )
    assert obligation.configured_coupling_residual_step_bound_units == (
        expected_residual_units
    )
    assert obligation.coupling_residual_step_bound_units == expected_residual_units
    assert obligation.coupling_residual_rate_bound_units_per_second == (
        expected_residual_rate_units
    )
    assert obligation.drive_bound_units == expected_residual_units
    assert obligation.continuous_drive_rate_bound_units_per_second == (
        expected_residual_rate_units
    )
    assert obligation.continuous_horizon_drive_bound_units == (
        expected_residual_units * obligation.horizon_steps
    )
    assert obligation.continuous_linear_budget_units == (
        obligation.initial_tolerance_units
        + obligation.continuous_horizon_drive_bound_units
    )
    assert obligation.continuous_margin_units >= 0
    assert obligation.continuous_envelope_discharged
    assert obligation.gronwall_budget_units == (
        obligation.initial_tolerance_units
        + obligation.horizon_steps * obligation.drive_bound_units
    )
    assert obligation.proof_obligations_discharged
    assert verify_pha_c_kinematic_proof_obligation(obligation) is obligation


def test_kinematic_obligation_supports_predictive_phase_drift_slack() -> None:
    record = _record()
    obligation = build_pha_c_kinematic_proof_obligation(
        record,
        phase_drift_bound_rad=2.5e-3,
    )

    expected_drift_units = _ceil_units(
        2.5e-3,
        obligation.fixed_point_scale_rad,
    )
    assert obligation.configured_phase_drift_bound_units == expected_drift_units
    assert obligation.phase_budget_units == (
        obligation.max_phase_dispersion_units + expected_drift_units
    )
    assert obligation.phase_margin_units == (
        obligation.phase_tolerance_units - obligation.phase_budget_units
    )
    assert obligation.phase_budget_discharged
    assert obligation.phase_margin_units >= 0
    assert obligation.proof_obligations_discharged
    assert verify_pha_c_kinematic_proof_obligation(obligation) is obligation


def test_kinematic_obligation_supports_predictive_relative_velocity_slack() -> None:
    record = _record()
    obligation = build_pha_c_kinematic_proof_obligation(
        record,
        relative_velocity_step_bound_m=1.0e-5,
    )

    expected_slack_units = _ceil_units(1.0e-5, obligation.fixed_point_scale_m)
    expected_rate_units = _ceil_units(
        1.0e-5 / record.dt,
        obligation.fixed_point_scale_m,
    )
    expected_sampled_units = ceil(
        expected_rate_units
        * obligation.time_step_units
        / obligation.time_scale_units_per_second,
    )
    assert expected_sampled_units == expected_slack_units
    assert obligation.relative_velocity_rate_bound_units_per_second == (
        expected_rate_units
    )
    assert obligation.relative_velocity_step_bound_units == expected_slack_units
    assert obligation.continuous_drive_rate_bound_units_per_second == (
        expected_rate_units
    )
    assert obligation.continuous_horizon_drive_bound_units == (
        expected_slack_units * obligation.horizon_steps
    )
    assert obligation.continuous_linear_budget_units == (
        obligation.initial_tolerance_units
        + obligation.continuous_horizon_drive_bound_units
    )
    assert obligation.continuous_margin_units == (
        obligation.merge_window_tolerance_units
        - obligation.continuous_linear_budget_units
    )
    assert obligation.continuous_envelope_discharged
    assert obligation.drive_bound_units == expected_slack_units
    assert obligation.linear_budget_units == (
        obligation.initial_tolerance_units
        + obligation.horizon_steps * obligation.drive_bound_units
    )
    assert obligation.gronwall_budget_units == obligation.linear_budget_units
    assert obligation.gronwall_budget_margin_units == (
        obligation.window_budget_margin_units
    )
    assert obligation.window_budget_margin_units >= 0
    assert obligation.proof_obligations_discharged
    assert verify_pha_c_kinematic_proof_obligation(obligation) is obligation


def test_kinematic_obligation_supports_nonzero_lipschitz_gain() -> None:
    record = _record(spatial_tol_m=0.1)
    obligation = build_pha_c_kinematic_proof_obligation(
        record,
        lipschitz_step_gain_units=1,
        relative_velocity_step_bound_m=1.0e-5,
    )

    expected_budget = obligation.initial_tolerance_units
    for _ in range(obligation.horizon_steps):
        expected_budget = (
            expected_budget
            + obligation.lipschitz_step_gain_units * expected_budget
            + obligation.drive_bound_units
        )

    assert obligation.lipschitz_step_gain_units == 1
    assert obligation.time_step_units > 0
    assert obligation.horizon_time_units == (
        obligation.horizon_steps * obligation.time_step_units
    )
    assert obligation.gronwall_budget_units == expected_budget
    assert obligation.gronwall_budget_margin_units == (
        obligation.merge_window_tolerance_units - expected_budget
    )
    assert obligation.continuous_linear_budget_units == (
        obligation.initial_tolerance_units
        + obligation.continuous_horizon_drive_bound_units
    )
    assert obligation.continuous_margin_units >= 0
    assert obligation.continuous_envelope_discharged
    assert obligation.window_budget_margin_units == (
        obligation.gronwall_budget_margin_units
    )
    assert obligation.proof_obligations_discharged
    assert verify_pha_c_kinematic_proof_obligation(obligation) is obligation


def test_kinematic_obligation_verifier_rejects_tampering() -> None:
    obligation = build_pha_c_kinematic_proof_obligation(_record())

    with pytest.raises(ValueError, match="record_sha256"):
        verify_pha_c_kinematic_proof_obligation(
            replace(obligation, record_sha256="0" * 64),
        )
    with pytest.raises(ValueError, match="drive_bound_units"):
        verify_pha_c_kinematic_proof_obligation(
            replace(obligation, drive_bound_units=obligation.drive_bound_units + 1),
        )
    with pytest.raises(ValueError, match="time_step_units"):
        verify_pha_c_kinematic_proof_obligation(
            replace(obligation, time_step_units=obligation.time_step_units + 1),
        )
    with pytest.raises(ValueError, match="relative_velocity_step_bound_units"):
        predictive = build_pha_c_kinematic_proof_obligation(
            _record(),
            relative_velocity_step_bound_m=1.0e-5,
        )
        verify_pha_c_kinematic_proof_obligation(
            replace(
                predictive,
                relative_velocity_step_bound_units=(
                    predictive.relative_velocity_step_bound_units + 1
                ),
            ),
        )
    with pytest.raises(
        ValueError,
        match="configured_coupling_residual_step_bound_units",
    ):
        predictive = build_pha_c_kinematic_proof_obligation(
            _record(spatial_tol_m=0.1),
            coupling_residual_step_bound_m=2.0e-5,
        )
        verify_pha_c_kinematic_proof_obligation(
            replace(
                predictive,
                configured_coupling_residual_step_bound_units=(
                    predictive.coupling_residual_step_bound_units + 1
                ),
            ),
        )
    with pytest.raises(ValueError, match="phase_budget_units"):
        predictive = build_pha_c_kinematic_proof_obligation(
            _record(),
            phase_drift_bound_rad=2.5e-3,
        )
        verify_pha_c_kinematic_proof_obligation(
            replace(
                predictive,
                phase_budget_units=predictive.phase_budget_units + 1,
            ),
        )
    with pytest.raises(ValueError, match="phase_margin_units"):
        predictive = build_pha_c_kinematic_proof_obligation(
            _record(),
            phase_drift_bound_rad=2.5e-3,
        )
        verify_pha_c_kinematic_proof_obligation(
            replace(
                predictive,
                phase_margin_units=predictive.phase_margin_units + 1,
            ),
        )
    with pytest.raises(ValueError, match="phase_budget_discharged"):
        predictive = build_pha_c_kinematic_proof_obligation(
            _record(),
            phase_drift_bound_rad=2.5e-3,
        )
        verify_pha_c_kinematic_proof_obligation(
            replace(predictive, phase_budget_discharged=False),
        )
    with pytest.raises(ValueError, match="acceptance_kinematic_equations_validated"):
        verify_pha_c_kinematic_proof_obligation(
            replace(obligation, acceptance_kinematic_equations_validated=False),
        )
    with pytest.raises(
        ValueError,
        match="acceptance_kinematic_summary_replay_tolerance",
    ):
        verify_pha_c_kinematic_proof_obligation(
            replace(
                obligation,
                acceptance_kinematic_summary_replay_tolerance=1.0e-9,
            ),
        )
    with pytest.raises(
        ValueError,
        match="acceptance_kinematic_summary_replay_tolerance_units",
    ):
        verify_pha_c_kinematic_proof_obligation(
            replace(
                obligation,
                acceptance_kinematic_summary_replay_tolerance_units=2,
            ),
        )
    with pytest.raises(
        ValueError,
        match="acceptance_kinematic_summary_replay_tolerance_limit_units",
    ):
        verify_pha_c_kinematic_proof_obligation(
            replace(
                obligation,
                acceptance_kinematic_summary_replay_tolerance_limit_units=2,
            ),
        )
    with pytest.raises(ValueError, match="acceptance_replay_certificate_discharged"):
        verify_pha_c_kinematic_proof_obligation(
            replace(obligation, acceptance_replay_certificate_discharged=False),
        )
    with pytest.raises(ValueError, match="acceptance_certificate_discharged"):
        verify_pha_c_kinematic_proof_obligation(
            replace(obligation, acceptance_certificate_discharged=False),
        )
    with pytest.raises(ValueError, match="gronwall_budget_units"):
        verify_pha_c_kinematic_proof_obligation(
            replace(
                obligation,
                gronwall_budget_units=obligation.gronwall_budget_units + 1,
            ),
        )
    with pytest.raises(ValueError, match="gronwall_budget_trace_sha256"):
        verify_pha_c_kinematic_proof_obligation(
            replace(obligation, gronwall_budget_trace_sha256="0" * 64),
        )
    with pytest.raises(ValueError, match="continuous_theorem"):
        verify_pha_c_kinematic_proof_obligation(
            replace(obligation, continuous_theorem="unchecked"),
        )
    with pytest.raises(ValueError, match="phase_theorem"):
        verify_pha_c_kinematic_proof_obligation(
            replace(obligation, phase_theorem="unchecked"),
        )
    with pytest.raises(ValueError, match="acceptance_certificate_theorem"):
        verify_pha_c_kinematic_proof_obligation(
            replace(obligation, acceptance_certificate_theorem="unchecked"),
        )
    with pytest.raises(ValueError, match="continuous_horizon_drive_bound_units"):
        verify_pha_c_kinematic_proof_obligation(
            replace(
                obligation,
                continuous_horizon_drive_bound_units=(
                    obligation.continuous_horizon_drive_bound_units + 1
                ),
            ),
        )
    with pytest.raises(ValueError, match="continuous_envelope_discharged"):
        verify_pha_c_kinematic_proof_obligation(
            replace(obligation, continuous_envelope_discharged=False),
        )
    with pytest.raises(ValueError, match="proof_obligations_discharged"):
        verify_pha_c_kinematic_proof_obligation(
            replace(obligation, proof_obligations_discharged=False),
        )
    with pytest.raises(ValueError, match="lean_theorem"):
        verify_pha_c_kinematic_proof_obligation(
            replace(obligation, lean_theorem="unchecked"),
        )


def test_kinematic_obligation_builder_fails_closed_on_invalid_controls() -> None:
    record = _record()

    with pytest.raises(ValueError, match="fixed_point_scale_m"):
        build_pha_c_kinematic_proof_obligation(record, fixed_point_scale_m=0.0)
    with pytest.raises(ValueError, match="fixed_point_scale_rad"):
        build_pha_c_kinematic_proof_obligation(record, fixed_point_scale_rad=np.inf)
    with pytest.raises(ValueError, match="fixed_point_time_scale_s"):
        build_pha_c_kinematic_proof_obligation(record, fixed_point_time_scale_s=0.0)
    with pytest.raises(ValueError, match="relative_velocity_step_bound_m"):
        build_pha_c_kinematic_proof_obligation(
            record,
            relative_velocity_step_bound_m=-1.0,
        )
    with pytest.raises(ValueError, match="coupling_residual_step_bound_m"):
        build_pha_c_kinematic_proof_obligation(
            record,
            coupling_residual_step_bound_m=np.inf,
        )
    with pytest.raises(ValueError, match="phase_drift_bound_rad"):
        build_pha_c_kinematic_proof_obligation(
            record,
            phase_drift_bound_rad=-1.0,
        )
    with pytest.raises(TypeError, match="PHACKinematicProofObligation"):
        verify_pha_c_kinematic_proof_obligation(object())  # type: ignore[arg-type]
