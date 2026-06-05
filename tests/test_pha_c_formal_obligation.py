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
    build_pha_c_acceptance_record,
)
from scpn_phase_orchestrator.upde.pha_c_formal_obligation import (
    PHA_C_FORMAL_CERTIFICATE_PREDICATE,
    PHA_C_FORMAL_CERTIFICATE_THEOREM,
    PHA_C_FORMAL_LEAN_MODULE,
    PHA_C_FORMAL_OBLIGATION_CLAIM_BOUNDARY,
    PHA_C_FORMAL_OBLIGATION_SCHEMA,
    PHACKinematicProofObligation,
    build_pha_c_kinematic_proof_obligation,
    pha_c_kinematic_proof_obligation_to_dict,
    verify_pha_c_kinematic_proof_obligation,
)


def _record():
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
    assert obligation.lipschitz_step_gain_units == 0
    assert obligation.relative_velocity_step_bound_units == 0
    assert obligation.coupling_residual_step_bound_units == 0
    assert obligation.drive_bound_units == 0
    assert obligation.horizon_steps == record.step_count
    assert obligation.initial_tolerance_units == ceil(
        record.max_spatial_dispersion_m / obligation.fixed_point_scale_m,
    )
    assert obligation.merge_window_tolerance_units == ceil(
        record.spatial_tol_m / obligation.fixed_point_scale_m,
    )
    assert obligation.linear_budget_units == obligation.initial_tolerance_units
    assert obligation.window_budget_margin_units >= 0
    assert obligation.phase_margin_units >= 0
    assert obligation.observed_velocity_step_units == ceil(
        record.max_abs_velocity_m_per_s * record.dt / obligation.fixed_point_scale_m,
    )
    assert obligation.path_length_units == ceil(
        record.path_length_max_m / obligation.fixed_point_scale_m,
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


def test_kinematic_obligation_supports_predictive_relative_velocity_slack() -> None:
    record = _record()
    obligation = build_pha_c_kinematic_proof_obligation(
        record,
        relative_velocity_step_bound_m=1.0e-5,
    )

    expected_slack_units = ceil(1.0e-5 / obligation.fixed_point_scale_m)
    assert obligation.relative_velocity_step_bound_units == expected_slack_units
    assert obligation.drive_bound_units == expected_slack_units
    assert obligation.linear_budget_units == (
        obligation.initial_tolerance_units
        + obligation.horizon_steps * obligation.drive_bound_units
    )
    assert obligation.window_budget_margin_units >= 0
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
    with pytest.raises(ValueError, match="relative_velocity_step_bound_m"):
        build_pha_c_kinematic_proof_obligation(
            record,
            relative_velocity_step_bound_m=-1.0,
        )
    with pytest.raises(TypeError, match="PHACKinematicProofObligation"):
        verify_pha_c_kinematic_proof_obligation(object())  # type: ignore[arg-type]
