# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — PHA-C handoff accelerator validation

"""Shared validation for PHA-C handoff accelerator contracts."""

from __future__ import annotations

from typing import Any, cast

from scpn_phase_orchestrator.upde.pha_c_handoff import (
    PHACHandoffRecord,
    build_pha_c_handoff_record,
    verify_pha_c_handoff_record,
)

_NUMERIC_FIELDS = (
    "t",
    "phase_dispersion_rad",
    "spatial_dispersion_m",
    "phase_margin_rad",
    "spatial_margin_m",
    "phase_order_parameter",
    "distance_to_reference_max_m",
    "reference_phase",
    "reference_point",
    "phase_tol_rad",
    "spatial_tol_m",
    "tolerance_profile_multiplier",
)
_DISCRETE_FIELDS = (
    "oscillator_count",
    "phase_locked",
    "spatial_locked",
    "lock_achieved",
    "consecutive_lock_samples",
    "required_consecutive_samples",
    "tolerance_profile_name",
    "claim_boundary",
    "evidence_kind",
    "execution_disabled",
    "actuating",
    "phase_state_sha256",
    "position_state_sha256",
    "merge_report_sha256",
    "source_chain_sha256",
    "record_sha256",
)


def expected_pha_c_handoff_record(*args: object, **kwargs: object) -> PHACHandoffRecord:
    """Return the Python reference handoff after fail-closed validation."""

    return build_pha_c_handoff_record(*cast(Any, args), **cast(Any, kwargs))


def validate_pha_c_handoff_record(
    got: PHACHandoffRecord,
    expected: PHACHandoffRecord,
    *,
    tolerance: float = 1.0e-12,
) -> PHACHandoffRecord:
    """Validate an accelerator handoff against the Python reference contract."""

    verify_pha_c_handoff_record(got)
    verify_pha_c_handoff_record(expected)
    got_dict = got.to_dict()
    expected_dict = expected.to_dict()
    for field in _NUMERIC_FIELDS:
        error = abs(float(got_dict[field]) - float(expected_dict[field]))
        if error > tolerance:
            raise ValueError(f"PHA-C handoff field {field!r} diverged by {error}")
    for field in _DISCRETE_FIELDS:
        if got_dict[field] != expected_dict[field]:
            raise ValueError(f"PHA-C handoff field {field!r} diverged")
    return got
