# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — PHA-C timeline accelerator validation

"""Shared validation for PHA-C event-timeline accelerator contracts."""

from __future__ import annotations

from typing import Any, cast

from scpn_phase_orchestrator.upde.pha_c_timeline import (
    PHACTimelineRecord,
    build_pha_c_event_timeline,
    verify_pha_c_event_timeline,
)

_NUMERIC_FIELDS = (
    "start_time",
    "end_time",
    "duration_s",
    "first_lock_time",
    "max_phase_dispersion_rad",
    "max_spatial_dispersion_m",
    "min_phase_margin_rad",
    "min_spatial_margin_m",
    "min_phase_order_parameter",
    "max_distance_to_reference_m",
    "reference_phase",
    "reference_point",
    "phase_tol_rad",
    "spatial_tol_m",
    "tolerance_profile_multiplier",
)
_DISCRETE_FIELDS = (
    "sample_count",
    "oscillator_count",
    "first_lock_index",
    "first_lock_observed",
    "final_lock_achieved",
    "lock_sample_count",
    "phase_lock_sample_count",
    "spatial_lock_sample_count",
    "lock_loss_count",
    "reset_count",
    "max_consecutive_lock_samples",
    "tolerance_profile_name",
    "required_consecutive_samples",
    "claim_boundary",
    "evidence_kind",
    "execution_disabled",
    "actuating",
    "time_state_sha256",
    "sample_records_sha256",
    "transition_table_sha256",
    "timeline_sha256",
)


def expected_pha_c_event_timeline(
    *args: object,
    **kwargs: object,
) -> PHACTimelineRecord:
    """Return the Python reference timeline after fail-closed validation."""
    return build_pha_c_event_timeline(*cast(Any, args), **cast(Any, kwargs))


def validate_pha_c_event_timeline(
    got: PHACTimelineRecord,
    expected: PHACTimelineRecord,
    *,
    tolerance: float = 1.0e-12,
) -> PHACTimelineRecord:
    """Validate an accelerator timeline against the Python reference contract."""
    verify_pha_c_event_timeline(got)
    verify_pha_c_event_timeline(expected)
    got_dict = got.to_dict()
    expected_dict = expected.to_dict()
    for field in _NUMERIC_FIELDS:
        error = abs(float(got_dict[field]) - float(expected_dict[field]))
        if error > tolerance:
            raise ValueError(f"PHA-C timeline field {field!r} diverged by {error}")
    for field in _DISCRETE_FIELDS:
        if got_dict[field] != expected_dict[field]:
            raise ValueError(f"PHA-C timeline field {field!r} diverged")
    return got
