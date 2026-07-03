# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — PHA-C timeline accelerator validation

"""Shared validation for PHA-C event-timeline accelerator contracts."""

from __future__ import annotations

from math import isfinite
from numbers import Integral, Real
from typing import Any, cast

import numpy as np

from scpn_phase_orchestrator.upde.pha_c_timeline import (
    PHACTimelineRecord,
    build_pha_c_event_timeline,
    verify_pha_c_event_timeline,
)

from ._validation_common import validate_non_negative_tolerance

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
_INT_FIELDS = (
    "sample_count",
    "oscillator_count",
    "first_lock_index",
    "lock_sample_count",
    "phase_lock_sample_count",
    "spatial_lock_sample_count",
    "lock_loss_count",
    "reset_count",
    "max_consecutive_lock_samples",
    "required_consecutive_samples",
)
_BOOL_FIELDS = (
    "first_lock_observed",
    "final_lock_achieved",
    "execution_disabled",
    "actuating",
)
_STRING_FIELDS = (
    "tolerance_profile_name",
    "claim_boundary",
    "evidence_kind",
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


def _validate_real_record_field(record: PHACTimelineRecord, field: str) -> float:
    """Return a record field as a finite non-boolean real scalar."""
    value = getattr(record, field)
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (Real, np.floating, np.integer),
    ):
        raise ValueError(f"PHA-C timeline field {field!r} must be a finite real scalar")
    parsed = float(value)
    if not isfinite(parsed):
        raise ValueError(f"PHA-C timeline field {field!r} must be finite")
    return parsed


def _validate_int_record_field(record: PHACTimelineRecord, field: str) -> int:
    """Return a record field as an integer after rejecting boolean aliases."""
    value = getattr(record, field)
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (Integral, np.integer),
    ):
        raise ValueError(f"PHA-C timeline field {field!r} must be an integer")
    return int(value)


def _validate_bool_record_field(record: PHACTimelineRecord, field: str) -> bool:
    """Return a record field after confirming it is a plain ``bool``."""
    value = getattr(record, field)
    if type(value) is not bool:
        raise ValueError(f"PHA-C timeline field {field!r} must be bool")
    return value


def _validate_string_record_field(record: PHACTimelineRecord, field: str) -> str:
    """Return a record field after confirming it is a string."""
    value = getattr(record, field)
    if not isinstance(value, str):
        raise ValueError(f"PHA-C timeline field {field!r} must be a string")
    return value


def pha_c_timeline_record_max_abs_error(
    got: PHACTimelineRecord,
    expected: PHACTimelineRecord,
) -> float:
    """Return strict maximum field error for PHA-C timeline parity."""
    numeric_error = max(
        abs(
            _validate_real_record_field(got, field)
            - _validate_real_record_field(expected, field)
        )
        for field in _NUMERIC_FIELDS
    )
    int_error = max(
        int(
            _validate_int_record_field(got, field)
            != _validate_int_record_field(expected, field)
        )
        for field in _INT_FIELDS
    )
    bool_error = max(
        int(
            _validate_bool_record_field(got, field)
            is not _validate_bool_record_field(expected, field)
        )
        for field in _BOOL_FIELDS
    )
    string_error = max(
        int(
            _validate_string_record_field(got, field)
            != _validate_string_record_field(expected, field)
        )
        for field in _STRING_FIELDS
    )
    verify_pha_c_event_timeline(got)
    verify_pha_c_event_timeline(expected)
    return max(numeric_error, float(int_error), float(bool_error), float(string_error))


def validate_pha_c_event_timeline(
    got: PHACTimelineRecord,
    expected: PHACTimelineRecord,
    *,
    tolerance: object = 1.0e-12,
) -> PHACTimelineRecord:
    """Validate an accelerator timeline against the Python reference contract."""
    tolerance_f = validate_non_negative_tolerance(tolerance)
    for field in _NUMERIC_FIELDS:
        error = abs(
            _validate_real_record_field(got, field)
            - _validate_real_record_field(expected, field)
        )
        if error > tolerance_f:
            raise ValueError(f"PHA-C timeline field {field!r} diverged by {error}")
    for field in _INT_FIELDS:
        if _validate_int_record_field(got, field) != _validate_int_record_field(
            expected,
            field,
        ):
            raise ValueError(f"PHA-C timeline field {field!r} diverged")
    for field in _BOOL_FIELDS:
        if _validate_bool_record_field(got, field) is not _validate_bool_record_field(
            expected,
            field,
        ):
            raise ValueError(f"PHA-C timeline field {field!r} diverged")
    for field in _STRING_FIELDS:
        if _validate_string_record_field(got, field) != _validate_string_record_field(
            expected,
            field,
        ):
            raise ValueError(f"PHA-C timeline field {field!r} diverged")
    verify_pha_c_event_timeline(got)
    verify_pha_c_event_timeline(expected)
    return got
