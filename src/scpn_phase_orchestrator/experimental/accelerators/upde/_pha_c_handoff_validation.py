# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — PHA-C handoff accelerator validation

"""Shared validation for PHA-C handoff accelerator contracts."""

from __future__ import annotations

from math import isfinite
from numbers import Integral, Real
from typing import Any, cast

import numpy as np

from scpn_phase_orchestrator.upde._validation_common import (
    validate_non_negative_tolerance,
)
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
_INT_FIELDS = (
    "oscillator_count",
    "consecutive_lock_samples",
    "required_consecutive_samples",
)
_BOOL_FIELDS = (
    "phase_locked",
    "spatial_locked",
    "lock_achieved",
    "execution_disabled",
    "actuating",
)
_STRING_FIELDS = (
    "tolerance_profile_name",
    "claim_boundary",
    "evidence_kind",
    "phase_state_sha256",
    "position_state_sha256",
    "merge_report_sha256",
    "source_chain_sha256",
    "record_sha256",
)


def expected_pha_c_handoff_record(*args: object, **kwargs: object) -> PHACHandoffRecord:
    """Return the Python reference handoff after fail-closed validation."""
    return build_pha_c_handoff_record(*cast(Any, args), **cast(Any, kwargs))


def _validate_real_record_field(record: PHACHandoffRecord, field: str) -> float:
    """Return a record field as a finite non-boolean real scalar."""
    value = getattr(record, field)
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (Real, np.floating, np.integer),
    ):
        raise ValueError(f"PHA-C handoff field {field!r} must be a finite real scalar")
    parsed = float(value)
    if not isfinite(parsed):
        raise ValueError(f"PHA-C handoff field {field!r} must be finite")
    return parsed


def _validate_int_record_field(record: PHACHandoffRecord, field: str) -> int:
    """Return a record field as an integer after rejecting boolean aliases."""
    value = getattr(record, field)
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (Integral, np.integer),
    ):
        raise ValueError(f"PHA-C handoff field {field!r} must be an integer")
    return int(value)


def _validate_bool_record_field(record: PHACHandoffRecord, field: str) -> bool:
    """Return a record field after confirming it is a plain ``bool``."""
    value = getattr(record, field)
    if type(value) is not bool:
        raise ValueError(f"PHA-C handoff field {field!r} must be bool")
    return value


def _validate_string_record_field(record: PHACHandoffRecord, field: str) -> str:
    """Return a record field after confirming it is a string."""
    value = getattr(record, field)
    if not isinstance(value, str):
        raise ValueError(f"PHA-C handoff field {field!r} must be a string")
    return value


def pha_c_handoff_record_max_abs_error(
    got: PHACHandoffRecord,
    expected: PHACHandoffRecord,
) -> float:
    """Return strict maximum field error for PHA-C handoff parity."""
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
    verify_pha_c_handoff_record(got)
    verify_pha_c_handoff_record(expected)
    return max(numeric_error, float(int_error), float(bool_error), float(string_error))


def validate_pha_c_handoff_record(
    got: PHACHandoffRecord,
    expected: PHACHandoffRecord,
    *,
    tolerance: object = 1.0e-12,
) -> PHACHandoffRecord:
    """Validate an accelerator handoff against the Python reference contract."""
    tolerance_f = validate_non_negative_tolerance(tolerance)
    for field in _NUMERIC_FIELDS:
        error = abs(
            _validate_real_record_field(got, field)
            - _validate_real_record_field(expected, field)
        )
        if error > tolerance_f:
            raise ValueError(f"PHA-C handoff field {field!r} diverged by {error}")
    for field in _INT_FIELDS:
        if _validate_int_record_field(got, field) != _validate_int_record_field(
            expected,
            field,
        ):
            raise ValueError(f"PHA-C handoff field {field!r} diverged")
    for field in _BOOL_FIELDS:
        if _validate_bool_record_field(got, field) is not _validate_bool_record_field(
            expected,
            field,
        ):
            raise ValueError(f"PHA-C handoff field {field!r} diverged")
    for field in _STRING_FIELDS:
        if _validate_string_record_field(got, field) != _validate_string_record_field(
            expected,
            field,
        ):
            raise ValueError(f"PHA-C handoff field {field!r} diverged")
    verify_pha_c_handoff_record(got)
    verify_pha_c_handoff_record(expected)
    return got
