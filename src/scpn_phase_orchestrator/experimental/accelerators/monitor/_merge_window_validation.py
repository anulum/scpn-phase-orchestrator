# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — merge-window accelerator validation

"""Shared validation for PHA-C merge-window accelerator contracts."""

from __future__ import annotations

from math import isfinite
from numbers import Integral, Real
from typing import Any, cast

import numpy as np

from scpn_phase_orchestrator.monitor.merge_window import (
    MergeReport,
    evaluate_merge_window,
)

_NUMERIC_FIELDS = (
    "t",
    "phase_dispersion_rad",
    "spatial_dispersion_m",
    "phase_margin_rad",
    "spatial_margin_m",
)
_BOOL_FIELDS = ("phase_locked", "spatial_locked", "lock_achieved")


def expected_merge_window_report(*args: object, **kwargs: object) -> MergeReport:
    """Return the Python reference report after fail-closed input validation."""
    return evaluate_merge_window(*cast(Any, args), **cast(Any, kwargs))


def _validate_real_report_field(report: MergeReport, field: str) -> float:
    """Return a report field as a finite non-boolean real scalar."""
    value = getattr(report, field)
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (Real, np.floating, np.integer),
    ):
        raise ValueError(
            f"merge-window accelerator field {field!r} must be a finite real scalar"
        )
    parsed = float(value)
    if not isfinite(parsed):
        raise ValueError(f"merge-window accelerator field {field!r} must be finite")
    return parsed


def _validate_bool_report_field(report: MergeReport, field: str) -> bool:
    """Return a report field after confirming it is a plain ``bool``."""
    value = getattr(report, field)
    if type(value) is not bool:
        raise ValueError(f"merge-window accelerator field {field!r} must be bool")
    return value


def _validate_consecutive_count(report: MergeReport) -> int:
    """Return the report's consecutive sample count as a non-negative integer."""
    value = report.consecutive_lock_samples
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (Integral, np.integer),
    ):
        raise ValueError(
            "merge-window accelerator field 'consecutive_lock_samples' "
            "must be a non-negative integer"
        )
    parsed = int(value)
    if parsed < 0:
        raise ValueError(
            "merge-window accelerator field 'consecutive_lock_samples' "
            "must be non-negative"
        )
    return parsed


def _validate_tolerance(value: object) -> float:
    """Return a finite non-negative comparison tolerance."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (Real, np.floating, np.integer),
    ):
        raise ValueError("merge-window accelerator tolerance must be a finite scalar")
    parsed = float(value)
    if not isfinite(parsed):
        raise ValueError("merge-window accelerator tolerance must be finite")
    if parsed < 0.0:
        raise ValueError("merge-window accelerator tolerance must be non-negative")
    return parsed


def validate_merge_window_report(
    got: MergeReport,
    expected: MergeReport,
    *,
    tolerance: float = 1.0e-12,
) -> MergeReport:
    """Validate an accelerator report against the Python reference contract."""
    tolerance_value = _validate_tolerance(tolerance)
    for field in _NUMERIC_FIELDS:
        got_value = _validate_real_report_field(got, field)
        expected_value = _validate_real_report_field(expected, field)
        error = abs(got_value - expected_value)
        if error > tolerance_value:
            raise ValueError(
                f"merge-window accelerator field {field!r} diverged by {error}"
            )
    for field in _BOOL_FIELDS:
        if _validate_bool_report_field(got, field) is not _validate_bool_report_field(
            expected,
            field,
        ):
            raise ValueError(f"merge-window accelerator field {field!r} diverged")
    if _validate_consecutive_count(got) != _validate_consecutive_count(expected):
        raise ValueError("merge-window accelerator consecutive count diverged")
    return got
