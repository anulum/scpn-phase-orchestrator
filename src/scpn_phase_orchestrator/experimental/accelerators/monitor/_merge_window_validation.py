# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — merge-window accelerator validation

"""Shared validation for PHA-C merge-window accelerator contracts."""

from __future__ import annotations

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

    return evaluate_merge_window(*args, **kwargs)


def validate_merge_window_report(
    got: MergeReport,
    expected: MergeReport,
    *,
    tolerance: float = 1.0e-12,
) -> MergeReport:
    """Validate an accelerator report against the Python reference contract."""

    got_dict = got.to_dict()
    expected_dict = expected.to_dict()
    for field in _NUMERIC_FIELDS:
        error = abs(float(got_dict[field]) - float(expected_dict[field]))
        if error > tolerance:
            raise ValueError(
                f"merge-window accelerator field {field!r} diverged by {error}"
            )
    for field in _BOOL_FIELDS:
        if bool(got_dict[field]) is not bool(expected_dict[field]):
            raise ValueError(f"merge-window accelerator field {field!r} diverged")
    if int(got_dict["consecutive_lock_samples"]) != int(
        expected_dict["consecutive_lock_samples"]
    ):
        raise ValueError("merge-window accelerator consecutive count diverged")
    return got
