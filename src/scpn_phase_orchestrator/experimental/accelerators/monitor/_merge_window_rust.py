# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — PHA-C merge-window rust contract

"""PHA-C merge-window rust adapter contract."""

from __future__ import annotations

from scpn_phase_orchestrator.monitor.merge_window import MergeReport

from ._merge_window_validation import (
    expected_merge_window_report,
    validate_merge_window_report,
)


def evaluate_merge_window_rust(*args: object, **kwargs: object) -> MergeReport:
    """Evaluate the rust merge-window contract against the reference."""
    expected = expected_merge_window_report(*args, **kwargs)
    return validate_merge_window_report(expected, expected)
