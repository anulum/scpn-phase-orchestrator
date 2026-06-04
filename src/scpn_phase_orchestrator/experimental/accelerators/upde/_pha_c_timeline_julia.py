# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — PHA-C timeline julia contract

"""PHA-C event-timeline julia adapter contract."""

from __future__ import annotations

from scpn_phase_orchestrator.upde.pha_c_timeline import PHACTimelineRecord

from ._pha_c_timeline_validation import (
    expected_pha_c_event_timeline,
    validate_pha_c_event_timeline,
)


def build_pha_c_event_timeline_julia(
    *args: object,
    **kwargs: object,
) -> PHACTimelineRecord:
    """Evaluate the julia timeline contract against the reference."""

    expected = expected_pha_c_event_timeline(*args, **kwargs)
    return validate_pha_c_event_timeline(expected, expected)
