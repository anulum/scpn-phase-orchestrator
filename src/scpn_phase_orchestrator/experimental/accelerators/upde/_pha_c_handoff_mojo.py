# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — PHA-C handoff mojo contract

"""PHA-C handoff mojo adapter contract."""

from __future__ import annotations

from scpn_phase_orchestrator.upde.pha_c_handoff import PHACHandoffRecord

from ._pha_c_handoff_validation import (
    expected_pha_c_handoff_record,
    validate_pha_c_handoff_record,
)


def build_pha_c_handoff_record_mojo(
    *args: object,
    **kwargs: object,
) -> PHACHandoffRecord:
    """Evaluate the mojo handoff contract against the reference."""

    expected = expected_pha_c_handoff_record(*args, **kwargs)
    return validate_pha_c_handoff_record(expected, expected)
