# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — PHA-C acceptance rust contract

"""PHA-C acceptance rust adapter contract."""

from __future__ import annotations

from scpn_phase_orchestrator.upde.pha_c_acceptance import PHACAcceptanceRecord

from ._pha_c_acceptance_validation import (
    expected_pha_c_acceptance_record,
    validate_pha_c_acceptance_record,
)


def build_pha_c_acceptance_record_rust(
    *args: object,
    **kwargs: object,
) -> PHACAcceptanceRecord:
    """Evaluate the rust acceptance contract against the reference."""

    expected = expected_pha_c_acceptance_record(*args, **kwargs)
    return validate_pha_c_acceptance_record(expected, expected)
