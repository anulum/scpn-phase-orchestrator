# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — seal test artefacts the way the CLI seals them

"""Seal hand-built review artefacts with the CLI's own canonical record hash.

The lifecycle, remediation and scheduler commands verify that an artefact's
own hash covers its content. Fixtures that build artefacts by hand seal them
here, so a test exercising a structural check sees a properly sealed but
internally inconsistent artefact, the case those checks exist for.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from scpn_phase_orchestrator.runtime.cli._payloads import _record_hash


def seal(
    payload: Mapping[str, Any], field: str, *, blanked: bool = False
) -> dict[str, Any]:
    """Return ``payload`` with ``field`` set to its canonical self-seal.

    By default the seal covers the payload without ``field``. With
    ``blanked=True`` it covers the payload with ``field`` set to ``""``, the
    rule of producers that hash the record before filling its seal field.
    """
    body = {key: value for key, value in payload.items() if key != field}
    covered = {**body, field: ""} if blanked else body
    return {**body, field: _record_hash(covered)}
