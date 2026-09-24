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


def seal(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    """Return ``payload`` with ``field`` set to its canonical self-seal."""
    body = {key: value for key, value in payload.items() if key != field}
    return {**body, field: _record_hash(body)}
