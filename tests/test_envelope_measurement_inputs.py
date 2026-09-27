# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — envelope measurement ingress

"""Refuse measurement aliases through both public envelope operations."""

from __future__ import annotations

import pytest

from scpn_phase_orchestrator.upde.envelope import (
    envelope_modulation_depth,
    extract_envelope,
)
from tests.measurement_samples import source_alias


@pytest.mark.parametrize("kind", ["text", "duration", "object_duration"])
def test_original_measurement_types_are_refused(kind: str) -> None:
    """Neither envelope computation nor modulation erases measurement units."""
    value = source_alias(kind, (16,))
    with pytest.raises(ValueError):
        extract_envelope(value, window=4)
    with pytest.raises(ValueError):
        envelope_modulation_depth(value)
