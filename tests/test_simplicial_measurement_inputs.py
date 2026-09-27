# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — simplicial measurement ingress

"""Verify original source-type refusal through upde.simplicial public calls."""

from __future__ import annotations

import numpy as np
import pytest

import scpn_phase_orchestrator.upde.simplicial as module
from tests.measurement_samples import source_alias


@pytest.mark.parametrize("kind", ["text", "duration", "object_duration"])
def test_original_measurement_types_are_refused(kind: str) -> None:
    """An otherwise valid numerical shape cannot erase text or duration units."""
    with pytest.raises(ValueError):
        module.SimplicialEngine(3, 0.01).step(
            source_alias(kind, (3,)),
            np.ones(3),
            np.zeros((3, 3)),
            0.0,
            0.0,
            np.zeros((3, 3)),
        )
