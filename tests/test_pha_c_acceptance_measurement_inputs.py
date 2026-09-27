# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — PHA-C acceptance measurement ingress

"""Refuse aliases at each input family of the public acceptance builder."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.pha_c_acceptance import build_pha_c_acceptance_record
from tests.measurement_samples import source_alias


@pytest.mark.parametrize("kind", ["text", "duration", "object_duration"])
@pytest.mark.parametrize("field", ["phases", "positions", "omega", "knm", "velocity"])
def test_original_measurement_types_are_refused(kind: str, field: str) -> None:
    """Valid surrounding schedules cannot erase the original source units."""
    phases = np.zeros(3)
    positions = np.zeros(3)
    omega = np.ones((3, 3))
    knm = np.zeros((3, 3))
    velocity = np.zeros((3, 3))
    if field == "phases":
        phases = source_alias(kind, (3,))
    elif field == "positions":
        positions = source_alias(kind, (3,))
    elif field == "omega":
        omega = source_alias(kind, (3, 3))
    elif field == "knm":
        knm = source_alias(kind, (3, 3))
    elif field == "velocity":
        velocity = source_alias(kind, (3, 3))
    with pytest.raises(ValueError):
        build_pha_c_acceptance_record(phases, positions, omega, knm, velocity)
