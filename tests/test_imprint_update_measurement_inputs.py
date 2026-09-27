# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — imprint update measurement ingress

"""Verify original source-type refusal through imprint.update public calls."""

from __future__ import annotations

import numpy as np
import pytest

import scpn_phase_orchestrator.imprint.update as module
from scpn_phase_orchestrator.imprint.state import ImprintState
from tests.measurement_samples import source_alias


@pytest.mark.parametrize("kind", ["text", "duration", "object_duration"])
def test_original_measurement_types_are_refused(kind: str) -> None:
    """An otherwise valid numerical shape cannot erase text or duration units."""
    with pytest.raises(ValueError):
        module.ImprintModel(0.1, 1.0).update(
            ImprintState(np.zeros(3), 0.0), source_alias(kind, (3,)), 0.01
        )
