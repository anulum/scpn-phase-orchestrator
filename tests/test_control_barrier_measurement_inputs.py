# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — control barrier measurement ingress

"""Reject text and temporal aliases at the public measurement boundary."""

from __future__ import annotations

import numpy as np
import pytest

import scpn_phase_orchestrator.actuation.control_barrier as module
from tests.measurement_samples import source_alias


@pytest.mark.parametrize("kind", ["text", "duration", "object_duration"])
def test_original_measurement_types_are_refused(kind: str) -> None:
    """An otherwise valid numerical shape cannot erase text or duration units."""
    with pytest.raises(ValueError):
        module.NeuralBarrier(
            weights=(np.array([[1.0, 0.0, 0.0]]),), biases=(np.zeros(1),)
        ).value(source_alias(kind, (3,)))
