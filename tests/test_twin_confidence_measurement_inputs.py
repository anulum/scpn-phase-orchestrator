# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — twin_confidence measurement ingress

"""Refuse measurement aliases through public and direct language operations."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

import scpn_phase_orchestrator.monitor.twin_confidence as module
from tests.measurement_samples import source_alias


@pytest.mark.parametrize("kind", ["text", "duration", "object_duration"])
def test_original_measurement_types_are_refused(kind: str) -> None:
    """The public operation refuses the original units before numeric conversion."""
    with pytest.raises(ValueError):
        module.phase_order_divergence(
            source_alias(kind, (3,)), np.zeros(3), np.zeros(4), np.zeros(4)
        )


@pytest.mark.parametrize("backend", ["go", "julia", "mojo"])
@pytest.mark.parametrize("kind", ["text", "duration", "object_duration"])
def test_direct_backend_refuses_original_measurement_types(
    backend: str, kind: str
) -> None:
    """All real language bridges preserve the measurement source contract."""
    bridge = importlib.import_module(
        f"scpn_phase_orchestrator.experimental.accelerators.monitor._twin_confidence_{backend}"
    )
    operation = getattr(bridge, "twin_divergence_" + backend)
    with pytest.raises(ValueError):
        operation(
            source_alias(kind, (3,)), np.zeros(3), np.zeros(4), np.zeros(4), 3, 4, 4
        )
