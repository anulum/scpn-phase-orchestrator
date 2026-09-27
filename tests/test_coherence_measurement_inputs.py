# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — coherence measurement ingress

"""Refuse invalid alignment evidence at the public phase-lock operation."""

from __future__ import annotations

import pytest

from scpn_phase_orchestrator.monitor.coherence import CoherenceMonitor
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState
from tests.measurement_samples import source_alias


@pytest.mark.parametrize("kind", ["text", "duration", "object_duration"])
def test_original_measurement_types_are_refused(kind: str) -> None:
    """The public lock detector validates the original alignment buffer."""
    state = UPDEState(
        layers=[LayerState(R=0.5, psi=0.0), LayerState(R=0.5, psi=0.0)],
        cross_layer_alignment=source_alias(kind, (2, 2)),
        stability_proxy=0.5,
        regime_id="nominal",
    )
    with pytest.raises(ValueError):
        CoherenceMonitor([0], [1]).detect_phase_lock(state)
