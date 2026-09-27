# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Plasma bridge measurement ingress

"""Refuse original aliases when importing a plasma coupling specification."""

from __future__ import annotations

import pytest

from scpn_phase_orchestrator.adapters.plasma_control_bridge import PlasmaControlBridge
from tests.measurement_samples import source_alias


@pytest.mark.parametrize("kind", ["text", "duration", "object_duration"])
def test_original_measurement_types_are_refused(kind: str) -> None:
    """The public matrix importer refuses source units before replication."""
    with pytest.raises(ValueError):
        PlasmaControlBridge(n_layers=3).import_knm_spec(source_alias(kind, (3, 3)))
