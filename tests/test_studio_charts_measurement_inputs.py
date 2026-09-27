# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — studio_charts measurement ingress

"""Validate original pairwise evidence when producing the Studio panel."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor.information_integration import (
    integrated_information,
)
from scpn_phase_orchestrator.studio.ui_helpers.charts import (
    build_integrated_information_panel,
)
from tests.measurement_samples import source_alias


@pytest.mark.parametrize("kind", ["text", "duration", "object_duration"])
def test_original_measurement_types_are_refused(kind: str) -> None:
    """A genuine monitor record cannot acquire unit-bearing pairwise evidence."""
    result = integrated_information(np.zeros((3, 32), dtype=np.float64))
    record = result.to_audit_record()
    record["pairwise_mi"] = source_alias(kind, (3, 3))
    with pytest.raises(ValueError):
        build_integrated_information_panel([record])
