# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — hybrid_order measurement ingress

"""Reject unit aliases in classical phases while retaining quantum state types."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor.hybrid_order import (
    compute_hybrid_entanglement_order_parameter,
)
from tests.measurement_samples import source_alias


@pytest.mark.parametrize("kind", ["text", "duration", "object_duration"])
def test_original_measurement_types_are_refused(kind: str) -> None:
    """A valid Bell state does not legitimise invalid classical measurements."""
    state = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    with pytest.raises(ValueError):
        compute_hybrid_entanglement_order_parameter(source_alias(kind, (3,)), state)
