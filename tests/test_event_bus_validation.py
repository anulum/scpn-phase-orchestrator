# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Event bus validation contracts

"""Validation contracts for EventBus retention bounds."""

from __future__ import annotations

import pytest

from scpn_phase_orchestrator.supervisor.events import EventBus


class TestEventBusValidation:
    def test_rejects_zero_maxlen(self) -> None:
        with pytest.raises(ValueError, match="maxlen must be >= 1"):
            EventBus(maxlen=0)

    def test_rejects_negative_maxlen(self) -> None:
        with pytest.raises(ValueError, match="maxlen must be >= 1"):
            EventBus(maxlen=-10)
