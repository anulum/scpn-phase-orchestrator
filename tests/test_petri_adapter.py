from __future__ import annotations

import pytest

from scpn_phase_orchestrator.supervisor.petri_adapter import PetriNetAdapter


def test_petri_adapter_not_implemented():
    with pytest.raises(NotImplementedError, match="v0.4"):
        PetriNetAdapter()
