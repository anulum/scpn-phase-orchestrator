# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("scpn-phase-orchestrator")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

from scpn_phase_orchestrator.binding.types import BindingSpec
from scpn_phase_orchestrator.oscillators.base import PhaseExtractor, PhaseState
from scpn_phase_orchestrator.supervisor import ControlAction
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

__all__ = [
    "BindingSpec",
    "ControlAction",
    "PhaseExtractor",
    "PhaseState",
    "StuartLandauEngine",
    "UPDEEngine",
]
