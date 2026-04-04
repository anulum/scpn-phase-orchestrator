# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Oscillator registry

from __future__ import annotations

from scpn_phase_orchestrator.oscillators.base import PhaseExtractor, PhaseState
from scpn_phase_orchestrator.oscillators.informational import InformationalExtractor
from scpn_phase_orchestrator.oscillators.init_phases import extract_initial_phases
from scpn_phase_orchestrator.oscillators.physical import PhysicalExtractor
from scpn_phase_orchestrator.oscillators.quality import PhaseQualityScorer
from scpn_phase_orchestrator.oscillators.symbolic import SymbolicExtractor

__all__ = [
    "InformationalExtractor",
    "PhaseExtractor",
    "PhaseQualityScorer",
    "PhaseState",
    "PhysicalExtractor",
    "SymbolicExtractor",
    "extract_initial_phases",
]
