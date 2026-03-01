# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations


class PetriNetAdapter:
    """Adapter for Petri net regime FSM integration.

    Interface contract:
        - accepts a RegimeManager and a set of Place/Transition definitions
        - fires transitions based on UPDEState token markings
        - returns the active Regime after firing
    """

    def __init__(self):
        raise NotImplementedError("Petri net FSM planned for v0.4 (see ROADMAP.md)")
