# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — External driver registry

"""Public registry for external reference-phase drivers.

Drivers provide deterministic `Psi` reference signals for Physical,
Informational, and Symbolic channels. They validate construction parameters and
compute-time inputs before returning scalar or vector phase-drive values to
engine or CLI paths.
"""

from __future__ import annotations

from scpn_phase_orchestrator.drivers.psi_informational import InformationalDriver
from scpn_phase_orchestrator.drivers.psi_physical import PhysicalDriver
from scpn_phase_orchestrator.drivers.psi_symbolic import SymbolicDriver

__all__ = ["PhysicalDriver", "InformationalDriver", "SymbolicDriver"]
