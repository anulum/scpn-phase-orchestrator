# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Imprint subsystem

"""Memory-imprint state and update model.

The imprint subsystem tracks bounded per-oscillator exposure memory and applies
it to coupling, lag, and bifurcation parameters. Public entry points fail
closed on non-finite state, exposure, and timing inputs so imprint corruption
cannot silently propagate into the UPDE or supervisor paths.
"""

from __future__ import annotations

from scpn_phase_orchestrator.imprint.state import ImprintState
from scpn_phase_orchestrator.imprint.update import ImprintModel

__all__ = ["ImprintModel", "ImprintState"]
