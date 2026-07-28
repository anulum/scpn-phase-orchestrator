# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — UPDE backend validation compatibility facade

"""Compatibility re-exports for the core-owned UPDE dispatch contracts."""

from scpn_phase_orchestrator.upde._engine_validation import (
    validate_upde_backend_inputs,
    validate_upde_backend_output,
    validate_upde_schedule_backend_inputs,
)

__all__ = [
    "validate_upde_backend_inputs",
    "validate_upde_backend_output",
    "validate_upde_schedule_backend_inputs",
]
