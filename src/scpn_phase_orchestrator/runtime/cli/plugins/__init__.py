# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI plugins command package

"""Command-line entry point for validation, replay, export, and review workflows.

The CLI wraps public SPO APIs behind explicit commands for binding validation,
inspection, auto-binding proposals, coupling estimation, formal export, replay,
plugin catalogs, scaffolding, and selected runtime utilities. Commands validate
local inputs and emit text or JSON review artifacts; they do not push commits,
start network services, or perform live actuation unless an explicit subcommand
is invoked for that runtime path.
"""

from __future__ import annotations

from scpn_phase_orchestrator.runtime.cli.plugins import _group as _group
from scpn_phase_orchestrator.runtime.cli.plugins import execution as execution
from scpn_phase_orchestrator.runtime.cli.plugins import lifecycle as lifecycle
from scpn_phase_orchestrator.runtime.cli.plugins import remediation as remediation
from scpn_phase_orchestrator.runtime.cli.plugins import revocation as revocation
from scpn_phase_orchestrator.runtime.cli.plugins import scheduler as scheduler
from scpn_phase_orchestrator.runtime.cli.plugins import (
    scheduler_control as scheduler_control,
)
from scpn_phase_orchestrator.runtime.cli.plugins import storage as storage
from scpn_phase_orchestrator.runtime.cli.plugins._group import (
    plugins_group as plugins_group,
)

__all__ = ["plugins_group"]
