# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Actuation layer

"""Actuation boundary for reviewed SPO control commands.

The package exposes the value/rate projector and the binding-spec mapper that
turn supervisor proposals into actuator-specific command dictionaries. It does
not open hardware transports; downstream connectors consume the mapped command
records after policy, value, and safety validation have already run.
"""

from __future__ import annotations

from scpn_phase_orchestrator.actuation.constraints import ActionProjector
from scpn_phase_orchestrator.actuation.hdl_compiler import KuramotoVerilogCompiler
from scpn_phase_orchestrator.actuation.mapper import ActuationMapper, ControlAction

__all__ = [
    "ActionProjector",
    "ActuationMapper",
    "ControlAction",
    "KuramotoVerilogCompiler",
]
