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

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scpn_phase_orchestrator.actuation.constraints import (
        ActionProjector as ActionProjector,
    )
    from scpn_phase_orchestrator.actuation.hdl_compiler import (
        KuramotoVerilogCompiler as KuramotoVerilogCompiler,
    )
    from scpn_phase_orchestrator.actuation.mapper import (
        ActuationMapper as ActuationMapper,
    )
    from scpn_phase_orchestrator.actuation.mapper import ControlAction as ControlAction

__all__ = [
    "ActionProjector",
    "ActuationMapper",
    "ControlAction",
    "KuramotoVerilogCompiler",
]

_EXPORTS = {
    "ActionProjector": (
        "scpn_phase_orchestrator.actuation.constraints",
        "ActionProjector",
    ),
    "ActuationMapper": (
        "scpn_phase_orchestrator.actuation.mapper",
        "ActuationMapper",
    ),
    "ControlAction": (
        "scpn_phase_orchestrator.actuation.mapper",
        "ControlAction",
    ),
    "KuramotoVerilogCompiler": (
        "scpn_phase_orchestrator.actuation.hdl_compiler",
        "KuramotoVerilogCompiler",
    ),
}


def __getattr__(name: str) -> object:
    """Load a public actuation symbol without importing unrelated subsystems."""
    export = _EXPORTS.get(name)
    if export is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = export
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return module globals and lazily exported public symbols."""
    return sorted({*globals(), *__all__})
