# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI plugins command group and execution-request builder

"""Command-line entry point for validation, replay, export, and review workflows.

The CLI wraps public SPO APIs behind explicit commands for binding validation,
inspection, auto-binding proposals, coupling estimation, formal export, replay,
plugin catalogs, scaffolding, and selected runtime utilities. Commands validate
local inputs and emit text or JSON review artifacts; they do not push commits,
start network services, or perform live actuation unless an explicit subcommand
is invoked for that runtime path.
"""

from __future__ import annotations

import click

from scpn_phase_orchestrator import plugins as plugin_api
from scpn_phase_orchestrator.plugins import (
    PluginExecutionApproval,
    PluginExecutionPlan,
)
from scpn_phase_orchestrator.plugins import registry as plugin_registry
from scpn_phase_orchestrator.runtime.cli._app import main


def _build_plugin_execution_request(
    plan: PluginExecutionPlan,
    approval: PluginExecutionApproval,
) -> object:
    builder_candidates = (
        "build_plugin_execution_request",
        "build_plugin_execution_request_from_approval",
        "build_plugin_execution_request_from_plan_and_approval",
    )
    for name in builder_candidates:
        for module in (plugin_registry, plugin_api):
            candidate = getattr(module, name, None)
            if not callable(candidate):
                continue
            try:
                return candidate(plan, approval)
            except TypeError:
                pass
            try:
                return candidate(plan=plan, approval=approval)
            except TypeError:
                pass
            try:
                return candidate(plan=plan, approved_execution=approval)
            except TypeError:
                pass
            try:
                return candidate(plan=plan, approval_record=approval)
            except TypeError:
                pass

    raise click.ClickException(
        "registry request builder not available: expected "
        "build_plugin_execution_request"
    )


@main.group("plugins")
def plugins_group() -> None:
    """Inspect extension plugin manifests."""
