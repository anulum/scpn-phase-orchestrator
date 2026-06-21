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
    """Build the registry execution-request artefact for ``plan`` under ``approval``.

    Parameters
    ----------
    plan : PluginExecutionPlan
        The execution plan to package.
    approval : PluginExecutionApproval
        The approval artefact authorising the plan.

    Returns
    -------
    object
        The deterministic, non-importing execution-request artefact.

    Raises
    ------
    click.ClickException
        If the registry does not expose ``build_plugin_execution_request``.
    """
    builder = getattr(plugin_registry, "build_plugin_execution_request", None)
    if not callable(builder):
        raise click.ClickException(
            "registry request builder not available: expected "
            "build_plugin_execution_request"
        )
    return builder(plan, approval)


@main.group("plugins")
def plugins_group() -> None:
    """Inspect extension plugin manifests."""
