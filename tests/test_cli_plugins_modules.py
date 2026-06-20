# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI plugins package registration tests

"""Registration contract for the plugins CLI command package.

The plugins command group is split across concern modules (execution, storage,
lifecycle, remediation, scheduler, scheduler_control, revocation); importing the
package must register every subcommand under a single ``plugins`` group. These
assertions pin that contract so a missing module import or a renamed command is
caught directly rather than only through a command-invocation test.
"""

from __future__ import annotations

import scpn_phase_orchestrator.runtime.cli.plugins.execution as execution
import scpn_phase_orchestrator.runtime.cli.plugins.lifecycle as lifecycle
import scpn_phase_orchestrator.runtime.cli.plugins.remediation as remediation
import scpn_phase_orchestrator.runtime.cli.plugins.revocation as revocation
import scpn_phase_orchestrator.runtime.cli.plugins.scheduler as scheduler
import scpn_phase_orchestrator.runtime.cli.plugins.scheduler_control as sched_control
import scpn_phase_orchestrator.runtime.cli.plugins.storage as storage
from scpn_phase_orchestrator.runtime.cli.plugins._group import (
    _build_plugin_execution_request,
    plugins_group,
)

_EXPECTED_COMMANDS = {
    "catalog",
    "plan-execution",
    "approve-execution-plan",
    "request-execution",
    "persist-execution-request",
    "storage-adapter-manifest",
    "lifecycle-status",
    "lifecycle-summary",
    "lifecycle-policy-report",
    "lifecycle-renewal-queue",
    "lifecycle-multistore-dashboard",
    "lifecycle-multistore-drilldown",
    "lifecycle-remediation-orchestration",
    "lifecycle-remediation-action-status",
    "lifecycle-remediation-execution-dashboard",
    "lifecycle-remediation-deployment-handoff",
    "lifecycle-remediation-scheduler-queue",
    "lifecycle-remediation-scheduler-telemetry",
    "lifecycle-remediation-scheduler-adapter-handoff",
    "lifecycle-remediation-scheduler-acknowledgement",
    "lifecycle-remediation-scheduler-acknowledgement-replay",
    "lifecycle-remediation-scheduler-execution-dashboard",
    "lifecycle-remediation-scheduler-control-plan",
    "lifecycle-remediation-scheduler-runbook",
    "lifecycle-remediation-scheduler-automation-profile",
    "lifecycle-remediation-scheduler-acknowledgement-capture",
    "lifecycle-remediation-scheduler-retry-profile",
    "lifecycle-remediation-scheduler-retry-orchestration",
    "revoke-execution-request",
    "revocation-list",
}

# Reference the concern modules so import linkage is explicit, not incidental.
_CONCERN_MODULES = (
    execution,
    storage,
    lifecycle,
    remediation,
    scheduler,
    sched_control,
    revocation,
)


def test_every_concern_module_is_imported() -> None:
    assert all(module is not None for module in _CONCERN_MODULES)


def test_plugins_group_registers_all_subcommands() -> None:
    assert set(plugins_group.commands) == _EXPECTED_COMMANDS


def test_execution_request_builder_is_exposed_on_group_module() -> None:
    assert callable(_build_plugin_execution_request)
