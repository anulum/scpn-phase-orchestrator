# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — lifecycle/remediation/scheduler artefacts verify their seals

"""An edited review artefact must not travel down the chain under its old hash.

The lifecycle, remediation and scheduler commands seal every artefact with
``payload[field] = _record_hash(payload)``; the consumers checked only that
the field was 64 hex characters. An operator- or attacker-edited dashboard
row (``blocked`` -> ``completed``) or a flipped ``overdue`` flag became the
basis of the next plan under the stale seal.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click
import pytest

from scpn_phase_orchestrator.runtime.cli._payloads import (
    _load_lifecycle_multistore_drilldown_payload,
    _load_lifecycle_remediation_action_status_payload,
    _load_lifecycle_remediation_deployment_handoff_payload,
    _load_lifecycle_remediation_execution_dashboard_payload,
    _load_lifecycle_remediation_plan_payload,
    _load_lifecycle_remediation_scheduler_acknowledgement_payload,
    _load_lifecycle_remediation_scheduler_adapter_handoff_payload,
    _load_lifecycle_remediation_scheduler_queue_payload,
    _load_lifecycle_remediation_scheduler_telemetry_payload,
)
from tests import test_cli_remediation_loaders as remediation
from tests import test_cli_scheduler_contracts as contracts
from tests import test_cli_scheduler_loaders as scheduler
from tests.scheduler_control_fixtures import (
    _assert_fails,
    _dashboard_payload,
    _invoke_json,
    _write_json,
)
from tests.sealing import seal

_LOADERS: list[tuple[str, Callable[..., Any], Callable[[], dict[str, Any]], str]] = [
    (
        "drilldown",
        _load_lifecycle_multistore_drilldown_payload,
        remediation._drilldown,
        "drilldown_hash",
    ),
    (
        "remediation plan",
        _load_lifecycle_remediation_plan_payload,
        remediation._remediation_plan,
        "plan_hash",
    ),
    (
        "action status",
        _load_lifecycle_remediation_action_status_payload,
        remediation._action_status,
        "status_hash",
    ),
    (
        "execution dashboard",
        _load_lifecycle_remediation_execution_dashboard_payload,
        scheduler._dashboard,
        "execution_hash",
    ),
    (
        "deployment handoff",
        _load_lifecycle_remediation_deployment_handoff_payload,
        scheduler._handoff,
        "handoff_hash",
    ),
    (
        "scheduler queue",
        _load_lifecycle_remediation_scheduler_queue_payload,
        scheduler._queue,
        "scheduler_hash",
    ),
    (
        "scheduler telemetry",
        _load_lifecycle_remediation_scheduler_telemetry_payload,
        scheduler._telemetry,
        "telemetry_hash",
    ),
    (
        "adapter handoff",
        _load_lifecycle_remediation_scheduler_adapter_handoff_payload,
        scheduler._adapter_handoff,
        "adapter_handoff_hash",
    ),
    (
        "acknowledgement",
        _load_lifecycle_remediation_scheduler_acknowledgement_payload,
        scheduler._acknowledgement,
        "acknowledgement_hash",
    ),
]


@pytest.mark.parametrize(
    ("loader", "build", "field"),
    [
        pytest.param(loader, build, field, id=name)
        for name, loader, build, field in _LOADERS
    ],
)
def test_loader_accepts_sealed_and_refuses_an_edit(
    loader: Callable[..., Any], build: Callable[[], dict[str, Any]], field: str
) -> None:
    sealed = build()
    assert loader(copy.deepcopy(sealed)) is not None
    edited = copy.deepcopy(sealed)
    edited["version"] = "9.9.9"  # any content change under the stale seal
    with pytest.raises(
        click.ClickException, match=f"{field} does not match its content"
    ):
        loader(edited)


def test_control_plan_refuses_a_dashboard_row_edited_under_its_seal(
    tmp_path: Path,
) -> None:
    dashboard = _dashboard_payload()
    rows = dashboard["rows"]
    blocked = next(row for row in rows if row["effective_state"] == "blocked")
    blocked["effective_state"] = "completed"  # would plan no_op instead of escalate
    _assert_fails(
        [
            "lifecycle-remediation-scheduler-control-plan",
            str(_write_json(tmp_path, "dashboard.json", dashboard)),
            "--created-by",
            "operator_console",
        ],
        "dashboard_hash does not match its content",
    )


def test_control_plan_refuses_a_text_overdue_flag(tmp_path: Path) -> None:
    dashboard = _dashboard_payload()
    dashboard["rows"][0]["overdue"] = "false"  # bool("false") is True
    _assert_fails(
        [
            "lifecycle-remediation-scheduler-control-plan",
            str(
                _write_json(
                    tmp_path, "dashboard.json", seal(dashboard, "dashboard_hash")
                )
            ),
            "--created-by",
            "operator_console",
        ],
        "row overdue must be a boolean",
    )


def test_sealed_dashboard_still_plans(tmp_path: Path) -> None:
    payload = _invoke_json(
        [
            "lifecycle-remediation-scheduler-control-plan",
            str(_write_json(tmp_path, "dashboard.json", _dashboard_payload())),
            "--created-by",
            "operator_console",
        ]
    )
    assert payload["control_counts"]["escalate"] == 1


def test_execution_dashboard_refuses_an_edited_replay(tmp_path: Path) -> None:
    telemetry = contracts._scheduler_telemetry_payload()
    replay = contracts._replay_payload(rows=[contracts._replay_row()])
    replay["rows"][0]["state"] = "blocked"  # edited after sealing
    runner = contracts.CliRunner()
    result = contracts._invoke(
        runner,
        [
            "lifecycle-remediation-scheduler-execution-dashboard",
            str(contracts._write_payload(tmp_path / "telemetry.json", telemetry)),
            str(contracts._write_payload(tmp_path / "replay.json", replay)),
            "--created-by",
            "deployment_scheduler",
        ],
    )
    contracts._assert_fails(result, "replay_hash does not match its content")
