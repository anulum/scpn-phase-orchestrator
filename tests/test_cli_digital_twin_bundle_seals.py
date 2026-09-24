# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — observability bundle links only sealed scheduler records

"""The bundle must count the rows its linkage hash actually covers.

The scheduler seals its dashboard and replay with ``_record_hash`` over the
record; the bundle copied those hashes as linkage and counted the rows
without checking them, so an edited row under a stale hash was bundled as
the sealed record. ``bool(row["overdue"])`` also counted ``"false"`` as
overdue.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from scpn_phase_orchestrator.runtime.cli import main

_DASHBOARD_SCHEMA = (
    "scpn_plugin_execution_request_lifecycle_remediation_"
    "scheduler_execution_dashboard_v1"
)
_REPLAY_SCHEMA = (
    "scpn_plugin_execution_request_lifecycle_remediation_"
    "scheduler_acknowledgement_replay_v1"
)


def _seal(record: dict[str, Any], field: str) -> dict[str, Any]:
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return {**record, field: hashlib.sha256(canonical.encode("utf-8")).hexdigest()}


def _evidence(tmp_path: Path) -> Path:
    path = tmp_path / "evidence.json"
    path.write_text(
        json.dumps(
            {
                "contract_hash": "1" * 64,
                "accepted_count": 1,
                "rejected_count": 0,
                "adapter_count": 1,
                "unhealthy_adapter_count": 0,
                "latest_sequence": 1,
                "max_abs_twin_residual": 0.01,
                "status": "healthy",
                "capability_counts": {"push": 1},
                "direction_counts": {"inbound": 1},
                "mismatch_reasons": [],
            }
        ),
        encoding="utf-8",
    )
    return path


def _dashboard(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return _seal(
        {"schema": _DASHBOARD_SCHEMA, "version": "1.0.0", "rows": rows},
        "dashboard_hash",
    )


def _replay(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return _seal(
        {"schema": _REPLAY_SCHEMA, "version": "1.0.0", "rows": rows}, "replay_hash"
    )


def _bundle(tmp_path: Path, *, dashboard: Any = None, replay: Any = None) -> Any:
    args = ["digital-twin-observability-bundle", str(_evidence(tmp_path))]
    for flag, record in (
        ("--scheduler-dashboard-json", dashboard),
        ("--scheduler-replay-json", replay),
    ):
        if record is not None:
            path = tmp_path / f"{flag.strip('-')}.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            args += [flag, str(path)]
    return CliRunner().invoke(main, [*args, "--created-by", "operator_console"])


_ROWS = [
    {"action_hash": "6" * 64, "effective_state": "completed", "overdue": False},
    {"action_hash": "7" * 64, "effective_state": "blocked", "overdue": True},
]


def test_sealed_dashboard_and_replay_are_bundled(tmp_path: Path) -> None:
    replay_rows = [{"action_hash": "6" * 64, "state": "completed"}]
    result = _bundle(tmp_path, dashboard=_dashboard(_ROWS), replay=_replay(replay_rows))
    assert result.exit_code == 0, result.output
    linkage = json.loads(result.output)["replay_linkage"]
    assert linkage["scheduler_row_count"] == 2
    assert linkage["scheduler_overdue_count"] == 1


def test_edited_dashboard_row_under_a_stale_hash_is_refused(tmp_path: Path) -> None:
    dashboard = _dashboard(_ROWS)
    dashboard["rows"][1]["effective_state"] = "completed"
    result = _bundle(tmp_path, dashboard=dashboard)
    assert result.exit_code != 0
    assert "dashboard_hash does not match its content" in result.output


def test_edited_replay_under_a_stale_hash_is_refused(tmp_path: Path) -> None:
    replay = _replay([{"action_hash": "6" * 64, "state": "completed"}])
    replay["rows"].append({"action_hash": "7" * 64, "state": "completed"})
    result = _bundle(tmp_path, replay=replay)
    assert result.exit_code != 0
    assert "replay_hash does not match its content" in result.output


@pytest.mark.parametrize("flag", ["false", "true", 0, 1, None])
def test_non_boolean_overdue_flag_is_refused(tmp_path: Path, flag: object) -> None:
    rows = [{"action_hash": "6" * 64, "effective_state": "pending", "overdue": flag}]
    result = _bundle(tmp_path, dashboard=_dashboard(rows))
    assert result.exit_code != 0
    assert "overdue must be a boolean" in result.output
