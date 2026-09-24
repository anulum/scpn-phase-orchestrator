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

import copy
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
    # deep copy: tests edit rows in place, and _ROWS is shared module state
    return _seal(
        {"schema": _DASHBOARD_SCHEMA, "version": "1.0.0", "rows": copy.deepcopy(rows)},
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


def _real_bundle(tmp_path: Path) -> dict[str, Any]:
    result = _bundle(tmp_path, dashboard=_dashboard(_ROWS))
    assert result.exit_code == 0, result.output
    return dict(json.loads(result.output))


def _write(tmp_path: Path, name: str, record: dict[str, Any]) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def _pack(tmp_path: Path, bundle: dict[str, Any]) -> Any:
    return CliRunner().invoke(
        main,
        [
            "digital-twin-grafana-dashboard-pack",
            str(_write(tmp_path, "bundle.json", bundle)),
            "--adapter-family",
            "kafka",
            "--created-by",
            "operator_console",
        ],
    )


def _playbook(tmp_path: Path, bundle: dict[str, Any], pack: dict[str, Any]) -> Any:
    return CliRunner().invoke(
        main,
        [
            "digital-twin-live-deployment-playbook",
            str(_write(tmp_path, "bundle.json", bundle)),
            str(_write(tmp_path, "pack.json", pack)),
            "--environment-name",
            "prod-eu-west",
            "--created-by",
            "operator_console",
        ],
    )


def test_real_chain_bundle_pack_playbook_passes(tmp_path: Path) -> None:
    bundle = _real_bundle(tmp_path)
    pack = _pack(tmp_path, bundle)
    assert pack.exit_code == 0, pack.output
    playbook = _playbook(tmp_path, bundle, json.loads(pack.output))
    assert playbook.exit_code == 0, playbook.output
    assert json.loads(playbook.output)["rollout_gate"] == "blocked"  # 1 blocked row


def test_pack_refuses_an_edited_bundle(tmp_path: Path) -> None:
    bundle = _real_bundle(tmp_path)
    bundle["status"] = "healthy" if bundle["status"] != "healthy" else "critical"
    result = _pack(tmp_path, bundle)
    assert result.exit_code != 0
    assert "bundle_hash does not match its content" in result.output


def test_pack_refuses_a_promql_breaking_prefix_even_when_resealed(
    tmp_path: Path,
) -> None:
    bundle = _real_bundle(tmp_path)
    bundle.pop("bundle_hash")
    bundle["prometheus_metric_prefix"] = 'spo{x="1"} or vector(1) #'
    result = _pack(tmp_path, _seal(bundle, "bundle_hash"))
    assert result.exit_code != 0
    assert "prometheus_metric_prefix must match" in result.output


def test_playbook_refuses_a_bundle_edited_to_clear_the_gate(tmp_path: Path) -> None:
    bundle = _real_bundle(tmp_path)
    pack = json.loads(_pack(tmp_path, bundle).output)
    bundle["replay_linkage"]["scheduler_overdue_count"] = 0
    bundle["replay_linkage"]["scheduler_blocked_count"] = (
        0  # the gate would read "ready"
    )
    result = _playbook(tmp_path, bundle, pack)
    assert result.exit_code != 0
    assert "bundle_hash does not match its content" in result.output


def test_playbook_refuses_an_edited_pack(tmp_path: Path) -> None:
    bundle = _real_bundle(tmp_path)
    pack = json.loads(_pack(tmp_path, bundle).output)
    pack["created_by"] = "someone_else"
    result = _playbook(tmp_path, bundle, pack)
    assert result.exit_code != 0
    assert "dashboard_pack_hash does not match its content" in result.output


def test_playbook_refuses_boolean_counts(tmp_path: Path) -> None:
    bundle = _real_bundle(tmp_path)
    bundle.pop("bundle_hash")
    bundle["replay_linkage"]["scheduler_blocked_count"] = True
    resealed = _seal(bundle, "bundle_hash")
    pack = _pack(tmp_path, resealed)
    assert pack.exit_code == 0, pack.output
    result = _playbook(tmp_path, resealed, json.loads(pack.output))
    assert result.exit_code != 0
    assert "scheduler_blocked_count must be non-negative integer" in result.output
