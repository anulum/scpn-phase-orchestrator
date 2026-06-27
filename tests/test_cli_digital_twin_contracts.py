# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — digital-twin CLI contract tests

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path

import pytest
from click.testing import CliRunner

from scpn_phase_orchestrator.runtime.cli import main

_HASH_1 = "1" * 64
_HASH_2 = "2" * 64
_HASH_3 = "3" * 64


def _record_hash(record: Mapping[str, object]) -> str:
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _evidence_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "contract_hash": _HASH_1,
        "accepted_count": 1,
        "rejected_count": 0,
        "adapter_count": 1,
        "unhealthy_adapter_count": 0,
        "latest_sequence": 7,
        "max_abs_twin_residual": 0.01,
        "status": "healthy",
        "capability_counts": {"phase_observation": 1},
        "direction_counts": {"twin_to_spo": 1},
        "mismatch_reasons": [],
    }
    payload.update(overrides)
    return payload


def _dashboard_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_remediation_scheduler_"
            "execution_dashboard_v1"
        ),
        "rows": [
            {"effective_state": "completed", "overdue": False},
            {"effective_state": "blocked", "overdue": True},
        ],
        "dashboard_hash": _HASH_2,
    }
    payload.update(overrides)
    return payload


def _replay_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_remediation_scheduler_"
            "acknowledgement_replay_v1"
        ),
        "rows": [{"state": "completed"}, {"state": "blocked"}],
        "replay_hash": _HASH_3,
    }
    payload.update(overrides)
    return payload


def _bundle_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "scpn_digital_twin_observability_bundle_v1",
        "version": "1.0.0",
        "contract_hash": _HASH_1,
        "status": "healthy",
        "accepted_count": 1,
        "rejected_count": 0,
        "prometheus_metric_prefix": "spo",
        "prometheus_text": "spo_digital_twin_sync_accepted_total 1\n",
        "replay_linkage": {
            "scheduler_dashboard_present": True,
            "scheduler_replay_present": True,
            "scheduler_row_count": 2,
            "scheduler_overdue_count": 0,
            "scheduler_blocked_count": 0,
            "scheduler_completed_count": 2,
            "scheduler_replay_count": 2,
            "scheduler_replay_blocked_count": 0,
            "scheduler_replay_completed_count": 2,
            "scheduler_dashboard_hash": _HASH_2,
            "scheduler_replay_hash": _HASH_3,
        },
        "created_by": "operator_console",
    }
    payload.update(overrides)
    payload["bundle_hash"] = _record_hash(payload)
    return payload


def _dashboard_pack_payload(
    *,
    bundle_hash: str,
    **overrides: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "scpn_digital_twin_grafana_dashboard_pack_v1",
        "version": "1.0.0",
        "adapter_family": "rest",
        "contract_hash": _HASH_1,
        "observability_bundle_hash": bundle_hash,
        "panel_count": 0,
        "panels": [],
        "created_by": "operator_console",
        "dashboard_pack_hash": _HASH_2,
    }
    payload.update(overrides)
    return payload


def _invoke(
    tmp_path: Path,
    args: Sequence[str],
    payloads: Mapping[str, Mapping[str, object]],
) -> object:
    for name, payload in payloads.items():
        _write_json(tmp_path / name, payload)
    return CliRunner().invoke(main, list(args))


@pytest.mark.parametrize(
    ("evidence_overrides", "match"),
    [
        ({"contract_hash": "not-a-hash"}, "contract_hash"),
        ({"accepted_count": True}, "accepted_count must be a non-negative integer"),
        ({"rejected_count": False}, "rejected_count must be a non-negative integer"),
    ],
)
def test_digital_twin_observability_bundle_rejects_bad_evidence(
    tmp_path: Path,
    evidence_overrides: dict[str, object],
    match: str,
) -> None:
    result = _invoke(
        tmp_path,
        (
            "digital-twin-observability-bundle",
            str(tmp_path / "evidence.json"),
            "--created-by",
            "operator_console",
        ),
        {"evidence.json": _evidence_payload(**evidence_overrides)},
    )

    assert result.exit_code == 1
    assert match in result.output


def test_digital_twin_observability_bundle_requires_created_by(
    tmp_path: Path,
) -> None:
    result = _invoke(
        tmp_path,
        (
            "digital-twin-observability-bundle",
            str(tmp_path / "evidence.json"),
            "--created-by",
            "",
        ),
        {"evidence.json": _evidence_payload()},
    )

    assert result.exit_code == 1
    assert "created_by must be non-empty" in result.output


@pytest.mark.parametrize(
    ("dashboard_overrides", "match"),
    [
        ({"rows": "not-a-list"}, "scheduler dashboard rows must be list"),
        ({"rows": ["not-an-object"]}, "scheduler dashboard row must be object"),
    ],
)
def test_digital_twin_observability_bundle_rejects_bad_dashboard_rows(
    tmp_path: Path,
    dashboard_overrides: dict[str, object],
    match: str,
) -> None:
    result = _invoke(
        tmp_path,
        (
            "digital-twin-observability-bundle",
            str(tmp_path / "evidence.json"),
            "--scheduler-dashboard-json",
            str(tmp_path / "dashboard.json"),
            "--created-by",
            "operator_console",
        ),
        {
            "evidence.json": _evidence_payload(),
            "dashboard.json": _dashboard_payload(**dashboard_overrides),
        },
    )

    assert result.exit_code == 1
    assert match in result.output


@pytest.mark.parametrize(
    ("replay_overrides", "match"),
    [
        ({"schema": "wrong"}, "unexpected scheduler replay schema"),
        ({"rows": "not-a-list"}, "scheduler replay rows must be list"),
        ({"rows": ["not-an-object"]}, "scheduler replay row must be object"),
    ],
)
def test_digital_twin_observability_bundle_rejects_bad_replay_rows(
    tmp_path: Path,
    replay_overrides: dict[str, object],
    match: str,
) -> None:
    result = _invoke(
        tmp_path,
        (
            "digital-twin-observability-bundle",
            str(tmp_path / "evidence.json"),
            "--scheduler-replay-json",
            str(tmp_path / "replay.json"),
            "--created-by",
            "operator_console",
        ),
        {
            "evidence.json": _evidence_payload(),
            "replay.json": _replay_payload(**replay_overrides),
        },
    )

    assert result.exit_code == 1
    assert match in result.output


@pytest.mark.parametrize(
    ("args", "bundle_overrides", "match"),
    [
        (
            (
                "digital-twin-grafana-dashboard-pack",
                "{bundle}",
                "--adapter-family",
                "rest",
                "--created-by",
                "",
            ),
            {},
            "created_by must be non-empty",
        ),
        (
            (
                "digital-twin-grafana-dashboard-pack",
                "{bundle}",
                "--adapter-family",
                "",
                "--created-by",
                "operator_console",
            ),
            {},
            "adapter_family must be non-empty",
        ),
        (
            (
                "digital-twin-grafana-dashboard-pack",
                "{bundle}",
                "--adapter-family",
                "rest",
                "--created-by",
                "operator_console",
            ),
            {"schema": "wrong"},
            "unexpected observability bundle schema",
        ),
        (
            (
                "digital-twin-grafana-dashboard-pack",
                "{bundle}",
                "--adapter-family",
                "rest",
                "--created-by",
                "operator_console",
            ),
            {"prometheus_metric_prefix": ""},
            "prometheus_metric_prefix must be non-empty string",
        ),
    ],
)
def test_digital_twin_grafana_dashboard_pack_rejects_bad_bundle(
    tmp_path: Path,
    args: tuple[str, ...],
    bundle_overrides: dict[str, object],
    match: str,
) -> None:
    bundle_path = tmp_path / "bundle.json"
    formatted_args = tuple(
        str(bundle_path) if arg == "{bundle}" else arg for arg in args
    )

    result = _invoke(
        tmp_path,
        formatted_args,
        {"bundle.json": _bundle_payload(**bundle_overrides)},
    )

    assert result.exit_code == 1
    assert match in result.output


@pytest.mark.parametrize(
    ("args", "bundle_overrides", "pack_overrides", "match"),
    [
        (
            (
                "digital-twin-live-deployment-playbook",
                "{bundle}",
                "{pack}",
                "--environment-name",
                "prod",
                "--created-by",
                "",
            ),
            {},
            {},
            "created_by must be non-empty",
        ),
        (
            (
                "digital-twin-live-deployment-playbook",
                "{bundle}",
                "{pack}",
                "--environment-name",
                "",
                "--created-by",
                "operator_console",
            ),
            {},
            {},
            "environment_name must be non-empty",
        ),
        (
            (
                "digital-twin-live-deployment-playbook",
                "{bundle}",
                "{pack}",
                "--environment-name",
                "prod",
                "--created-by",
                "operator_console",
            ),
            {"schema": "wrong"},
            {},
            "unexpected observability bundle schema",
        ),
        (
            (
                "digital-twin-live-deployment-playbook",
                "{bundle}",
                "{pack}",
                "--environment-name",
                "prod",
                "--created-by",
                "operator_console",
            ),
            {},
            {"schema": "wrong"},
            "unexpected grafana dashboard pack schema",
        ),
        (
            (
                "digital-twin-live-deployment-playbook",
                "{bundle}",
                "{pack}",
                "--environment-name",
                "prod",
                "--created-by",
                "operator_console",
            ),
            {"replay_linkage": "not-an-object"},
            {},
            "replay_linkage must be object",
        ),
        (
            (
                "digital-twin-live-deployment-playbook",
                "{bundle}",
                "{pack}",
                "--environment-name",
                "prod",
                "--created-by",
                "operator_console",
            ),
            {
                "replay_linkage": {
                    "scheduler_overdue_count": -1,
                    "scheduler_blocked_count": 0,
                }
            },
            {},
            "scheduler_overdue_count must be non-negative integer",
        ),
        (
            (
                "digital-twin-live-deployment-playbook",
                "{bundle}",
                "{pack}",
                "--environment-name",
                "prod",
                "--created-by",
                "operator_console",
            ),
            {
                "replay_linkage": {
                    "scheduler_overdue_count": 0,
                    "scheduler_blocked_count": -1,
                }
            },
            {},
            "scheduler_blocked_count must be non-negative integer",
        ),
    ],
)
def test_digital_twin_live_deployment_playbook_rejects_bad_inputs(
    tmp_path: Path,
    args: tuple[str, ...],
    bundle_overrides: dict[str, object],
    pack_overrides: dict[str, object],
    match: str,
) -> None:
    bundle = _bundle_payload(**bundle_overrides)
    pack = _dashboard_pack_payload(
        bundle_hash=str(bundle["bundle_hash"]),
        **pack_overrides,
    )
    bundle_path = tmp_path / "bundle.json"
    pack_path = tmp_path / "pack.json"
    formatted_args = tuple(
        str(bundle_path)
        if arg == "{bundle}"
        else str(pack_path)
        if arg == "{pack}"
        else arg
        for arg in args
    )

    result = _invoke(
        tmp_path,
        formatted_args,
        {"bundle.json": bundle, "pack.json": pack},
    )

    assert result.exit_code == 1
    assert match in result.output
