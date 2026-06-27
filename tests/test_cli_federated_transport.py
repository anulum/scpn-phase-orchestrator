# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Federated transport CLI contract tests

"""CLI contract tests for federated transport preflight evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from scpn_phase_orchestrator.runtime.cli import (
    federated_transport as _federated_transport_cli,
)
from scpn_phase_orchestrator.runtime.cli._app import main

assert _federated_transport_cli is not None


def _record_hash(record: Mapping[str, object]) -> str:
    """Return a canonical SHA-256 hash for fixture records."""
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def _update(
    *,
    node_id: str,
    policy_delta: Mapping[str, float],
    sample_count: int,
    previous_audit_hash: str,
) -> dict[str, object]:
    """Return one valid node update audit record."""
    payload: dict[str, object] = {
        "node_id": node_id,
        "policy_delta": dict(policy_delta),
        "sample_count": sample_count,
        "local_loss": 0.21,
        "previous_audit_hash": previous_audit_hash,
        "privacy_epsilon_spent": 0.4,
        "clipped_l2_norm": 0.25,
        "clip_scale": 1.0,
        "accepted": True,
        "rejection_reasons": [],
    }
    hash_payload = dict(payload)
    hash_payload["policy_delta"] = [
        [key, value] for key, value in sorted(policy_delta.items())
    ]
    payload["update_hash"] = _record_hash(hash_payload)
    return payload


def _updates() -> tuple[dict[str, object], ...]:
    """Return a compact valid federated update batch."""
    return (
        _update(
            node_id="node-a",
            policy_delta={"K": 0.1, "alpha": -0.01},
            sample_count=100,
            previous_audit_hash="a" * 64,
        ),
        _update(
            node_id="node-b",
            policy_delta={"K": 0.2, "alpha": 0.01},
            sample_count=120,
            previous_audit_hash="b" * 64,
        ),
    )


def _declaration(**overrides: object) -> dict[str, object]:
    """Return a valid live-transport declaration fixture."""
    payload: dict[str, object] = {
        "transport": "rest",
        "endpoint": "https://transport.local/replay",
        "owner": "node-owner-a",
        "auth_policy": "mtls+token",
        "tls": True,
        "replay_supported": True,
        "operator_approved": True,
    }
    payload.update(overrides)
    return payload


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    """Write rows as newline-delimited JSON."""
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    """Write one JSON object."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def test_federated_transport_preflight_emits_review_bundle(tmp_path: Path) -> None:
    updates_path = tmp_path / "updates.jsonl"
    updates_path.write_text(
        "\n".join(["", *(json.dumps(row, sort_keys=True) for row in _updates()), ""]),
        encoding="utf-8",
    )
    declaration_path = _write_json(tmp_path / "transport.json", _declaration())
    output_path = tmp_path / "bundle.json"

    result = CliRunner().invoke(
        main,
        [
            "federated-transport-preflight",
            str(updates_path),
            str(declaration_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    stdout_payload = json.loads(result.output)
    written_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert stdout_payload == written_payload
    assert stdout_payload["schema"] == "scpn_federated_transport_preflight_bundle_v1"
    assert stdout_payload["version"] == "1.0.0"
    assert stdout_payload["envelope_count"] == 2
    assert stdout_payload["transport_execution_permitted"] is False
    assert stdout_payload["raw_data_export_permitted"] is False
    assert stdout_payload["operator_review_required"] is True
    assert stdout_payload["non_actuating"] is True
    assert len(stdout_payload["bundle_hash"]) == 64
    assert stdout_payload["preflight_manifest"]["transport"] == "rest"
    assert stdout_payload["replay_ledger"]["envelope_count"] == 2
    assert len(stdout_payload["envelopes"]) == 2


def test_federated_transport_preflight_is_deterministic(tmp_path: Path) -> None:
    updates_path = _write_jsonl(tmp_path / "updates.jsonl", _updates())
    declaration_path = _write_json(tmp_path / "transport.json", _declaration())
    runner = CliRunner()

    first = runner.invoke(
        main,
        ["federated-transport-preflight", str(updates_path), str(declaration_path)],
    )
    second = runner.invoke(
        main,
        ["federated-transport-preflight", str(updates_path), str(declaration_path)],
    )

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert json.loads(first.output) == json.loads(second.output)


def test_federated_transport_preflight_rejects_raw_data_export(
    tmp_path: Path,
) -> None:
    bad_update = dict(_updates()[0])
    bad_update["raw_time_series"] = [0.1, 0.2]
    updates_path = _write_jsonl(tmp_path / "updates.jsonl", (bad_update,))
    declaration_path = _write_json(tmp_path / "transport.json", _declaration())

    result = CliRunner().invoke(
        main,
        ["federated-transport-preflight", str(updates_path), str(declaration_path)],
    )

    assert result.exit_code == 1
    assert "raw time-series" in result.output


def test_federated_transport_preflight_rejects_live_without_tls(
    tmp_path: Path,
) -> None:
    updates_path = _write_jsonl(tmp_path / "updates.jsonl", _updates())
    declaration_path = _write_json(tmp_path / "transport.json", _declaration(tls=False))

    result = CliRunner().invoke(
        main,
        ["federated-transport-preflight", str(updates_path), str(declaration_path)],
    )

    assert result.exit_code == 1
    assert "TLS" in result.output


def test_federated_transport_preflight_rejects_malformed_jsonl(
    tmp_path: Path,
) -> None:
    updates_path = tmp_path / "updates.jsonl"
    updates_path.write_text("{not valid json\n", encoding="utf-8")
    declaration_path = _write_json(tmp_path / "transport.json", _declaration())

    result = CliRunner().invoke(
        main,
        ["federated-transport-preflight", str(updates_path), str(declaration_path)],
    )

    assert result.exit_code == 1
    assert "malformed node-update JSONL" in result.output


def test_federated_transport_preflight_rejects_empty_jsonl(tmp_path: Path) -> None:
    updates_path = tmp_path / "updates.jsonl"
    updates_path.write_text("\n\n", encoding="utf-8")
    declaration_path = _write_json(tmp_path / "transport.json", _declaration())

    result = CliRunner().invoke(
        main,
        ["federated-transport-preflight", str(updates_path), str(declaration_path)],
    )

    assert result.exit_code == 1
    assert "at least one record" in result.output


def test_federated_transport_preflight_rejects_non_object_jsonl(
    tmp_path: Path,
) -> None:
    updates_path = tmp_path / "updates.jsonl"
    updates_path.write_text("[1, 2, 3]\n", encoding="utf-8")
    declaration_path = _write_json(tmp_path / "transport.json", _declaration())

    result = CliRunner().invoke(
        main,
        ["federated-transport-preflight", str(updates_path), str(declaration_path)],
    )

    assert result.exit_code == 1
    assert "must be a JSON object" in result.output


def test_federated_transport_preflight_reports_missing_update_file() -> None:
    missing_path = Path("/definitely/missing/federated-updates.jsonl")

    with pytest.raises(click.ClickException, match="cannot read node-update JSONL"):
        _federated_transport_cli._load_node_update_jsonl(missing_path)


def test_federated_transport_preflight_reports_output_write_failure(
    tmp_path: Path,
) -> None:
    updates_path = _write_jsonl(tmp_path / "updates.jsonl", _updates())
    declaration_path = _write_json(tmp_path / "transport.json", _declaration())
    output_path = tmp_path / "missing-parent" / "bundle.json"

    result = CliRunner().invoke(
        main,
        [
            "federated-transport-preflight",
            str(updates_path),
            str(declaration_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 1
    assert "cannot write federated transport preflight bundle" in result.output
