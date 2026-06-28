# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Federated secure aggregation CLI contract tests

"""CLI contract tests for federated secure-aggregation preflight evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from scpn_phase_orchestrator.runtime.cli import (
    federated_secure_aggregation as _secure_aggregation_cli,
)
from scpn_phase_orchestrator.runtime.cli._app import main

assert _secure_aggregation_cli is not None


def _stable_hash(value: object) -> str:
    """Return the canonical SHA-256 hash used by the secure-aggregation module."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _node_commitment(
    node_id: str,
    masked_policy_delta: dict[str, float],
    sample_count: int,
) -> dict[str, object]:
    """Return one valid secure-aggregation node commitment record."""
    delta_items = [
        [key, float(value)] for key, value in sorted(masked_policy_delta.items())
    ]
    return {
        "node_id": node_id,
        "masked_policy_delta": dict(sorted(masked_policy_delta.items())),
        "sample_count": sample_count,
        "share_commitment": f"commit-{node_id}",
        "share_commitment_hash": _stable_hash(
            {"node_id": node_id, "share_commitment": f"commit-{node_id}"}
        ),
        "share_hash": _stable_hash(
            {"node_id": node_id, "masked_policy_delta": delta_items}
        ),
    }


def _label(node_id: str, kind: str, tag: str) -> str:
    """Return a deterministic custody label."""
    return _stable_hash({"node_id": node_id, "kind": kind, "tag": tag})


def _custody_record(node_id: str, rotation_policy: str) -> dict[str, str]:
    """Return one valid node custody record for ``rotation_policy``."""
    key_previous = _label(node_id, "key", "previous-current")
    share_previous = _label(node_id, "share", "previous-current")
    key_label = _label(node_id, "key", "current")
    share_label = _label(node_id, "share", "current")
    return {
        "node_id": node_id,
        "key_custody_label": key_label,
        "share_custody_label": share_label,
        "previous_key_custody_label": key_previous,
        "previous_share_custody_label": share_previous,
        "key_custody_continuity_hash": _stable_hash(
            {
                "node_id": node_id,
                "rotation_policy": rotation_policy,
                "previous_key_custody_label": key_previous,
                "key_custody_label": key_label,
            }
        ),
        "share_custody_continuity_hash": _stable_hash(
            {
                "node_id": node_id,
                "rotation_policy": rotation_policy,
                "previous_share_custody_label": share_previous,
                "share_custody_label": share_label,
            }
        ),
    }


def _quorum_evidence(node_id: str) -> dict[str, str]:
    """Return one valid per-node quorum evidence record."""
    return {
        "node_id": node_id,
        "evidence_hash": _stable_hash({"node_id": node_id, "kind": "quorum"}),
    }


_NODE_IDS = ("node-a", "node-b", "node-c")


def _commitments() -> tuple[dict[str, object], ...]:
    """Return a valid three-node commitment batch."""
    return (
        _node_commitment("node-a", {"alpha": 0.2, "theta": 1.0}, 100),
        _node_commitment("node-b", {"theta": 0.4, "alpha": 0.1}, 40),
        _node_commitment("node-c", {"alpha": 0.0, "theta": -0.2}, 60),
    )


def _deployment(**overrides: object) -> dict[str, object]:
    """Return a valid secure-aggregation deployment declaration fixture."""
    payload: dict[str, object] = {
        "aggregation": {
            "required_policy_keys": ["theta", "alpha"],
            "clipping_norm": 2.0,
            "min_node_count": 3,
        },
        "quorum_evidence": [_quorum_evidence(node) for node in _NODE_IDS],
        "custody_rotation_policy": "continuous",
        "custody_records": [_custody_record(node, "continuous") for node in _NODE_IDS],
        "accepted_node_threshold": 3,
        "operator_approved": True,
        "operator_id": "ops-1",
        "service_owner": "svc-phase-orchestrator",
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


def _invoke(commitments_path: Path, deployment_path: Path, *extra: str) -> object:
    """Invoke the secure-aggregation preflight command."""
    return CliRunner().invoke(
        main,
        [
            "federated-secure-aggregation-preflight",
            str(commitments_path),
            str(deployment_path),
            *extra,
        ],
    )


def test_secure_aggregation_preflight_emits_review_bundle(tmp_path: Path) -> None:
    commitments_path = tmp_path / "commitments.jsonl"
    commitments_path.write_text(
        "\n".join(
            ["", *(json.dumps(row, sort_keys=True) for row in _commitments()), ""]
        ),
        encoding="utf-8",
    )
    deployment_path = _write_json(tmp_path / "deployment.json", _deployment())
    output_path = tmp_path / "bundle.json"

    result = _invoke(commitments_path, deployment_path, "--output", str(output_path))

    assert result.exit_code == 0, result.output
    stdout_payload = json.loads(result.output)
    written_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert stdout_payload == written_payload
    assert (
        stdout_payload["schema"]
        == "scpn_federated_secure_aggregation_preflight_bundle_v1"
    )
    assert stdout_payload["version"] == "1.0.0"
    assert stdout_payload["accepted_node_count"] == 3
    assert stdout_payload["accepted_node_threshold"] == 3
    assert stdout_payload["custody_rotation_policy"] == "continuous"
    assert stdout_payload["operator_id"] == "ops-1"
    assert stdout_payload["service_owner"] == "svc-phase-orchestrator"
    assert stdout_payload["secure_aggregation_execution_permitted"] is False
    assert stdout_payload["raw_data_export_permitted"] is False
    assert stdout_payload["operator_review_required"] is True
    assert stdout_payload["non_actuating"] is True
    assert len(stdout_payload["bundle_hash"]) == 64
    assert stdout_payload["preflight_manifest"]["accepted_node_count"] == 3
    assert stdout_payload["secure_aggregation_manifest"]["quorum_met"] is True


def test_secure_aggregation_preflight_is_deterministic(tmp_path: Path) -> None:
    commitments_path = _write_jsonl(tmp_path / "commitments.jsonl", _commitments())
    deployment_path = _write_json(tmp_path / "deployment.json", _deployment())

    first = _invoke(commitments_path, deployment_path)
    second = _invoke(commitments_path, deployment_path)

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert json.loads(first.output) == json.loads(second.output)


def test_secure_aggregation_preflight_infers_policy_keys_without_aggregation(
    tmp_path: Path,
) -> None:
    commitments_path = _write_jsonl(tmp_path / "commitments.jsonl", _commitments())
    deployment = _deployment(aggregation={"clipping_norm": 2.0, "min_node_count": 3})
    deployment_path = _write_json(tmp_path / "deployment.json", deployment)

    result = _invoke(commitments_path, deployment_path)

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["accepted_node_count"] == 3


def test_secure_aggregation_preflight_rejects_missing_operator_approval(
    tmp_path: Path,
) -> None:
    commitments_path = _write_jsonl(tmp_path / "commitments.jsonl", _commitments())
    deployment_path = _write_json(
        tmp_path / "deployment.json", _deployment(operator_approved=False)
    )

    result = _invoke(commitments_path, deployment_path)

    assert result.exit_code == 1
    assert "operator approval is required" in result.output


def test_secure_aggregation_preflight_rejects_unsupported_rotation_policy(
    tmp_path: Path,
) -> None:
    commitments_path = _write_jsonl(tmp_path / "commitments.jsonl", _commitments())
    deployment_path = _write_json(
        tmp_path / "deployment.json",
        _deployment(custody_rotation_policy="ad-hoc"),
    )

    result = _invoke(commitments_path, deployment_path)

    assert result.exit_code == 1
    assert "federated secure aggregation preflight failed" in result.output


def test_secure_aggregation_preflight_rejects_malformed_jsonl(tmp_path: Path) -> None:
    commitments_path = tmp_path / "commitments.jsonl"
    commitments_path.write_text("{not valid json\n", encoding="utf-8")
    deployment_path = _write_json(tmp_path / "deployment.json", _deployment())

    result = _invoke(commitments_path, deployment_path)

    assert result.exit_code == 1
    assert "malformed node-commitment JSONL" in result.output


def test_secure_aggregation_preflight_rejects_empty_jsonl(tmp_path: Path) -> None:
    commitments_path = tmp_path / "commitments.jsonl"
    commitments_path.write_text("\n\n", encoding="utf-8")
    deployment_path = _write_json(tmp_path / "deployment.json", _deployment())

    result = _invoke(commitments_path, deployment_path)

    assert result.exit_code == 1
    assert "at least one record" in result.output


def test_secure_aggregation_preflight_rejects_non_object_jsonl(tmp_path: Path) -> None:
    commitments_path = tmp_path / "commitments.jsonl"
    commitments_path.write_text("[1, 2, 3]\n", encoding="utf-8")
    deployment_path = _write_json(tmp_path / "deployment.json", _deployment())

    result = _invoke(commitments_path, deployment_path)

    assert result.exit_code == 1
    assert "must be a JSON object" in result.output


def test_secure_aggregation_preflight_reports_missing_commitment_file() -> None:
    missing_path = Path("/definitely/missing/secure-aggregation-commitments.jsonl")

    with pytest.raises(click.ClickException, match="cannot read node-commitment JSONL"):
        _secure_aggregation_cli._load_node_commitment_jsonl(missing_path)


def test_secure_aggregation_preflight_reports_output_write_failure(
    tmp_path: Path,
) -> None:
    commitments_path = _write_jsonl(tmp_path / "commitments.jsonl", _commitments())
    deployment_path = _write_json(tmp_path / "deployment.json", _deployment())
    output_path = tmp_path / "missing-parent" / "bundle.json"

    result = _invoke(commitments_path, deployment_path, "--output", str(output_path))

    assert result.exit_code == 1
    assert "cannot write federated secure aggregation preflight bundle" in result.output


def test_aggregation_config_rejects_non_object() -> None:
    with pytest.raises(ValueError, match="aggregation must be a JSON object"):
        _secure_aggregation_cli._aggregation_config({"aggregation": [1, 2]})


def test_mapping_sequence_rejects_non_array() -> None:
    with pytest.raises(ValueError, match="quorum_evidence must be a JSON array"):
        _secure_aggregation_cli._mapping_sequence(
            {"quorum_evidence": "no"}, "quorum_evidence"
        )


def test_mapping_sequence_rejects_non_object_item() -> None:
    with pytest.raises(ValueError, match=r"custody_records\[1\] must be a JSON object"):
        _secure_aggregation_cli._mapping_sequence(
            {"custody_records": [{}, 5]}, "custody_records"
        )


def test_string_sequence_rejects_non_array() -> None:
    with pytest.raises(ValueError, match="keys must be a JSON array of strings"):
        _secure_aggregation_cli._string_sequence("theta", "keys")


def test_string_sequence_rejects_non_string_item() -> None:
    with pytest.raises(ValueError, match=r"keys\[0\] must be a string"):
        _secure_aggregation_cli._string_sequence([1], "keys")


def test_string_sequence_accepts_valid() -> None:
    assert _secure_aggregation_cli._string_sequence(["a", "b"], "keys") == ("a", "b")


def test_text_field_rejects_blank() -> None:
    with pytest.raises(ValueError, match="operator_id must be a non-empty string"):
        _secure_aggregation_cli._text_field({"operator_id": "  "}, "operator_id")


def test_bool_field_rejects_non_bool() -> None:
    with pytest.raises(ValueError, match="operator_approved must be a boolean"):
        _secure_aggregation_cli._bool_field(
            {"operator_approved": "yes"}, "operator_approved"
        )


def test_int_field_requires_value_without_default() -> None:
    with pytest.raises(ValueError, match="accepted_node_threshold is required"):
        _secure_aggregation_cli._int_field({}, "accepted_node_threshold", default=None)


def test_int_field_rejects_bool() -> None:
    with pytest.raises(ValueError, match="min_node_count must be an integer"):
        _secure_aggregation_cli._int_field(
            {"min_node_count": True}, "min_node_count", default=3
        )


def test_int_field_uses_default_when_absent() -> None:
    assert _secure_aggregation_cli._int_field({}, "min_node_count", default=3) == 3


def test_float_field_rejects_non_number() -> None:
    with pytest.raises(ValueError, match="clipping_norm must be a number"):
        _secure_aggregation_cli._float_field(
            {"clipping_norm": "x"}, "clipping_norm", default=1.0
        )


def test_float_field_uses_default_when_absent() -> None:
    assert _secure_aggregation_cli._float_field({}, "epsilon", default=3.0) == 3.0


def test_float_field_accepts_integer() -> None:
    assert (
        _secure_aggregation_cli._float_field({"delta": 0}, "delta", default=1e-6) == 0.0
    )
