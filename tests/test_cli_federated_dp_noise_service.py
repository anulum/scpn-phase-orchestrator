# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Federated DP noise-service CLI contract tests

"""CLI contract tests for federated DP noise-service preflight evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest
from click.testing import CliRunner

from scpn_phase_orchestrator.runtime.cli import (
    federated_dp_noise_service as _dp_noise_cli,
)
from scpn_phase_orchestrator.runtime.cli._app import main

assert _dp_noise_cli is not None


def _request(**overrides: object) -> dict[str, object]:
    """Return a valid DP-noise request declaration fixture."""
    payload: dict[str, object] = {
        "epsilon": 2.5,
        "delta": 1e-6,
        "sensitivity": 1.75,
        "noise_multiplier": 0.9,
        "node_count": 2,
        "seed_hash": "a" * 64,
        "policy_keys": ["alpha", "beta", "gamma"],
        "node_budgets": [
            {"node_id": "node-a", "epsilon_spent": 0.9},
            {"node_id": "node-b", "epsilon_spent": 1.2},
        ],
    }
    payload.update(overrides)
    return payload


def _deployment(**overrides: object) -> dict[str, object]:
    """Return a valid DP-noise deployment declaration fixture."""
    payload: dict[str, object] = {
        "mechanism_label": "mechanism-v1",
        "privacy_accountant_owner": "accountant-a",
        "seed_custody_label": "seed-custody-a",
        "budget_issuer_label": "budget-issuer-a",
        "service_endpoint_label": "https://dp-noise.internal",
        "operator_approved": True,
    }
    payload.update(overrides)
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    """Write one JSON object."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _invoke(request_path: Path, deployment_path: Path, *extra: str) -> object:
    """Invoke the DP noise-service preflight command."""
    return CliRunner().invoke(
        main,
        [
            "federated-dp-noise-service-preflight",
            str(request_path),
            str(deployment_path),
            *extra,
        ],
    )


def test_dp_noise_preflight_emits_ready_review_bundle(tmp_path: Path) -> None:
    request_path = _write_json(tmp_path / "request.json", _request())
    deployment_path = _write_json(tmp_path / "deployment.json", _deployment())
    output_path = tmp_path / "bundle.json"

    result = _invoke(request_path, deployment_path, "--output", str(output_path))

    assert result.exit_code == 0, result.output
    stdout_payload = json.loads(result.output)
    written_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert stdout_payload == written_payload
    assert (
        stdout_payload["schema"]
        == "scpn_federated_dp_noise_service_preflight_bundle_v1"
    )
    assert stdout_payload["version"] == "1.0.0"
    assert stdout_payload["deployment_ready"] is True
    assert stdout_payload["deployment_reason"] == "offline_deployment_preflight_ready"
    assert stdout_payload["service_execution_permitted"] is False
    assert stdout_payload["raw_data_export_permitted"] is False
    assert stdout_payload["operator_review_required"] is True
    assert stdout_payload["non_actuating"] is True
    assert len(stdout_payload["request_hash"]) == 64
    assert len(stdout_payload["response_hash"]) == 64
    assert len(stdout_payload["bundle_hash"]) == 64
    assert stdout_payload["request_manifest"]["node_count"] == 2
    assert stdout_payload["response_manifest"]["privacy_budget_spent"] == pytest.approx(
        2.1
    )
    assert stdout_payload["preflight_manifest"]["mechanism_label"] == "mechanism-v1"


def test_dp_noise_preflight_is_deterministic(tmp_path: Path) -> None:
    request_path = _write_json(tmp_path / "request.json", _request())
    deployment_path = _write_json(tmp_path / "deployment.json", _deployment())

    first = _invoke(request_path, deployment_path)
    second = _invoke(request_path, deployment_path)

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert json.loads(first.output) == json.loads(second.output)


def test_dp_noise_preflight_reports_not_ready_without_operator_approval(
    tmp_path: Path,
) -> None:
    request_path = _write_json(tmp_path / "request.json", _request())
    deployment_path = _write_json(
        tmp_path / "deployment.json", _deployment(operator_approved=False)
    )

    result = _invoke(request_path, deployment_path)

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["deployment_ready"] is False
    assert "operator approval required" in payload["deployment_reason"]


def test_dp_noise_preflight_reports_not_ready_with_empty_custody_label(
    tmp_path: Path,
) -> None:
    request_path = _write_json(tmp_path / "request.json", _request())
    deployment_path = _write_json(
        tmp_path / "deployment.json", _deployment(seed_custody_label="")
    )

    result = _invoke(request_path, deployment_path)

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["deployment_ready"] is False
    assert "seed_custody_label is required" in payload["deployment_reason"]


def test_dp_noise_preflight_rejects_invalid_request_scalar(tmp_path: Path) -> None:
    request_path = _write_json(tmp_path / "request.json", _request(epsilon=-1.0))
    deployment_path = _write_json(tmp_path / "deployment.json", _deployment())

    result = _invoke(request_path, deployment_path)

    assert result.exit_code == 1
    assert "epsilon must be greater than 0" in result.output


def test_dp_noise_preflight_rejects_non_bool_operator_flag(tmp_path: Path) -> None:
    request_path = _write_json(tmp_path / "request.json", _request())
    deployment_path = _write_json(
        tmp_path / "deployment.json", _deployment(operator_approved="yes")
    )

    result = _invoke(request_path, deployment_path)

    assert result.exit_code == 1
    assert "operator_approved must be a boolean" in result.output


def test_dp_noise_preflight_rejects_missing_label(tmp_path: Path) -> None:
    request_path = _write_json(tmp_path / "request.json", _request())
    deployment = _deployment()
    del deployment["mechanism_label"]
    deployment_path = _write_json(tmp_path / "deployment.json", deployment)

    result = _invoke(request_path, deployment_path)

    assert result.exit_code == 1
    assert "mechanism_label is required" in result.output


def test_dp_noise_preflight_rejects_non_object_request(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text("[1, 2, 3]", encoding="utf-8")
    deployment_path = _write_json(tmp_path / "deployment.json", _deployment())

    result = _invoke(request_path, deployment_path)

    assert result.exit_code == 1
    assert "payload must be a JSON object" in result.output


def test_dp_noise_preflight_reports_output_write_failure(tmp_path: Path) -> None:
    request_path = _write_json(tmp_path / "request.json", _request())
    deployment_path = _write_json(tmp_path / "deployment.json", _deployment())
    output_path = tmp_path / "missing-parent" / "bundle.json"

    result = _invoke(request_path, deployment_path, "--output", str(output_path))

    assert result.exit_code == 1
    assert "cannot write federated DP noise-service preflight bundle" in result.output


def test_build_node_budgets_rejects_non_array() -> None:
    with pytest.raises(ValueError, match="node_budgets must be a JSON array"):
        _dp_noise_cli._build_node_budgets({"node_budgets": "no"})


def test_build_node_budgets_rejects_non_object_item() -> None:
    with pytest.raises(ValueError, match=r"node_budgets\[0\] must be a JSON object"):
        _dp_noise_cli._build_node_budgets({"node_budgets": [5]})


def test_string_tuple_rejects_non_array() -> None:
    with pytest.raises(ValueError, match="policy_keys must be a JSON array of strings"):
        _dp_noise_cli._string_tuple({"policy_keys": "alpha"}, "policy_keys")


def test_string_tuple_rejects_non_string_item() -> None:
    with pytest.raises(ValueError, match=r"policy_keys\[1\] must be a string"):
        _dp_noise_cli._string_tuple({"policy_keys": ["a", 2]}, "policy_keys")


def test_string_field_requires_presence() -> None:
    with pytest.raises(ValueError, match="mechanism_label is required"):
        _dp_noise_cli._string_field({}, "mechanism_label")


def test_string_field_rejects_non_string() -> None:
    with pytest.raises(ValueError, match="mechanism_label must be a string"):
        _dp_noise_cli._string_field({"mechanism_label": 5}, "mechanism_label")


def test_string_field_allows_empty_value() -> None:
    assert (
        _dp_noise_cli._string_field({"seed_custody_label": ""}, "seed_custody_label")
        == ""
    )


def test_text_field_rejects_blank() -> None:
    with pytest.raises(ValueError, match="seed_hash must be a non-empty string"):
        _dp_noise_cli._text_field({"seed_hash": "  "}, "seed_hash")


def test_bool_field_rejects_non_bool() -> None:
    with pytest.raises(ValueError, match="operator_approved must be a boolean"):
        _dp_noise_cli._bool_field({"operator_approved": 1}, "operator_approved")


def test_int_field_requires_presence() -> None:
    with pytest.raises(ValueError, match="node_count is required"):
        _dp_noise_cli._int_field({}, "node_count")


def test_int_field_rejects_bool() -> None:
    with pytest.raises(ValueError, match="node_count must be an integer"):
        _dp_noise_cli._int_field({"node_count": True}, "node_count")


def test_int_field_accepts_integer() -> None:
    assert _dp_noise_cli._int_field({"node_count": 3}, "node_count") == 3


def test_float_field_requires_presence() -> None:
    with pytest.raises(ValueError, match="epsilon is required"):
        _dp_noise_cli._float_field({}, "epsilon")


def test_float_field_rejects_non_number() -> None:
    with pytest.raises(ValueError, match="epsilon must be a number"):
        _dp_noise_cli._float_field({"epsilon": "x"}, "epsilon")


def test_float_field_accepts_integer() -> None:
    assert _dp_noise_cli._float_field({"node_count": 2}, "node_count") == 2.0
