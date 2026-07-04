# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — power-grid CLI bundle tests

"""CLI tests for the review-only power-grid PRC bundle command."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash
from scpn_phase_orchestrator.assurance.power_grid_prc_bundle import (
    DVOC_DAMPING_ROLE,
    IBR_RIDE_THROUGH_ROLE,
    PMU_RINGDOWN_ROLE,
    POWER_GRID_PRC_AUDIT_BUNDLE_SCHEMA,
    POWER_GRID_PRC_CLAIM_BOUNDARY,
)

_CREATED_AT = "2026-07-04T15:10:00Z"
DVOC_OSCILLATION_AUDIT_SCHEMA = "scpn_dvoc_oscillation_damping_audit_v1"
PMU_RINGDOWN_AUDIT_SCHEMA = "scpn_pmu_ringdown_prc_audit_v1"
IBR_RIDE_THROUGH_AUDIT_SCHEMA = "scpn_ibr_ride_through_prc029_audit_v1"
_MODULE_NAME = "scpn_phase_orchestrator.runtime.cli.power_grid"
_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "scpn_phase_orchestrator"
    / "runtime"
    / "cli"
    / "power_grid.py"
)


def _evidence_record(schema: str, event_id: str) -> dict[str, object]:
    """Return a deterministic JSON evidence record for CLI fixtures."""
    payload: dict[str, object] = {
        "schema": schema,
        "event_id": event_id,
        "flagged_count": 1,
        "claim_boundary": POWER_GRID_PRC_CLAIM_BOUNDARY,
        "review_only": True,
    }
    payload["content_hash"] = canonical_record_hash(payload)
    return payload


def _write_record(path: Path, record: dict[str, object]) -> Path:
    """Write a strict JSON evidence fixture."""
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def _load_power_grid_cli(monkeypatch: pytest.MonkeyPatch) -> click.Group:
    """Load the power-grid CLI command without importing the full CLI package."""
    main = click.Group()
    cli_package = types.ModuleType("scpn_phase_orchestrator.runtime.cli")
    cli_package.__path__ = [str(_MODULE_PATH.parent)]  # type: ignore[attr-defined]
    app_module = types.ModuleType("scpn_phase_orchestrator.runtime.cli._app")
    app_module.__dict__["main"] = main

    monkeypatch.setitem(sys.modules, "scpn_phase_orchestrator.runtime.cli", cli_package)
    monkeypatch.setitem(
        sys.modules,
        "scpn_phase_orchestrator.runtime.cli._app",
        app_module,
    )

    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, _MODULE_NAME, module)
    spec.loader.exec_module(module)
    return main


def test_power_grid_prc_bundle_cli_writes_assessor_handoff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The CLI binds three existing evidence files into one bundle JSON."""
    main = _load_power_grid_cli(monkeypatch)
    dvoc_path = _write_record(
        tmp_path / "dvoc.json",
        _evidence_record(DVOC_OSCILLATION_AUDIT_SCHEMA, "dvoc-event"),
    )
    pmu_path = _write_record(
        tmp_path / "pmu.json",
        _evidence_record(PMU_RINGDOWN_AUDIT_SCHEMA, "pmu-event"),
    )
    ibr_path = _write_record(
        tmp_path / "ibr.json",
        _evidence_record(IBR_RIDE_THROUGH_AUDIT_SCHEMA, "ibr-event"),
    )
    output_path = tmp_path / "power-grid-prc-bundle.json"

    result = CliRunner().invoke(
        main,
        [
            "power-grid-prc-bundle",
            "--bundle-id",
            "PG-REVIEW-001",
            "--created-at",
            _CREATED_AT,
            "--operator-context",
            "western interconnection post-event review",
            "--dvoc-evidence",
            str(dvoc_path),
            "--pmu-ringdown",
            str(pmu_path),
            "--ibr-ride-through",
            str(ibr_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "=== Power-grid PRC assessor bundle ===" in result.output
    assert "artifacts=3" in result.output
    assert f"bundle written to {output_path}" in result.output

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema"] == POWER_GRID_PRC_AUDIT_BUNDLE_SCHEMA
    assert payload["claim_boundary"] == POWER_GRID_PRC_CLAIM_BOUNDARY
    assert payload["review_only"] is True
    assert payload["evidence_hashes"] == {
        DVOC_DAMPING_ROLE: json.loads(dvoc_path.read_text())["content_hash"],
        PMU_RINGDOWN_ROLE: json.loads(pmu_path.read_text())["content_hash"],
        IBR_RIDE_THROUGH_ROLE: json.loads(ibr_path.read_text())["content_hash"],
    }
    artifacts = payload["artifacts"]
    assert [artifact["role"] for artifact in artifacts] == [
        DVOC_DAMPING_ROLE,
        PMU_RINGDOWN_ROLE,
        IBR_RIDE_THROUGH_ROLE,
    ]
    assert (
        artifacts[0]["source_sha256"]
        == hashlib.sha256(dvoc_path.read_bytes()).hexdigest()
    )
    assert len(str(payload["content_hash"])) == 64


def test_power_grid_prc_bundle_cli_reports_invalid_evidence_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Invalid evidence JSON is surfaced as a CLI error."""
    main = _load_power_grid_cli(monkeypatch)
    dvoc_path = tmp_path / "dvoc.json"
    dvoc_path.write_text('{"schema": NaN}', encoding="utf-8")
    pmu_path = _write_record(
        tmp_path / "pmu.json",
        _evidence_record(PMU_RINGDOWN_AUDIT_SCHEMA, "pmu-event"),
    )
    ibr_path = _write_record(
        tmp_path / "ibr.json",
        _evidence_record(IBR_RIDE_THROUGH_AUDIT_SCHEMA, "ibr-event"),
    )

    result = CliRunner().invoke(
        main,
        [
            "power-grid-prc-bundle",
            "--bundle-id",
            "PG-REVIEW-001",
            "--created-at",
            _CREATED_AT,
            "--operator-context",
            "review",
            "--dvoc-evidence",
            str(dvoc_path),
            "--pmu-ringdown",
            str(pmu_path),
            "--ibr-ride-through",
            str(ibr_path),
        ],
    )

    assert result.exit_code != 0
    assert "dvoc.json must be strict JSON" in result.output


@pytest.mark.parametrize(
    ("writer", "match"),
    [
        (lambda path: path.write_bytes(b"\xff"), "dvoc.json must be UTF-8 JSON"),
        (
            lambda path: path.write_text("{", encoding="utf-8"),
            "dvoc.json must be valid JSON",
        ),
        (
            lambda path: path.write_text("[]", encoding="utf-8"),
            "dvoc.json must contain a JSON object",
        ),
    ],
)
def test_power_grid_prc_bundle_cli_reports_malformed_json_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    writer: object,
    match: str,
) -> None:
    """Malformed source JSON files fail before bundle validation."""
    main = _load_power_grid_cli(monkeypatch)
    dvoc_path = tmp_path / "dvoc.json"
    if not callable(writer):
        raise AssertionError("writer must be callable")
    writer(dvoc_path)
    pmu_path = _write_record(
        tmp_path / "pmu.json",
        _evidence_record(PMU_RINGDOWN_AUDIT_SCHEMA, "pmu-event"),
    )
    ibr_path = _write_record(
        tmp_path / "ibr.json",
        _evidence_record(IBR_RIDE_THROUGH_AUDIT_SCHEMA, "ibr-event"),
    )

    result = CliRunner().invoke(
        main,
        [
            "power-grid-prc-bundle",
            "--bundle-id",
            "PG-REVIEW-001",
            "--created-at",
            _CREATED_AT,
            "--operator-context",
            "review",
            "--dvoc-evidence",
            str(dvoc_path),
            "--pmu-ringdown",
            str(pmu_path),
            "--ibr-ride-through",
            str(ibr_path),
        ],
    )

    assert result.exit_code != 0
    assert match in result.output
