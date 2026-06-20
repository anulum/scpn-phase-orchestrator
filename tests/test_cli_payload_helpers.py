# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI plan-payload helper validation guards

from __future__ import annotations

from pathlib import Path

import click
import pytest

from scpn_phase_orchestrator.runtime.cli._payloads import (
    _build_plan_payload_for_hash,
    _load_json_file,
    _normalize_approved_target_hashes,
    _require_sha256,
)

_HEX = "a" * 64


class TestRequireSha256:
    def test_rejects_non_string(self) -> None:
        with pytest.raises(click.ClickException, match="64-character SHA-256 digest"):
            _require_sha256(12345, "plan_hash")

    def test_rejects_non_hex_string(self) -> None:
        with pytest.raises(click.ClickException, match="not a valid SHA-256 digest"):
            _require_sha256("z" * 64, "plan_hash")

    def test_lowercases_valid_digest(self) -> None:
        assert _require_sha256("A" * 64, "plan_hash") == "a" * 64


class TestNormalizeApprovedTargetHashes:
    def test_rejects_invalid_hash(self) -> None:
        with pytest.raises(click.ClickException, match="not a valid SHA-256 digest"):
            _normalize_approved_target_hashes(("not-a-digest",))

    def test_deduplicates_and_lowercases(self) -> None:
        result = _normalize_approved_target_hashes(("A" * 64, "a" * 64))
        assert result == ("a" * 64,)


class TestLoadJsonFile:
    def test_rejects_unreadable_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "does-not-exist.json"
        with pytest.raises(click.ClickException, match="cannot read plan file"):
            _load_json_file(missing)

    def test_rejects_malformed_json(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(click.ClickException, match="malformed plan JSON"):
            _load_json_file(path)

    def test_rejects_non_object_payload(self, tmp_path: Path) -> None:
        path = tmp_path / "list.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(click.ClickException, match="payload must be a JSON object"):
            _load_json_file(path)

    def test_uses_artifact_label_in_errors(self, tmp_path: Path) -> None:
        path = tmp_path / "list.json"
        path.write_text("[]", encoding="utf-8")
        with pytest.raises(click.ClickException, match="approval payload must be"):
            _load_json_file(path, artifact="approval")

    def test_returns_parsed_object(self, tmp_path: Path) -> None:
        path = tmp_path / "ok.json"
        path.write_text('{"a": 1}', encoding="utf-8")
        assert _load_json_file(path) == {"a": 1}


class TestBuildPlanPayloadForHash:
    def test_requires_plan_hash(self) -> None:
        with pytest.raises(
            click.ClickException, match="missing required field plan_hash"
        ):
            _build_plan_payload_for_hash({"target_hash": _HEX})

    def test_requires_target_hash(self) -> None:
        with pytest.raises(
            click.ClickException, match="missing required field target_hash"
        ):
            _build_plan_payload_for_hash({"plan_hash": _HEX})

    def test_strips_derived_fields_before_hashing(self) -> None:
        payload = {
            "plan_hash": _HEX,
            "target_hash": _HEX,
            "manifest": {"x": 1},
            "capability": {"y": 2},
            "compatible": True,
            "compatibility_reasons": [],
            "argument_count": 0,
        }
        stripped = _build_plan_payload_for_hash(payload)
        assert "plan_hash" not in stripped
        assert "manifest" not in stripped
        assert "capability" not in stripped
        assert "compatible" not in stripped
        assert "compatibility_reasons" not in stripped
        assert stripped["target_hash"] == _HEX
        assert stripped["argument_count"] == 0
