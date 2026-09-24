# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — assurance evidence rows carry real typed fields

"""An evidence row with a null id must not enter the case as evidence "None".

``str(row["evidence_id"])`` turned ``null`` into ``"None"`` and a number into
its digits, and ``dict(row["record"])`` reshaped a list of pairs into a
record, so malformed evidence entered the assurance case under fabricated
identifiers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from scpn_phase_orchestrator.runtime.cli import main

_ROW = {
    "evidence_id": "replay-1",
    "category": "replay_determinism",
    "summary": "deterministic replay over run 1",
    "record": {"deterministic": True, "verified_transitions": 10},
}


def _case(tmp_path: Path, payload: Any) -> Any:
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8"
    )
    return CliRunner().invoke(
        main, ["assurance-case", "--system", "Sys", "--evidence-file", str(evidence)]
    )


def test_well_formed_row_builds_a_case(tmp_path: Path) -> None:
    result = _case(tmp_path, [_ROW])
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("evidence_id", None),
        ("evidence_id", 42),
        ("summary", None),
        ("summary", ""),
        ("category", None),
    ],
)
def test_non_string_field_is_refused(tmp_path: Path, field: str, value: object) -> None:
    result = _case(tmp_path, [{**_ROW, field: value}])
    assert result.exit_code != 0
    assert f"field {field!r} must be a non-empty string" in result.output
    assert '"None"' not in result.output


def test_record_given_as_pairs_is_refused(tmp_path: Path) -> None:
    result = _case(tmp_path, [{**_ROW, "record": [["deterministic", True]]}])
    assert result.exit_code != 0
    assert "field 'record' must be a JSON object" in result.output


def test_malformed_json_is_a_clean_error(tmp_path: Path) -> None:
    result = _case(tmp_path, "[{not json")
    assert result.exit_code != 0
    assert "is not valid JSON" in result.output
