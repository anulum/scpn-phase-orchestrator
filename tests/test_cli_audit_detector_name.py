# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — audit-detector seals a real detector name

"""``detector_name: null`` must not be sealed as the detector "None"."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from scpn_phase_orchestrator.runtime.cli import main

_SCORES = {"event_scores": [0.9, 0.8, 0.95, 0.7], "null_scores": [0.1, 0.2, 0.15, 0.3]}


def _audit(tmp_path: Path, spec: dict[str, object]) -> tuple[int, str]:
    path = tmp_path / "scores.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    result = CliRunner().invoke(
        main,
        [
            "audit-detector",
            str(path),
            "--n-permutations",
            "50",
            "--corpus-id",
            "corpus-1",
            "--captured-at",
            "2026-09-24T00:00:00Z",
        ],
    )
    return result.exit_code, result.output


@pytest.mark.parametrize("name", [None, 7, "", ["d"]])
def test_non_string_detector_name_is_refused(tmp_path: Path, name: object) -> None:
    code, output = _audit(tmp_path, {**_SCORES, "detector_name": name})
    assert code != 0
    assert "'detector_name' must be a non-empty string" in output


def test_default_and_explicit_names_are_sealed(tmp_path: Path) -> None:
    code, output = _audit(tmp_path, dict(_SCORES))
    assert code == 0, output
    assert '"detector": "detector"' in output or '"detector_name": "detector"' in output
    code, output = _audit(tmp_path, {**_SCORES, "detector_name": "sigma-rate"})
    assert code == 0, output
    assert "sigma-rate" in output
