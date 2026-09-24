# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — scaffold never discards an existing binding spec

"""``spo scaffold --llm`` must not replace an existing pack's binding spec.

The plain scaffold keeps an existing ``binding_spec.yaml``; the LLM scaffold
overwrote it wholesale, discarding a hand-tuned pack. ``re.match`` with
``^...$`` also accepted a name ending in a newline and created that directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from scpn_phase_orchestrator.runtime.cli import main

_RESPONSE = {
    "name": "traffic_grid",
    "sample_period_s": 1.0,
    "control_period_s": 5.0,
    "oscillators": [
        {"id": "north_south", "channel": "I", "extractor_type": "event", "omega": 0.9},
        {"id": "east_west", "channel": "I", "extractor_type": "event", "omega": 1.1},
    ],
    "coupling": {"base_strength": 0.22, "decay_alpha": 0.18},
}


def _llm_scaffold(tmp_path: Path, response: dict[str, object]) -> tuple[int, str]:
    response_path = tmp_path / "response.json"
    response_path.write_text(json.dumps(response), encoding="utf-8")
    result = CliRunner().invoke(
        main,
        [
            "scaffold",
            "traffic_grid",
            "--llm",
            "--description",
            "traffic lights",
            "--llm-response-json",
            str(response_path),
        ],
    )
    return result.exit_code, result.output


def test_llm_scaffold_keeps_an_existing_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the response is valid: it scaffolds the pack where none exists yet
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    monkeypatch.chdir(fresh)
    fresh_code, fresh_output = _llm_scaffold(fresh, _RESPONSE)
    assert fresh_code == 0, fresh_output

    existing = tmp_path / "existing"
    pack = existing / "domainpacks" / "traffic_grid"
    pack.mkdir(parents=True)
    tuned = "# hand-tuned\nname: traffic_grid\n"
    (pack / "binding_spec.yaml").write_text(tuned, encoding="utf-8")
    monkeypatch.chdir(existing)
    code, output = _llm_scaffold(existing, _RESPONSE)
    assert code != 0
    assert "already exists" in output
    assert (pack / "binding_spec.yaml").read_text(encoding="utf-8") == tuned
    assert not (pack / "llm_scaffold_audit.json").exists()


@pytest.mark.parametrize("name", ["bad\n", "bad name", "../escape", ""])
def test_scaffold_refuses_names_outside_the_pattern(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(main, ["scaffold", name])
    assert result.exit_code != 0
    assert not (tmp_path / "domainpacks").exists()
