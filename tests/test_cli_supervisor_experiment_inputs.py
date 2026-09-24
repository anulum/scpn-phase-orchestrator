# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — supervisor experiment input contracts

"""``spo supervisor-baseline-experiment`` input contracts.

``json.loads`` accepts ``NaN``/``Infinity`` and ``nan <= 0`` is false, so a
non-finite scenario value passed validation and failed later with an
unrelated message after the experiment ran. A dependency-lock label given
twice with different digests silently kept only the last one.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from scpn_phase_orchestrator.runtime.cli import main
from scpn_phase_orchestrator.runtime.cli.verification import _parse_dependency_locks

_BASE = {
    "phases": [0.0, 0.1, 2.7, 3.1],
    "omegas": [0.04, 0.03, -0.03, -0.04],
    "base_coupling_off_diagonal": 0.03,
    "good_mask": [1.0, 1.0, 0.0, 0.0],
    "bad_mask": [0.0, 0.0, 1.0, 1.0],
    "dt": 0.05,
    "inner_steps": 4,
    "horizon": 6,
}


def _invoke(
    tmp_path: Path, scenario: dict[str, object], *locks: str
) -> tuple[int, str]:
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(json.dumps(scenario), encoding="utf-8")
    lock_args = [
        arg for lock in (locks or ("uv:aaa",)) for arg in ("--dependency-lock", lock)
    ]
    result = CliRunner().invoke(
        main,
        [
            "supervisor-baseline-experiment",
            "--scenario-json",
            str(scenario_path),
            "--config-json",
            str(tmp_path / "config.json"),
            "--metrics-jsonl",
            str(tmp_path / "metrics.jsonl"),
            "--summary-json",
            str(tmp_path / "summary.json"),
            "--git-sha",
            "abc1234",
            "--seed",
            "0",
            *lock_args,
        ],
    )
    return result.exit_code, result.output


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dt", math.nan),
        ("dt", math.inf),
        ("base_coupling_off_diagonal", math.nan),
        ("phases", [0.0, math.nan, 2.7, 3.1]),
        ("omegas", [0.04, math.inf, -0.03, -0.04]),
    ],
)
def test_non_finite_scenario_value_is_refused_before_running(
    tmp_path: Path, field: str, value: object
) -> None:
    scenario = dict(_BASE)
    scenario[field] = value
    code, output = _invoke(tmp_path, scenario)
    assert code != 0
    assert f"scenario {field}" in output and "finite" in output
    assert not (tmp_path / "summary.json").exists()


def test_conflicting_dependency_locks_are_refused(tmp_path: Path) -> None:
    code, output = _invoke(tmp_path, dict(_BASE), "uv:aaa", "uv:bbb")
    assert code != 0
    assert "given twice with different digests" in output
    assert not (tmp_path / "config.json").exists()


def test_repeated_identical_lock_is_accepted() -> None:
    assert _parse_dependency_locks(("uv:aaa", "uv:aaa", "pip:ccc")) == {
        "uv": "aaa",
        "pip": "ccc",
    }
    with pytest.raises(click.ClickException, match="different digests"):
        _parse_dependency_locks(("uv:aaa", "uv:bbb"))
