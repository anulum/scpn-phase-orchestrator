# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — supervisor-candidate flags and labels are not coerced

"""The CLI must not coerce what the autotune dataclasses would refuse.

``RewardObservation`` and ``SafetyConstraintConfig`` refuse a non-boolean
flag, but the CLI applied ``bool()`` first: ``"require_stl": ""`` silently
disabled a safety requirement and ``"unsafe": "false"`` turned a safe tick
unsafe. ``str(None)`` put a backend named "None" into the sealed provenance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from scpn_phase_orchestrator.runtime.cli import main


def _scenario() -> dict[str, Any]:
    return {
        "candidate": {"alpha": 0.2, "zeta": 0.05, "channel_weights": [1.0]},
        "baseline": {"alpha": 0.0, "zeta": 0.0, "channel_weights": [0.0]},
        "incumbent": {"alpha": 0.1, "zeta": 0.02, "channel_weights": [0.5]},
        "observations": [
            {"coherence": 0.82, "previous_coherence": 0.74, "lyapunov_exponent": -0.02}
        ],
        "constraints": {"max_lyapunov_exponent": 0.0},
        "safety_tier": "research",
        "numeric_provenance": {"active_backend": "python", "parity_tolerance": 1e-9},
    }


def _run(tmp_path: Path, scenario: dict[str, Any]) -> tuple[int, str]:
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(scenario), encoding="utf-8")
    result = CliRunner().invoke(main, ["supervisor-candidate", str(path)])
    return result.exit_code, result.output


def test_real_boolean_flags_are_accepted(tmp_path: Path) -> None:
    scenario = _scenario()
    scenario["observations"][0]["unsafe"] = False
    scenario["constraints"]["require_lyapunov"] = True
    code, output = _run(tmp_path, scenario)
    assert code == 0, output


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("constraints", "require_stl", ""),
        ("constraints", "require_safety_cost", "false"),
        ("constraints", "require_lyapunov", 1),
        ("observation", "unsafe", "false"),
        ("observation", "regime_changed", "no"),
    ],
)
def test_text_or_numeric_flags_are_refused(
    tmp_path: Path, section: str, key: str, value: object
) -> None:
    scenario = _scenario()
    target = (
        scenario["observations"][0] if section == "observation" else scenario[section]
    )
    target[key] = value
    code, output = _run(tmp_path, scenario)
    assert code != 0
    assert f"{key} must be a JSON boolean" in output


@pytest.mark.parametrize(
    ("path", "value"),
    [(("numeric_provenance", "active_backend"), None), (("safety_tier",), 7)],
)
def test_non_string_labels_are_refused(
    tmp_path: Path, path: tuple[str, ...], value: object
) -> None:
    scenario = _scenario()
    target = scenario
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    code, output = _run(tmp_path, scenario)
    assert code != 0
    assert f"{path[-1]} must be a non-empty string" in output
