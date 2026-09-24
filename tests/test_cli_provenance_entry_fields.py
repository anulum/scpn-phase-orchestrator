# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — provenance spec entries need real string fields

"""A missing subject name must not be attested as the artefact ``"None"``.

``str(entry.get("name"))`` turned a missing field into the text ``"None"``
(and a number into its digits), which the signed SLSA statement then carried
as the subject name or dependency URI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from scpn_phase_orchestrator.runtime.cli import main
from tests.test_audit_pqc import requires_mldsa

_A = "a" * 64


def _spec(**overrides: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "subjects": [{"name": "scpn-1.0-py3-none-any.whl", "sha256": _A}],
        "build_type": "https://slsa.dev/build/pypi@v1",
        "external_parameters": {"ref": "refs/tags/v1.0"},
        "resolved_dependencies": [
            {"uri": "git+https://github.com/anulum/spo@" + "b" * 40, "sha256": "c" * 64}
        ],
        "builder_id": "https://github.com/anulum/spo/ci",
        "invocation_id": "run-42",
    }
    spec.update(overrides)
    return spec


def _attest(tmp_path: Path, spec: dict[str, Any]) -> tuple[int, str]:
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    result = CliRunner().invoke(
        main, ["provenance-attest", str(path), "--signing-seed", "00" * 32]
    )
    return result.exit_code, result.output


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"subjects": [{"sha256": _A}]}, id="subject-without-name"),
        pytest.param({"subjects": [{"name": 42, "sha256": _A}]}, id="numeric-name"),
        pytest.param({"subjects": [{"name": None, "sha256": _A}]}, id="null-name"),
        pytest.param(
            {"resolved_dependencies": [{"sha256": "c" * 64}]}, id="dependency-no-uri"
        ),
        pytest.param(
            {
                "resolved_dependencies": [
                    {"uri": "git+x", "sha256": "c" * 64, "name": 7}
                ]
            },
            id="dependency-numeric-name",
        ),
    ],
)
def test_entry_without_a_string_field_is_refused(
    tmp_path: Path, overrides: dict[str, Any]
) -> None:
    code, output = _attest(tmp_path, _spec(**overrides))
    assert code != 0
    assert "needs a non-empty string" in output
    assert '"None"' not in output


@requires_mldsa
def test_optional_descriptor_name_may_be_absent_or_empty(tmp_path: Path) -> None:
    for entry in (
        {"uri": "git+x", "sha256": "c" * 64},
        {"uri": "git+x", "sha256": "c" * 64, "name": ""},
    ):
        code, output = _attest(tmp_path, _spec(resolved_dependencies=[entry]))
        assert code == 0, output
