# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — formal evidence cites only the package it was sealed as

"""An edited formal-verification manifest must not become assurance evidence.

``FormalVerificationPackage`` seals its manifest as the canonical SHA-256 of
every other field, but the evidence builder only required ``package_hash`` to
be a non-empty string, so a manifest with an added or altered property was
cited as evidence under the original package's hash.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from scpn_phase_orchestrator.assurance import build_formal_verification_evidence
from scpn_phase_orchestrator.runtime.cli import main
from scpn_phase_orchestrator.supervisor.formal_export import (
    FormalSafetyProperty,
    FormalTextArtifact,
    build_formal_verification_package,
)


def _manifest() -> dict[str, object]:
    package = build_formal_verification_package(
        {"safety": FormalTextArtifact("smt2", "(assert (= x x))")},
        [
            FormalSafetyProperty(
                name="bounded",
                artifact_name="safety",
                checker="smt",
                expression="(check-sat)",
            )
        ],
    )
    return package.to_audit_record()


def test_real_package_is_accepted() -> None:
    item = build_formal_verification_evidence(_manifest())
    assert item.evidence_id == "formal-verification-package"


@pytest.mark.parametrize(
    "edit",
    [
        lambda m: m["properties"].append(dict(m["properties"][0], name="extra")),
        lambda m: m.__setitem__("package_name", "renamed"),
        lambda m: m["artifact_hashes"].__setitem__("safety", "0" * 64),
    ],
    ids=["added-property", "renamed", "artifact-hash"],
)
def test_edited_manifest_is_refused(edit) -> None:  # type: ignore[no-untyped-def]
    manifest = _manifest()
    edit(manifest)
    with pytest.raises(ValueError, match="package_hash does not match its content"):
        build_formal_verification_evidence(manifest)


def test_assurance_case_cli_refuses_an_edited_package(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["package_name"] = "renamed"
    path = tmp_path / "formal.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    result = CliRunner().invoke(
        main, ["assurance-case", "--system", "Sys", "--formal-package", str(path)]
    )
    assert result.exit_code != 0
    assert "package_hash does not match its content" in result.output
