# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — public documentation release-contract drift guards

"""Keep primary public documentation aligned with package release metadata."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
DOCS_HOME = ROOT / "docs" / "index.md"


def _project_metadata() -> dict[str, object]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = data["project"]
    assert isinstance(project, dict)
    return project


def _public_markdown_paths() -> list[Path]:
    paths = [
        path for path in (ROOT / "docs").rglob("*.md") if "internal" not in path.parts
    ]
    paths.extend(
        path for path in ROOT.glob("*.md") if path.name not in {"CHANGELOG.md"}
    )
    return sorted(paths)


def test_docs_home_names_the_package_release() -> None:
    version = _project_metadata()["version"]
    assert isinstance(version, str)
    home = DOCS_HOME.read_text(encoding="utf-8")
    assert f"Version `{version}`" in home


def test_public_docs_do_not_advertise_unsupported_python_310() -> None:
    unsupported = re.compile(r"Python\s+3\.10|python-3\.10", re.IGNORECASE)
    offenders: list[str] = []
    for path in _public_markdown_paths():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if unsupported.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}")
    assert not offenders, f"public docs advertise unsupported Python 3.10: {offenders}"


def test_primary_install_docs_name_supported_python_range() -> None:
    requires_python = _project_metadata()["requires-python"]
    assert requires_python == ">=3.11,<3.14"
    expected = "Python 3.11–3.13"
    for relative in (
        "README.md",
        "docs/getting-started/installation.md",
        "docs/getting-started/troubleshooting.md",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert expected in text, f"{relative} omits {expected}"
