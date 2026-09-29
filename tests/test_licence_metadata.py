# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for the repository licence declarations

"""The repository must state one licence, machine-readably, on every surface.

Licence scanners, package indexes and dataset builders read the root
``LICENSE`` and the package metadata only. A root ``LICENSE`` that is not the
standard text is reported as ``NOASSERTION`` and the project is treated as
unlicensed, so the text outside the "How to Apply" notice must be the FSF
original byte for byte, and the notice must name this project.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Any, cast

import yaml

ROOT = Path(__file__).resolve().parents[1]
LICENCE = "AGPL-3.0-or-later"

# Canonical GNU AGPL-3.0 text, https://www.gnu.org/licenses/agpl-3.0.txt,
# read on 2026-09-29.
FSF_AGPL_SHA256 = "0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0"
_FSF_NOTICE_PLACEHOLDER = (
    "    <one line to give the program's name and a brief idea of what it does.>\n"
    "    Copyright (C) <year>  <name of author>\n"
)
_NOTICE_START = (
    'the "copyright" line and a pointer to where the full notice is found.\n\n'
)
_NOTICE_END = (
    "\n\n    This program is free software: you can redistribute it and/or modify\n"
)
_SPDX_TAG = "SPDX-License-" + "Identifier:"
_RETIRED_START_YEAR = chr(0xA9) + " 1998"

# Files that quote the retired ``X | text`` expression on purpose: the header
# normaliser that rewrites it, its fixtures, and the changelog entry recording it.
_QUOTES_PIPED_SPDX = frozenset(
    {
        "CHANGELOG.md",
        "tools/normalise_spdx_headers.py",
        "tests/test_tools_normalise_spdx_io.py",
    }
)


def _read(*parts: str) -> str:
    """Return a repository file as text."""
    return ROOT.joinpath(*parts).read_text(encoding="utf-8")


def _split_license_notice() -> tuple[str, list[str]]:
    """Return LICENSE with the FSF placeholder restored, and the notice lines."""
    text = _read("LICENSE")
    start = text.index(_NOTICE_START) + len(_NOTICE_START)
    end = text.index(_NOTICE_END, start)
    standard = text[:start] + _FSF_NOTICE_PLACEHOLDER + text[end + 1 :]
    return standard, text[start:end].splitlines()


def test_root_license_is_the_standard_agpl_text_with_this_project_notice() -> None:
    """Scanners must match the standard text; only the notice names this project."""
    standard, notice = _split_license_notice()

    assert hashlib.sha256(standard.encode("utf-8")).hexdigest() == FSF_AGPL_SHA256
    assert notice[0].startswith("    SCPN Phase Orchestrator — ")
    for line in (
        "© Concepts 1996–2026 Miroslav Šotek. All rights reserved.",
        "© Code 2020–2026 Miroslav Šotek. All rights reserved.",
        f"{_SPDX_TAG} {LICENCE}",
    ):
        assert f"    {line}" in notice
    for other_project in ("Fusion", "Quantum", "SCPN Control", "Studio"):
        assert all(other_project not in line for line in notice)


def test_licence_is_declared_identically_on_every_metadata_surface() -> None:
    """LICENSES/, NOTICE and every package manifest state the same licence."""
    pyproject = tomllib.loads(_read("pyproject.toml"))
    ffi = tomllib.loads(_read("spo-kernel", "crates", "spo-ffi", "pyproject.toml"))
    citation = cast("dict[str, Any]", yaml.safe_load(_read("CITATION.cff")))
    zenodo = json.loads(_read(".zenodo.json"))
    studio_web = json.loads(_read("studio-web", "package.json"))
    workspace = tomllib.loads(_read("spo-kernel", "Cargo.toml"))
    crates = sorted((ROOT / "spo-kernel" / "crates").glob("*/Cargo.toml"))
    notice = _read("NOTICE.md")

    assert (ROOT / "LICENSES" / f"{LICENCE}.txt").is_file()
    assert pyproject["project"]["license"] == LICENCE
    assert ffi["project"]["license"] == {"text": LICENCE}
    declared = [citation["license"], zenodo["license"], studio_web["license"]]
    assert declared == [LICENCE] * 3
    assert workspace["workspace"]["package"]["license"] == LICENCE
    assert len(crates) >= 4
    for manifest in crates:
        package = tomllib.loads(manifest.read_text(encoding="utf-8"))["package"]
        assert package["license"] in (LICENCE, {"workspace": True}), manifest
    assert "© Concepts 1996–2026 Miroslav Šotek." in notice
    assert "© Code 2020–2026 Miroslav Šotek." in notice


def test_tracked_files_use_valid_spdx_lines_and_the_canonical_start_year() -> None:
    """Tracked files carry no piped SPDX expression and no retired 1998 start year."""
    git = shutil.which("git")
    assert git is not None
    listed = subprocess.run(
        [git, "-C", str(ROOT), "ls-files", "-z"], check=True, stdout=subprocess.PIPE
    ).stdout
    piped_spdx = re.compile(re.escape(_SPDX_TAG) + r"[^\n|]*\|")
    offenders: list[str] = []
    for name in listed.decode("utf-8").split("\0"):
        path = ROOT / name
        if not name or not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        text = data.decode("utf-8", errors="replace")
        piped = name not in _QUOTES_PIPED_SPDX and piped_spdx.search(text)
        if piped or _RETIRED_START_YEAR in text:
            offenders.append(name)

    assert offenders == []
    assert all((ROOT / name).is_file() for name in _QUOTES_PIPED_SPDX)
