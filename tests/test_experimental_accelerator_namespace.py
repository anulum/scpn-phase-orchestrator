# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — accelerator namespace boundary tests

"""Contract tests for the historical ``experimental`` accelerator namespace."""

from __future__ import annotations

from pathlib import Path

import scpn_phase_orchestrator.experimental as experimental
from scpn_phase_orchestrator.experimental import accelerators
from scpn_phase_orchestrator.experimental.accelerators import coupling, monitor, upde


def _repo_root() -> Path:
    """Return the repository root containing this test file."""
    return Path(__file__).resolve().parents[1]


def test_experimental_namespace_readme_defines_load_bearing_boundary() -> None:
    """The source-local README must describe the actual dispatcher contract."""
    readme = (
        _repo_root()
        / "src"
        / "scpn_phase_orchestrator"
        / "experimental"
        / "README.md"
    )

    text = readme.read_text(encoding="utf-8")

    assert "historical package name" in text
    assert "load-bearing polyglot backend implementations" in text
    assert "Use the owning production API" in text
    assert "validation failures" in text.lower()
    assert "import backend files directly" in text


def test_experimental_namespace_docstrings_preserve_private_api_boundary() -> None:
    """Package docstrings should not present backend bridges as public APIs."""
    package_docs = "\n".join(
        (
            experimental.__doc__ or "",
            accelerators.__doc__ or "",
            coupling.__doc__ or "",
            monitor.__doc__ or "",
            upde.__doc__ or "",
        )
    )
    normalised_docs = package_docs.lower()

    assert "historical" in normalised_docs
    assert "load-bearing" in normalised_docs
    assert "private" in normalised_docs
    assert "not standalone user-facing APIs" in package_docs
