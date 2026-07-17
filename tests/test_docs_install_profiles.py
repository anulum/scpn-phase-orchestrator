# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — install-profiles PyPI-boundary drift guard

"""Keep the install guide's PyPI-availability boundary honest and in step.

``docs/guide/install_profiles.md`` states that a few extras pull in packages that
are not on public PyPI (verified at source 2026-07-11: ``spo-kernel`` and
``scpn-fusion-core`` both 404). This drift guard derives, from ``pyproject.toml``,
exactly which extras depend on one of those private packages, and fails if the
guide stops naming any such extra — so the honest boundary can never silently rot
back into the misleading "just install the extra" framing (E0.1 / Option C).
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "guide" / "install_profiles.md"
PYPROJECT = ROOT / "pyproject.toml"

# Verified at source 2026-07-11: neither distribution is on public PyPI
# (``curl https://pypi.org/pypi/<name>/json`` -> HTTP 404). ``spo-kernel`` is the
# project's own in-repo Rust accel; ``scpn-fusion-core`` is a separate product.
PRIVATE_PACKAGES = frozenset({"spo-kernel", "scpn-fusion-core"})


def _distribution_name(requirement: str) -> str:
    """Return the bare distribution name from a requirement specifier."""
    return re.split(r"[<>=!~;\[ ]", requirement, maxsplit=1)[0].strip()


def _extras_needing_private_packages() -> dict[str, set[str]]:
    """Map each extra that requires a private package to those package names."""
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    mapping: dict[str, set[str]] = {}
    for extra, requirements in extras.items():
        private = {
            _distribution_name(req)
            for req in requirements
            if _distribution_name(req) in PRIVATE_PACKAGES
        }
        if private:
            mapping[extra] = private
    return mapping


def test_pyproject_still_has_private_backed_extras() -> None:
    # Guards the guard: if this ever empties, the boundary premise changed and the
    # guide section (and this test) should be revisited, not silently passing.
    assert _extras_needing_private_packages(), (
        "no extra depends on a private package — re-verify the PyPI boundary"
    )


def test_guide_names_every_private_backed_extra() -> None:
    doc = GUIDE.read_text(encoding="utf-8")
    missing = {
        extra for extra in _extras_needing_private_packages() if f"`{extra}`" not in doc
    }
    assert not missing, f"install guide omits private-backed extras: {missing}"


def test_guide_names_every_private_package() -> None:
    doc = GUIDE.read_text(encoding="utf-8")
    named = {
        pkg for pkgs in _extras_needing_private_packages().values() for pkg in pkgs
    }
    missing = {pkg for pkg in named if f"`{pkg}`" not in doc}
    assert not missing, f"install guide omits private packages: {missing}"


def test_guide_states_the_boundary_honestly() -> None:
    # Collapse emphasis markers and case so a bold/capitalised phrasing still
    # matches; the honest phrasing and the pure-Python fallback caveat must remain.
    normalised = " ".join(GUIDE.read_text(encoding="utf-8").replace("*", "").split())
    normalised = normalised.lower()
    assert "not on public pypi" in normalised
    assert "pure-python" in normalised


_PIP_INSTALL_SPO_KERNEL = re.compile(r"pip install[^\n`]*\bspo-kernel\b(?!/)")


def _public_doc_paths() -> list[Path]:
    """Return every shipped Markdown doc plus the README (excluding docs/internal)."""
    paths = [p for p in (ROOT / "docs").rglob("*.md") if "internal" not in p.parts]
    readme = ROOT / "README.md"
    if readme.exists():
        paths.append(readme)
    return paths


def test_no_public_doc_advertises_pip_install_spo_kernel() -> None:
    # `spo-kernel` is 404 on public PyPI, so `pip install ... spo-kernel` is a false
    # instruction wherever it appears in shipped docs (E0.1 / Option C). A path like
    # `spo-kernel/` (build-from-source) is fine; a bare package argument is not.
    offenders: list[str] = []
    for path in _public_doc_paths():
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            if _PIP_INSTALL_SPO_KERNEL.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}")
    assert not offenders, f"public docs advertise `pip install spo-kernel`: {offenders}"
