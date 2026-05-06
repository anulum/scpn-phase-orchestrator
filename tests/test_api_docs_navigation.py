# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — API documentation navigation regressions

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import yaml

DOCS_ROOT = Path("docs")
API_ROOT = DOCS_ROOT / "reference" / "api"
MKDOCS_CONFIG = Path("mkdocs.yml")
SOURCE_ROOT = Path("src") / "scpn_phase_orchestrator"
AUTODOC_EXCLUSIONS = {
    "scpn_phase_orchestrator.grpc_gen.spo_pb2",
    "scpn_phase_orchestrator.grpc_gen.spo_pb2_grpc",
}


def _nav_paths(nav: object) -> Iterator[str]:
    if isinstance(nav, str):
        yield nav
        return
    if isinstance(nav, list):
        for item in nav:
            yield from _nav_paths(item)
        return
    if isinstance(nav, dict):
        for value in nav.values():
            yield from _nav_paths(value)


def test_all_api_reference_pages_are_in_mkdocs_nav() -> None:
    config_text = MKDOCS_CONFIG.read_text(encoding="utf-8")
    for tag in (
        "!!python/name:pymdownx.superfences.fence_code_format",
        "!!python/name:material.extensions.emoji.twemoji",
        "!!python/name:material.extensions.emoji.to_svg",
    ):
        config_text = config_text.replace(tag, "")
    config = yaml.safe_load(config_text)
    nav_paths = set(_nav_paths(config["nav"]))
    api_pages = {
        str(path.relative_to(DOCS_ROOT))
        for path in API_ROOT.glob("*.md")
    }

    assert api_pages <= nav_paths


def test_public_source_modules_have_api_autodoc() -> None:
    autodoc_text = "\n".join(
        path.read_text(encoding="utf-8") for path in API_ROOT.glob("*.md")
    )
    documented_modules = {
        match.group(1)
        for match in re.finditer(
            r"^:{3,4}\s+(scpn_phase_orchestrator(?:\.[\w_]+)+)",
            autodoc_text,
            re.MULTILINE,
        )
    }

    public_modules = set()
    for path in SOURCE_ROOT.rglob("*.py"):
        relative = path.relative_to(SOURCE_ROOT)
        if relative.name == "__init__.py":
            continue
        module_parts = relative.with_suffix("").parts
        if any(part.startswith("_") for part in module_parts):
            continue
        public_modules.add("scpn_phase_orchestrator." + ".".join(module_parts))

    missing = public_modules - documented_modules - AUTODOC_EXCLUSIONS

    assert missing == set()
