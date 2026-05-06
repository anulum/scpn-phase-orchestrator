# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — API documentation navigation regressions

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import yaml

DOCS_ROOT = Path("docs")
API_ROOT = DOCS_ROOT / "reference" / "api"
MKDOCS_CONFIG = Path("mkdocs.yml")


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
