# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — public learning-inventory drift guards

"""Keep notebook, example, and API documentation inventories reproducible."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "notebooks"
EXAMPLE_ROOT = ROOT / "examples"
API_ROOT = ROOT / "docs" / "reference" / "api"
GALLERY = ROOT / "docs" / "galleries" / "notebooks_and_demos.md"
MATRIX = ROOT / "docs" / "galleries" / "notebook_execution_matrix.md"
COVERAGE = ROOT / "docs" / "reference" / "documentation_coverage.md"
ROADMAP = ROOT / "docs" / "roadmap.md"
VALIDATION = ROOT / "VALIDATION.md"


def _notebooks() -> tuple[Path, ...]:
    return tuple(sorted(NOTEBOOK_ROOT.glob("*.ipynb")))


def _top_level_examples() -> tuple[Path, ...]:
    return tuple(sorted(EXAMPLE_ROOT.glob("*.py")))


def _api_pages() -> tuple[Path, ...]:
    return tuple(sorted(API_ROOT.glob("*.md")))


def test_public_inventory_counts_are_derived_from_the_tree() -> None:
    notebook_count = len(_notebooks())
    example_count = len(_top_level_examples())
    api_count = len(_api_pages())

    assert notebook_count == 21
    assert example_count == 28
    assert api_count == 96

    gallery = GALLERY.read_text(encoding="utf-8")
    matrix = MATRIX.read_text(encoding="utf-8")
    coverage = COVERAGE.read_text(encoding="utf-8")
    roadmap = ROADMAP.read_text(encoding="utf-8")
    validation = VALIDATION.read_text(encoding="utf-8")

    assert f"All {notebook_count} committed notebooks" in matrix
    assert f"{notebook_count} notebook workflows" in coverage
    assert f"{notebook_count} notebooks" in roadmap
    assert f"| Notebook execution | {notebook_count} |" in validation
    assert f"There are `{example_count}` terminal-first" in gallery
    assert f"`{example_count}` top-level `examples/*.py`" in coverage
    assert f"{example_count} terminal examples" in roadmap
    assert f"{api_count} MkDocs API pages" in coverage
    assert f"`{api_count}` Markdown files" in coverage


def test_every_notebook_is_listed_as_ci_executed() -> None:
    matrix = MATRIX.read_text(encoding="utf-8")
    gallery = GALLERY.read_text(encoding="utf-8")
    for path in _notebooks():
        assert f"`{path.name}`" in matrix
        assert f"`{path.name}`" in gallery
        matrix_row = next(
            line for line in matrix.splitlines() if f"`{path.name}`" in line
        )
        assert "| executed |" in matrix_row


def test_committed_notebooks_have_clean_python_metadata() -> None:
    for path in _notebooks():
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert notebook["metadata"]["kernelspec"]["name"] == "python3"

        cells = notebook["cells"]
        first_markdown = next(cell for cell in cells if cell["cell_type"] == "markdown")
        source = first_markdown["source"]
        text = "".join(source) if isinstance(source, list) else source
        assert text.lstrip().startswith("# "), f"{path.name} lacks an opening title"

        code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
        assert code_cells, f"{path.name} has no executable cells"
        assert all(cell["execution_count"] is None for cell in code_cells)
        assert all(cell["outputs"] == [] for cell in code_cells)
