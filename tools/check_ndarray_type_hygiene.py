#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — NDArray type hygiene gate

from __future__ import annotations

import ast
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "scpn_phase_orchestrator"


def _contains_bare_ndarray_annotation(expr: ast.AST, *, root: bool = True) -> bool:
    if isinstance(expr, ast.Name):
        return root and expr.id == "NDArray"

    for child in ast.iter_child_nodes(expr):
        if _contains_bare_ndarray_annotation(child, root=False):
            return True

    return False


def _scan_annotation_text(annotation_text: str) -> bool:
    stripped = annotation_text.strip()
    if not stripped:
        return False

    try:
        parsed = ast.parse(stripped, mode="eval").body
    except SyntaxError:
        return False

    if isinstance(parsed, ast.Constant) and isinstance(parsed.value, str):
        try:
            parsed = ast.parse(parsed.value, mode="eval").body
        except SyntaxError:
            return False

    return _contains_bare_ndarray_annotation(parsed, root=True)


def _iter_offending_annotations(tree: ast.AST, path: Path) -> list[str]:
    offenders: list[str] = []

    for node in ast.walk(tree):
        annotation = getattr(node, "annotation", None)
        if annotation is not None and _contains_bare_ndarray_annotation(
            annotation, root=True
        ):
            offenders.append(f"{path}:{getattr(annotation, 'lineno', 1)}")

        returns = getattr(node, "returns", None)
        if returns is not None and _contains_bare_ndarray_annotation(
            returns, root=True
        ):
            offenders.append(f"{path}:{getattr(returns, 'lineno', 1)}")

        type_comment = getattr(node, "type_comment", None)
        if isinstance(type_comment, str) and _scan_annotation_text(type_comment):
            offenders.append(f"{path}:{getattr(node, 'lineno', 1)}")

        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "cast"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and _scan_annotation_text(node.args[0].value)
        ):
            offenders.append(f"{path}:{getattr(node.args[0], 'lineno', 1)}")

    return offenders


def main() -> int:
    offenders: set[str] = set()
    for path in sorted(SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        offenders.update(_iter_offending_annotations(tree, path))

    if offenders:
        print("FAIL: unparameterized NDArray annotations found in src/")
        for loc in sorted(offenders):
            print(f"  - {loc}")
        print(
            "Hint: migrate bare NDArray to NDArray[np.*] or adjust runtime-only usage."
        )
        return 1

    print("OK: all NDArray annotations in src/ are parameterized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
