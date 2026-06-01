# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — NDArray type hygiene tests

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "scpn_phase_orchestrator"


def _contains_bare_ndarray_annotation(expr: ast.AST, *, root: bool = True) -> bool:
    if isinstance(expr, ast.Name):
        return root and expr.id == "NDArray"

    for child in ast.iter_child_nodes(expr):
        if _contains_bare_ndarray_annotation(child, root=False):
            return True

    return False


def _contains_numpy_ndarray_annotation(expr: ast.AST) -> bool:
    if (
        isinstance(expr, ast.Attribute)
        and expr.attr == "ndarray"
        and isinstance(expr.value, ast.Name)
        and expr.value.id == "np"
    ):
        return True

    return any(
        _contains_numpy_ndarray_annotation(child)
        for child in ast.iter_child_nodes(expr)
    )


def _is_ndarray_annotation_violation(expr: ast.AST) -> bool:
    return _contains_bare_ndarray_annotation(
        expr,
        root=True,
    ) or _contains_numpy_ndarray_annotation(expr)


def _scan_annotation_text(annotation_text: str) -> bool:
    stripped = annotation_text.strip()
    if not stripped:
        return False

    if not stripped.startswith(('"', "'")):
        try:
            parsed = ast.parse(stripped, mode="eval").body
        except SyntaxError:
            return False
    else:
        parsed = ast.parse(stripped, mode="eval").body
        # The string can still parse to expression; if not parseable above,
        # fallback to plain string inspection.
        # pragma: no cover - defensive path for unusual annotation syntax.
        if isinstance(parsed, ast.Constant) and isinstance(parsed.value, str):
            parsed = ast.parse(parsed.value, mode="eval").body

    return _is_ndarray_annotation_violation(parsed)


def _iter_offending_annotations(
    tree: ast.AST, path: Path
) -> list[tuple[int, int, str]]:
    offenders: list[tuple[int, int, str]] = []

    for node in ast.walk(tree):
        annotation = getattr(node, "annotation", None)
        if annotation is not None and _is_ndarray_annotation_violation(annotation):
            offenders.append(
                (
                    getattr(annotation, "lineno", 1),
                    getattr(node, "col_offset", 0) or 0,
                    f"{path}:{getattr(annotation, 'lineno', 1)}",
                )
            )

        returns = getattr(node, "returns", None)
        if returns is not None and _is_ndarray_annotation_violation(returns):
            offenders.append(
                (
                    getattr(returns, "lineno", 1),
                    getattr(node, "col_offset", 0) or 0,
                    f"{path}:{getattr(returns, 'lineno', 1)}",
                )
            )

        type_comment = getattr(node, "type_comment", None)
        if isinstance(type_comment, str) and _scan_annotation_text(type_comment):
            offenders.append(
                (
                    getattr(node, "lineno", 1),
                    getattr(node, "col_offset", 0) or 0,
                    f"{path}:{getattr(node, 'lineno', 1)}",
                )
            )

        if isinstance(node, ast.Call):
            fn = node.func
            if (
                isinstance(fn, ast.Name)
                and fn.id == "cast"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and _scan_annotation_text(node.args[0].value)
            ):
                first = node.args[0]
                offenders.append(
                    (
                        getattr(first, "lineno", 1),
                        getattr(first, "col_offset", 0) or 0,
                        f"{path}:{getattr(first, 'lineno', 1)}",
                    )
                )

    return offenders


def test_no_bare_ndarray_annotations_in_src() -> None:
    """
    Enforce the typed-NumPy maintenance sweep contract in src.

    Array annotations should use parameterised `NDArray[...]` aliases, not bare
    `NDArray` or runtime-only `np.ndarray` annotations.
    """
    offenders: list[str] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for _line, _col, location in _iter_offending_annotations(tree, path):
            offenders.append(location)

    if offenders:
        pytest.fail(
            "Unparameterised NDArray annotations remain in src: "
            + ", ".join(sorted(set(offenders)))
        )
