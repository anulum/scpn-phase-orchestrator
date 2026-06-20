# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Module size review guard tests

"""Tests for ``tools/check_module_size.py``.

Cover module discovery and line counting, tier classification, allowlist
loading (including the non-empty-reason contract), the ``--check`` ratchet, and
the live repository state (no un-allowlisted module above the split threshold,
which the god-file refactor campaign established).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_tool() -> Any:
    tool_path = _repo_root() / "tools" / "check_module_size.py"
    spec = importlib.util.spec_from_file_location("check_module_size", tool_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so the @dataclass decorator can resolve __module__.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, lines: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join("x = 1" for _ in range(lines)) + "\n", encoding="utf-8")


def _make_tree(root: Path, sizes: dict[str, int]) -> Path:
    src = root / "src" / "scpn_phase_orchestrator"
    for rel, count in sizes.items():
        _write(src / rel, count)
    return src


def test_collect_modules_counts_lines_and_excludes_generated(tmp_path: Path) -> None:
    tool = _load_tool()
    src = _make_tree(
        tmp_path,
        {"small.py": 10, "pkg/big.py": 1300, "grpc_gen/spo_pb2.py": 5000},
    )

    modules = tool.collect_modules(src)
    by_path = {m.dotted_path: m for m in modules}

    assert "scpn_phase_orchestrator.grpc_gen.spo_pb2" not in by_path
    assert by_path["scpn_phase_orchestrator.small"].line_count == 10
    assert by_path["scpn_phase_orchestrator.pkg.big"].line_count == 1300


def test_package_init_dotted_path_drops_init(tmp_path: Path) -> None:
    tool = _load_tool()
    src = _make_tree(tmp_path, {"pkg/__init__.py": 5})

    modules = tool.collect_modules(src)

    assert modules[0].dotted_path == "scpn_phase_orchestrator.pkg"


def test_tier_classification_uses_thresholds(tmp_path: Path) -> None:
    tool = _load_tool()
    src = _make_tree(
        tmp_path,
        {
            "ok.py": tool.REVIEW_THRESHOLD - 1,
            "review.py": tool.REVIEW_THRESHOLD,
            "split.py": tool.SPLIT_THRESHOLD,
        },
    )
    modules = tool.collect_modules(src)

    review = {m.dotted_path for m in tool.review_tier(modules)}
    split = {m.dotted_path for m in tool.split_tier(modules)}

    assert "scpn_phase_orchestrator.ok" not in review | split
    assert review == {"scpn_phase_orchestrator.review"}
    assert split == {"scpn_phase_orchestrator.split"}


def test_load_allowlist_requires_non_empty_reason(tmp_path: Path) -> None:
    tool = _load_tool()
    path = tmp_path / "allow.json"
    path.write_text(
        json.dumps({"reviewed_large_modules": [{"module": "a.b", "reason": "  "}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-empty reason"):
        tool.load_allowlist(path)


def test_load_allowlist_returns_mapping(tmp_path: Path) -> None:
    tool = _load_tool()
    path = tmp_path / "allow.json"
    path.write_text(
        json.dumps(
            {"reviewed_large_modules": [{"module": "a.b", "reason": "cohesive solver"}]}
        ),
        encoding="utf-8",
    )

    assert tool.load_allowlist(path) == {"a.b": "cohesive solver"}


def test_load_allowlist_missing_file_is_empty(tmp_path: Path) -> None:
    tool = _load_tool()
    assert tool.load_allowlist(tmp_path / "absent.json") == {}


def test_check_mode_fails_on_unallowlisted_split_module(tmp_path: Path) -> None:
    tool = _load_tool()
    _make_tree(tmp_path, {"giant.py": tool.SPLIT_THRESHOLD + 50})
    allow = tmp_path / "allow.json"
    allow.write_text(json.dumps({"reviewed_large_modules": []}), encoding="utf-8")

    rc = tool.main(
        [
            "--src-root",
            str(tmp_path / "src" / "scpn_phase_orchestrator"),
            "--allowlist",
            str(allow),
            "--check",
        ]
    )
    assert rc == 1


def test_check_mode_passes_when_split_module_allowlisted(tmp_path: Path) -> None:
    tool = _load_tool()
    _make_tree(tmp_path, {"giant.py": tool.SPLIT_THRESHOLD + 50})
    allow = tmp_path / "allow.json"
    allow.write_text(
        json.dumps(
            {
                "reviewed_large_modules": [
                    {
                        "module": "scpn_phase_orchestrator.giant",
                        "reason": "single cohesive solver",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rc = tool.main(
        [
            "--src-root",
            str(tmp_path / "src" / "scpn_phase_orchestrator"),
            "--allowlist",
            str(allow),
            "--check",
        ]
    )
    assert rc == 0


def test_report_mode_exits_zero_as_warning(tmp_path: Path) -> None:
    tool = _load_tool()
    _make_tree(tmp_path, {"giant.py": tool.SPLIT_THRESHOLD + 50})

    rc = tool.main(["--src-root", str(tmp_path / "src" / "scpn_phase_orchestrator")])
    assert rc == 0


def test_live_repository_has_no_unallowlisted_split_modules() -> None:
    tool = _load_tool()
    modules = tool.collect_modules(_repo_root() / "src" / "scpn_phase_orchestrator")
    allowlist = tool.load_allowlist(
        _repo_root() / "tools" / "large_module_allowlist.json"
    )
    offenders = tool.unreviewed_split_modules(modules, allowlist)
    assert offenders == [], [m.relative_path for m in offenders]
