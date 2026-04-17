# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for check_test_module_linkage.py

"""Unit tests for ``tools/check_test_module_linkage.py``.

The guard enforces that every source module has at least one direct
mention in the test corpus (via dotted import path or matching
``test_<stem>`` reference). Scratch repos under tmp_path exercise the
four behavioural branches:

* Linked module (by import path or by test_<stem>) is not flagged.
* Fully unlinked module is flagged unless allowlisted.
* Stale allowlist entry (module now has tests) fails unless the
  ``--allow-stale-allowlist`` flag is passed.
* Allowlist JSON schema validation — missing key, wrong shape,
  missing file.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
SCRIPT = TOOLS_DIR / "check_test_module_linkage.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_module_linkage_test_mod", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load()


def _scratch_package(tmp_path: Path, modules: list[str]) -> Path:
    """Build src/scpn_phase_orchestrator/<modules> tree; return src root."""
    src_root = tmp_path / "src" / "scpn_phase_orchestrator"
    src_root.mkdir(parents=True)
    (src_root / "__init__.py").write_text("", encoding="utf-8")
    for rel in modules:
        path = src_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.parent != src_root:
            (path.parent / "__init__.py").touch()
        path.write_text("x = 1\n", encoding="utf-8")
    return src_root


def _scratch_tests(tmp_path: Path, bodies: dict[str, str]) -> Path:
    """Build tests/ directory with file → body mapping; return test root."""
    test_root = tmp_path / "tests"
    test_root.mkdir()
    for name, body in bodies.items():
        (test_root / name).write_text(body, encoding="utf-8")
    return test_root


# ---------------------------------------------------------------------
# collect_source_modules
# ---------------------------------------------------------------------


def test_collect_source_modules_excludes_init(tmp_path: Path) -> None:
    src_root = _scratch_package(tmp_path, ["upde/engine.py", "monitor/pid.py"])
    modules = mod.collect_source_modules(src_root)
    names = sorted(p.name for p in modules)
    assert names == ["engine.py", "pid.py"]
    # __init__.py files are excluded
    assert not any(p.name == "__init__.py" for p in modules)


def test_module_import_path(tmp_path: Path) -> None:
    src_root = _scratch_package(tmp_path, ["upde/engine.py"])
    (module_path,) = mod.collect_source_modules(src_root)
    assert (
        mod._module_import_path(src_root, module_path)
        == "scpn_phase_orchestrator.upde.engine"
    )


# ---------------------------------------------------------------------
# collect_unlinked_modules
# ---------------------------------------------------------------------


def test_linked_by_import_path(tmp_path: Path) -> None:
    src_root = _scratch_package(tmp_path, ["upde/engine.py"])
    test_root = _scratch_tests(
        tmp_path,
        {
            "test_something.py": (
                "from scpn_phase_orchestrator.upde.engine import X\n"
            ),
        },
    )
    unlinked = mod.collect_unlinked_modules(
        source_root=src_root, test_root=test_root
    )
    assert unlinked == []


def test_linked_by_test_stem(tmp_path: Path) -> None:
    src_root = _scratch_package(tmp_path, ["upde/engine.py"])
    test_root = _scratch_tests(
        tmp_path,
        {"test_engine.py": "# references test_engine implicitly\n"},
    )
    unlinked = mod.collect_unlinked_modules(
        source_root=src_root, test_root=test_root
    )
    assert unlinked == []


def test_unlinked_module_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    src_root = _scratch_package(
        tmp_path, ["upde/engine.py", "monitor/orphan.py"]
    )
    test_root = _scratch_tests(
        tmp_path,
        {
            "test_engine.py": (
                "from scpn_phase_orchestrator.upde.engine import X\n"
            ),
        },
    )
    unlinked = mod.collect_unlinked_modules(
        source_root=src_root, test_root=test_root
    )
    assert len(unlinked) == 1
    assert unlinked[0].endswith("monitor/orphan.py")


# ---------------------------------------------------------------------
# load_allowlist
# ---------------------------------------------------------------------


def test_load_allowlist_happy_path(tmp_path: Path) -> None:
    f = tmp_path / "allow.json"
    f.write_text(
        json.dumps(
            {
                "allowlisted_modules": [
                    {"path": "src/scpn_phase_orchestrator/a.py", "reason": "x"},
                    {"path": "src/scpn_phase_orchestrator/b.py"},
                ]
            }
        ),
        encoding="utf-8",
    )
    entries = mod.load_allowlist(f)
    assert entries == {
        "src/scpn_phase_orchestrator/a.py",
        "src/scpn_phase_orchestrator/b.py",
    }


def test_load_allowlist_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        mod.load_allowlist(tmp_path / "nope.json")


def test_load_allowlist_non_dict_root(tmp_path: Path) -> None:
    f = tmp_path / "allow.json"
    f.write_text(json.dumps([1, 2]), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        mod.load_allowlist(f)


def test_load_allowlist_missing_key(tmp_path: Path) -> None:
    f = tmp_path / "allow.json"
    f.write_text(json.dumps({"other": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="allowlisted_modules must be a list"):
        mod.load_allowlist(f)


def test_load_allowlist_entry_not_object(tmp_path: Path) -> None:
    f = tmp_path / "allow.json"
    f.write_text(
        json.dumps({"allowlisted_modules": ["plain string"]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"allowlisted_modules\[0\]"):
        mod.load_allowlist(f)


def test_load_allowlist_empty_path_rejected(tmp_path: Path) -> None:
    f = tmp_path / "allow.json"
    f.write_text(
        json.dumps({"allowlisted_modules": [{"path": ""}]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="non-empty string"):
        mod.load_allowlist(f)


# ---------------------------------------------------------------------
# main integration
# ---------------------------------------------------------------------


def _write_allow(path: Path, entries: list[str]) -> None:
    path.write_text(
        json.dumps(
            {"allowlisted_modules": [{"path": e, "reason": "x"} for e in entries]}
        ),
        encoding="utf-8",
    )


def test_main_pass_when_everything_linked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src_root = _scratch_package(tmp_path, ["upde/engine.py"])
    test_root = _scratch_tests(
        tmp_path,
        {"test_engine.py": "from scpn_phase_orchestrator.upde.engine import X\n"},
    )
    allow = tmp_path / "allow.json"
    _write_allow(allow, [])
    rc = mod.main(
        [
            "--source-root",
            str(src_root),
            "--test-root",
            str(test_root),
            "--allowlist",
            str(allow),
        ]
    )
    assert rc == 0
    assert "Untested-module guard passed" in capsys.readouterr().out


def test_main_fails_on_unexpected_module(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    src_root = _scratch_package(tmp_path, ["upde/orphan.py"])
    test_root = _scratch_tests(tmp_path, {})
    allow = tmp_path / "allow.json"
    _write_allow(allow, [])
    rc = mod.main(
        [
            "--source-root",
            str(src_root),
            "--test-root",
            str(test_root),
            "--allowlist",
            str(allow),
        ]
    )
    assert rc == 1
    assert "new modules without direct test linkage" in capsys.readouterr().out


def test_main_fails_on_stale_allowlist(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Allowlisted module now has test linkage → allowlist is stale."""
    src_root = _scratch_package(tmp_path, ["upde/engine.py"])
    test_root = _scratch_tests(
        tmp_path,
        {"test_engine.py": "from scpn_phase_orchestrator.upde.engine import X\n"},
    )
    allow = tmp_path / "allow.json"
    # Allowlist entry uses REPO_ROOT-relative posix paths in the script;
    # compute what collect_unlinked_modules would produce — but since
    # engine.py is now linked, any allowlist entry is "stale".
    _write_allow(allow, ["src/scpn_phase_orchestrator/upde/engine.py"])
    rc = mod.main(
        [
            "--source-root",
            str(src_root),
            "--test-root",
            str(test_root),
            "--allowlist",
            str(allow),
        ]
    )
    assert rc == 1
    assert "stale allowlist entries" in capsys.readouterr().out


def test_main_allow_stale_flag_bypasses_stale_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src_root = _scratch_package(tmp_path, ["upde/engine.py"])
    test_root = _scratch_tests(
        tmp_path,
        {"test_engine.py": "from scpn_phase_orchestrator.upde.engine import X\n"},
    )
    allow = tmp_path / "allow.json"
    _write_allow(allow, ["src/scpn_phase_orchestrator/upde/engine.py"])
    rc = mod.main(
        [
            "--source-root",
            str(src_root),
            "--test-root",
            str(test_root),
            "--allowlist",
            str(allow),
            "--allow-stale-allowlist",
        ]
    )
    assert rc == 0
    assert "Untested-module guard passed" in capsys.readouterr().out
