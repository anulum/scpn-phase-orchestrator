# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Contracts for the test typing ratchet

"""``tools/tests_typing_ratchet.py`` counts, compares and only lowers ceilings.

The end-to-end cases run the real ``mypy`` on a small temporary checkout,
so new debt, stale ceilings and the refusal to raise a ceiling are observed
from actual type-checker output.
"""

from __future__ import annotations

import json
import runpy
import sys
from collections import Counter
from pathlib import Path

import pytest

from tools import tests_typing_ratchet as ratchet

_UNTYPED = "def helper(x):\n    return x\n"
_TYPED = "def helper(x: int) -> int:\n    return x\n"
_IGNORED = 'value: int = "text"  # type: ignore[assignment]\n'


def test_counts_only_errors_in_test_files() -> None:
    """Errors are counted per ``tests/`` file; notes and other modules are not."""
    output = "\n".join(
        [
            "tests/test_a.py:3: error: Missing return type  [no-untyped-def]",
            "tests/test_a.py:9: error: Bad argument  [arg-type]",
            "tests/test_a.py:9: note: See the docs",
            "tests\\sub\\test_b.py:1: error: Untyped call  [no-untyped-call]",
            "benchmarks/bench.py:4: error: Missing type arguments  [type-arg]",
            "Success: no issues found",
        ]
    )
    assert ratchet.count_errors(output) == Counter(
        {"tests/test_a.py": 2, "tests/sub/test_b.py": 1}
    )


def test_compare_flags_new_debt_raised_counts_and_stale_ceilings() -> None:
    """New errors fail, lower counts and deleted files make ceilings stale."""
    report = ratchet.compare(
        current={"tests/a.py": 3, "tests/new.py": 1, "tests/c.py": 1},
        ceilings={"tests/a.py": 2, "tests/c.py": 4, "tests/gone.py": 5},
        existing={"tests/a.py", "tests/new.py", "tests/c.py"},
    )
    assert report.increased == {"tests/a.py": (3, 2), "tests/new.py": (1, 0)}
    assert report.stale == {"tests/c.py": (1, 4), "tests/gone.py": (0, 5)}
    assert not report.passed


def test_compare_passes_at_the_ceilings() -> None:
    """Counts equal to their ceilings and clean files pass."""
    report = ratchet.compare(
        current={"tests/a.py": 2},
        ceilings={"tests/a.py": 2},
        existing={"tests/a.py", "tests/clean.py"},
    )
    assert report.passed


def test_lowered_baseline_drops_clean_files_and_refuses_increases() -> None:
    """Rewriting only lowers ceilings; a first baseline takes the counts."""
    assert ratchet.lowered_baseline({"tests/b.py": 2, "tests/a.py": 0}, None) == {
        "tests/b.py": 2
    }
    assert ratchet.lowered_baseline({"tests/a.py": 1}, {"tests/a.py": 3}) == {
        "tests/a.py": 1
    }
    with pytest.raises(ValueError, match=r"refusing to raise ceilings: tests/a.py"):
        ratchet.lowered_baseline({"tests/a.py": 4}, {"tests/a.py": 3})


def test_baseline_round_trips_and_rejects_malformed_files(tmp_path: Path) -> None:
    """The baseline file round-trips; non-integer or negative ceilings are refused."""
    path = tmp_path / "baseline.json"
    assert ratchet.load_baseline(path) is None
    ratchet.write_baseline(path, {"tests/a.py": 2, "tests/b.py": 1})
    assert ratchet.load_baseline(path) == {"tests/a.py": 2, "tests/b.py": 1}
    assert json.loads(path.read_text(encoding="utf-8"))["total"] == 3
    for files in ({"tests/a.py": -1}, {"tests/a.py": True}, ["tests/a.py"]):
        path.write_text(json.dumps({"files": files}), encoding="utf-8")
        with pytest.raises(ValueError, match="non-negative integers"):
            ratchet.load_baseline(path)


def test_counts_type_ignore_comments_but_not_strings(tmp_path: Path) -> None:
    """Only comment tokens count; ignore text inside strings does not."""
    tests = tmp_path / "tests"
    (tests / "sub").mkdir(parents=True)
    (tests / "test_a.py").write_text(
        "x: int = 1  # type: ignore[assignment]\n"
        "y = 2  #type:ignore\n"
        "z = 3  # noqa: E501  # type: ignore\n"
        'text = "# type: ignore inside a string"\n'
        "w = 4  # type: ignored-word is not an ignore? no: ignore\n",
        encoding="utf-8",
    )
    (tests / "sub" / "test_b.py").write_text(
        "v = 1  # type: ignore\n", encoding="utf-8"
    )
    (tests / "test_clean.py").write_text("u = 1  # plain comment\n", encoding="utf-8")

    assert ratchet.count_ignores(tmp_path) == Counter(
        {"tests/test_a.py": 3, "tests/sub/test_b.py": 1}
    )


def test_count_ignores_rejects_untokenizable_files(tmp_path: Path) -> None:
    """A test file that cannot be tokenized is an error, not a zero count."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_bad.py").write_text('x = """open\n', encoding="utf-8")
    with pytest.raises(ValueError, match="cannot tokenize tests/test_bad.py"):
        ratchet.count_ignores(tmp_path)


def test_ignore_baseline_round_trips_and_rejects_malformed_sections(
    tmp_path: Path,
) -> None:
    """The ignores section round-trips; an absent section is unrecorded."""
    path = tmp_path / "baseline.json"
    assert ratchet.load_ignore_baseline(path) is None
    ratchet.write_baseline(path, {"tests/a.py": 1}, {"tests/a.py": 2, "tests/b.py": 3})
    assert ratchet.load_ignore_baseline(path) == {"tests/a.py": 2, "tests/b.py": 3}
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["ignore_total"] == 5
    assert payload["total"] == 1

    path.write_text(json.dumps({"files": {}}), encoding="utf-8")
    assert ratchet.load_ignore_baseline(path) is None
    for ignores in ({"tests/a.py": -1}, {"tests/a.py": False}, ["tests/a.py"]):
        path.write_text(json.dumps({"files": {}, "ignores": ignores}), encoding="utf-8")
        with pytest.raises(ValueError, match="non-negative integers"):
            ratchet.load_ignore_baseline(path)


def _checkout(tmp_path: Path, source: str) -> Path:
    """Return a minimal checkout with one test module of ``source``."""
    root = tmp_path / "repo"
    (root / "tests").mkdir(parents=True)
    (root / "tools").mkdir()
    (root / "tests" / "test_a.py").write_text(source, encoding="utf-8")
    return root


def test_ratchet_holds_new_debt_and_lowers_on_real_mypy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Real mypy output drives check, refusal, staleness and lowering."""
    monkeypatch.setenv("MYPY_CACHE_DIR", str(tmp_path / "cache"))
    root = _checkout(tmp_path, _UNTYPED)
    args = ["--root", str(root)]

    assert ratchet.main(args) == 1
    assert "new strict-typing errors in tests/test_a.py" in capsys.readouterr().err

    assert ratchet.main([*args, "--write"]) == 0
    baseline = root / "tools" / "tests_typing_baseline.json"
    ceiling = ratchet.load_baseline(baseline)
    assert ceiling is not None and ceiling["tests/test_a.py"] > 0
    assert ratchet.main(args) == 0
    assert "none new" in capsys.readouterr().out

    (root / "tests" / "test_a.py").write_text(_UNTYPED * 2, encoding="utf-8")
    assert ratchet.main(args) == 1
    assert ratchet.main([*args, "--write"]) == 1
    assert "refusing to raise ceilings" in capsys.readouterr().err

    (root / "tests" / "test_a.py").write_text(_TYPED, encoding="utf-8")
    assert ratchet.main(args) == 1
    assert "stale ceilings" in capsys.readouterr().err
    assert ratchet.main([*args, "--write"]) == 0
    assert ratchet.load_baseline(baseline) == {}
    assert ratchet.main(args) == 0


def test_ratchet_holds_type_ignore_ceilings_on_real_mypy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An ignore that silences an error is held like an error ceiling."""
    monkeypatch.setenv("MYPY_CACHE_DIR", str(tmp_path / "cache"))
    root = _checkout(tmp_path, _TYPED + _IGNORED)
    args = ["--root", str(root)]
    baseline = root / "tools" / "tests_typing_baseline.json"

    assert ratchet.main(args) == 1
    assert "new type: ignore comments in tests/test_a.py (1/0)" in (
        capsys.readouterr().err
    )

    assert ratchet.main([*args, "--write"]) == 0
    assert "1 type: ignore comments" in capsys.readouterr().out
    assert ratchet.load_baseline(baseline) == {}
    assert ratchet.load_ignore_baseline(baseline) == {"tests/test_a.py": 1}
    assert ratchet.main(args) == 0
    assert "1 type: ignore comments, none new" in capsys.readouterr().out

    (root / "tests" / "test_a.py").write_text(
        _TYPED + _IGNORED + _IGNORED.replace("value", "other"), encoding="utf-8"
    )
    assert ratchet.main(args) == 1
    assert "new type: ignore comments in tests/test_a.py (2/1)" in (
        capsys.readouterr().err
    )
    assert ratchet.main([*args, "--write"]) == 1
    assert "type: ignore refusing to raise ceilings" in capsys.readouterr().err

    (root / "tests" / "test_a.py").write_text(_TYPED, encoding="utf-8")
    assert ratchet.main(args) == 1
    assert "stale type: ignore ceilings" in capsys.readouterr().err
    assert ratchet.main([*args, "--write"]) == 0
    assert ratchet.load_ignore_baseline(baseline) == {}
    assert ratchet.main(args) == 0


def test_script_entry_point_passes_a_clean_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run as a script, the tool exits 0 when every test file is clean."""
    monkeypatch.setenv("MYPY_CACHE_DIR", str(tmp_path / "cache"))
    root = _checkout(tmp_path, _TYPED)
    monkeypatch.setattr(sys, "argv", ["tests_typing_ratchet.py", "--root", str(root)])
    with pytest.raises(SystemExit) as info:
        runpy.run_path(ratchet.__file__, run_name="__main__")
    assert info.value.code == 0


def test_mypy_that_cannot_check_is_an_error(tmp_path: Path) -> None:
    """A checkout without ``tests/`` makes mypy fail, which is not a pass."""
    with pytest.raises(RuntimeError, match="mypy failed with status 2"):
        ratchet.run_mypy(tmp_path)


def test_repository_baseline_is_well_formed() -> None:
    """The committed baseline loads and its total matches its entries."""
    baseline = ratchet.load_baseline(ratchet.BASELINE)
    assert baseline is not None
    payload = json.loads(ratchet.BASELINE.read_text(encoding="utf-8"))
    assert payload["total"] == sum(baseline.values())
    assert all(path.startswith("tests/") for path in baseline)
    ignores = ratchet.load_ignore_baseline(ratchet.BASELINE)
    assert ignores is not None
    assert payload["ignore_total"] == sum(ignores.values())
    assert all(path.startswith("tests/") for path in ignores)
