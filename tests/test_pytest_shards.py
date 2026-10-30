# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Duration-weighted pytest scheduling tests

"""Shard selection is complete, balanced and fed by real pytest timings."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from tools import pytest_shards
from tools.pytest_shards import discover, main, partition, read_timings, record, verify

pytest_plugins = ["pytester"]


def _tree(root: Path, names: list[str]) -> None:
    """Create empty test modules and one non-test helper under ``root``."""
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    (root / "tests" / "helpers.py").write_text("", encoding="utf-8")


def _manifest(path: Path, durations: object, schema: object = 1) -> Path:
    """Write a timing manifest with the given content."""
    path.write_text(json.dumps({"schema": schema, "durations": durations}), "utf-8")
    return path


_NAMES = [
    "tests/test_a.py",
    "tests/test_b.py",
    "tests/deep/test_c.py",
    "tests/deep/d_test.py",
    "tests/skip/test_e.py",
    "tests/test_nn_physics_validation_p1.py",
]


def test_discover_follows_pytest_names_and_ignore_semantics(tmp_path: Path) -> None:
    """Select pytest's module names and apply path, directory and glob exclusions."""
    _tree(tmp_path, _NAMES)
    ignored = ["tests/test_b.py", "tests/skip/", "tests/test_nn_physics_validation*.py"]
    assert discover(tmp_path, "tests", ignored) == [
        "tests/deep/d_test.py",
        "tests/deep/test_c.py",
        "tests/test_a.py",
    ]
    assert len(discover(tmp_path, "tests", [])) == len(_NAMES)


def test_discover_refuses_an_empty_selection(tmp_path: Path) -> None:
    """Refuse a run that would select nothing."""
    _tree(tmp_path, ["tests/test_a.py"])
    with pytest.raises(ValueError, match="no test modules"):
        discover(tmp_path, "tests", ["tests"])


def test_read_timings_accepts_cold_and_valid_manifests(tmp_path: Path) -> None:
    """Treat a missing manifest as a cold run and return valid seconds as floats."""
    assert read_timings(None) == {}
    assert read_timings(tmp_path / "absent.json") == {}
    path = _manifest(tmp_path / "t.json", {"tests/test_a.py": 2})
    assert read_timings(path) == {"tests/test_a.py": 2.0}


@pytest.mark.parametrize(
    ("durations", "schema", "message"),
    [
        ({}, 2, "unsupported"),
        ([], 1, "must be an object"),
        ({"tests/test_a.py": 0}, 1, "positive seconds"),
        ({"tests/test_a.py": -1.0}, 1, "positive seconds"),
        ({"tests/test_a.py": True}, 1, "positive seconds"),
        ({"tests/test_a.py": "1"}, 1, "positive seconds"),
        ({"": 1.0}, 1, "positive seconds"),
    ],
)
def test_read_timings_refuses_malformed_manifests(
    tmp_path: Path, durations: object, schema: object, message: str
) -> None:
    """Refuse other schemas, non-objects and non-positive or non-numeric seconds."""
    with pytest.raises(ValueError, match=message):
        read_timings(_manifest(tmp_path / "t.json", durations, schema))


def test_read_timings_refuses_nonfinite_seconds(tmp_path: Path) -> None:
    """Refuse NaN and infinity, which JSON carries as bare constants."""
    path = tmp_path / "t.json"
    path.write_text('{"schema": 1, "durations": {"tests/test_a.py": NaN}}', "utf-8")
    with pytest.raises(ValueError, match="positive seconds"):
        read_timings(path)
    path.write_text("[]", "utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        read_timings(path)


def test_cold_partition_deals_the_sorted_list_round_robin() -> None:
    """Without history every module still runs exactly once."""
    shards = partition(["c", "a", "d", "b", "e"], 2, {})
    assert shards == [["a", "c", "e"], ["b", "d"]]


def test_measured_partition_balances_longest_first_with_median_for_new() -> None:
    """Put the longest modules on the least loaded shard; new ones get the median."""
    timings = {"a": 10.0, "b": 6.0, "c": 4.0, "d": 1.0, "gone": 2.0}
    shards = partition(["a", "b", "c", "d", "new"], 2, timings)
    assert shards == [["a", "new"], ["b", "c", "d"]]
    assert sorted(name for shard in shards for name in shard) == [
        "a",
        "b",
        "c",
        "d",
        "new",
    ]


@pytest.mark.parametrize(
    ("modules", "count"),
    [(["a"], 0), (["a"], 2), (["a", "a"], 1), ([], 1)],
)
def test_partition_refuses_counts_that_do_not_fit(
    modules: list[str], count: int
) -> None:
    """Refuse empty shards and repeated modules."""
    with pytest.raises(ValueError, match="shard count"):
        partition(modules, count, {})


def test_partition_refuses_empty_names() -> None:
    """Refuse an empty module name."""
    with pytest.raises(ValueError, match="nonempty"):
        partition(["a", ""], 1, {})


def _receipt(path: Path, shard: object, modules: object, count: object = 2) -> Path:
    """Write one selection receipt."""
    payload = {"schema": 1, "shard": shard, "count": count, "modules": modules}
    path.write_text(json.dumps(payload), "utf-8")
    return path


def test_verify_accepts_exactly_once_selection(tmp_path: Path) -> None:
    """Accept shards whose union is the discovery list without overlap."""
    one = _receipt(tmp_path / "1.json", 1, ["a", "c"])
    two = _receipt(tmp_path / "2.json", 2, ["b"])
    verify(["a", "b", "c"], [one, two], 2)


@pytest.mark.parametrize(
    ("first", "second", "message"),
    [
        ((1, ["a"]), (1, ["b"]), "shard index"),
        ((1, ["a"]), (3, ["b"]), "shard index"),
        ((1, ["a"]), (True, ["b"]), "shard index"),
        ((1, ["a"]), (2, []), "selected no module"),
        ((1, ["a"]), (2, ["a"]), "duplicated"),
        ((1, ["a"]), (2, [7]), "duplicated or invalid"),
        ((1, ["a"]), (2, ["b", "x"]), "invented"),
        ((1, ["a"]), (2, ["c"]), "omitted"),
    ],
)
def test_verify_refuses_incomplete_or_inconsistent_shards(
    tmp_path: Path,
    first: tuple[object, object],
    second: tuple[object, object],
    message: str,
) -> None:
    """Refuse repeated, foreign, empty, overlapping, omitting or inventing shards."""
    one = _receipt(tmp_path / "1.json", *first)
    two = _receipt(tmp_path / "2.json", *second)
    with pytest.raises(ValueError, match=message):
        verify(["a", "b"], [one, two], 2)


def test_verify_refuses_missing_shards_and_foreign_receipts(tmp_path: Path) -> None:
    """Refuse a run with a missing shard or a receipt of another run shape."""
    one = _receipt(tmp_path / "1.json", 1, ["a", "b"])
    with pytest.raises(ValueError, match=r"shards \[2\] missing"):
        verify(["a", "b"], [one], 2)
    foreign = _receipt(tmp_path / "f.json", 1, ["a", "b"], count=3)
    with pytest.raises(ValueError, match="foreign or malformed"):
        verify(["a", "b"], [foreign], 2)


def test_record_keeps_live_history_and_refuses_double_timing(tmp_path: Path) -> None:
    """Merge shard reports over live history and drop removed modules."""
    previous = _manifest(tmp_path / "p.json", {"a": 5.0, "b": 3.0, "gone": 1.0})
    one = _manifest(tmp_path / "1.json", {"a": 2.0, "outside": 4.0})
    two = _manifest(tmp_path / "2.json", {"c": 1.5})
    manifest = json.loads(record([one, two], ["a", "b", "c"], previous))
    assert manifest == {"schema": 1, "durations": {"a": 2.0, "b": 3.0, "c": 1.5}}
    with pytest.raises(ValueError, match="timed by two shards"):
        record([one, one], ["a"], None)
    with pytest.raises(ValueError, match="no live module"):
        record([two], ["a"], None)


_REPOSITORY = Path(__file__).resolve().parents[1]


def test_cli_selects_verifies_and_records_a_real_run(
    pytester: pytest.Pytester,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run select, real pytest with the timing plugin, verify and record end to end."""
    monkeypatch.setenv("PYTHONPATH", str(_REPOSITORY))
    pytester.makepyfile(
        test_alpha="def test_one():\n    assert True\n",
        test_beta="def test_two():\n    assert True\n",
    )
    pytester.mkpydir("tests")
    for name in ("test_alpha.py", "test_beta.py"):
        (pytester.path / name).rename(pytester.path / "tests" / name)
    root = pytester.path
    base = ["--root", str(root)]
    receipts = []
    reports = []
    for index in (1, 2):
        receipt = root / f"receipt-{index}.json"
        capsys.readouterr()
        assert (
            main([*base, "select", "--shard", f"{index}/2", "--receipt", str(receipt)])
            == 0
        )
        selected = capsys.readouterr().out.split()
        report = root / f"timings-{index}.json"
        result = pytester.runpytest_subprocess(
            *selected,
            "-p",
            "tools.pytest_timing_plugin",
            "--module-timings",
            str(report),
        )
        result.assert_outcomes(passed=1)
        receipts.append(str(receipt))
        reports.append(str(report))
    assert main([*base, "verify", "--count", "2", *receipts]) == 0
    assert "2 modules selected exactly once across 2 shards" in capsys.readouterr().out
    output = root / "next.json"
    assert main([*base, "record", "--output", str(output), *reports]) == 0
    assert set(read_timings(output)) == {"tests/test_alpha.py", "tests/test_beta.py"}
    assert (
        main(
            [
                *base,
                "select",
                "--shard",
                "1/1",
                "--timings",
                str(output),
                "--receipt",
                str(root / "r.json"),
            ]
        )
        == 0
    )


@pytest.mark.parametrize("shard", ["0/2", "3/2"])
def test_cli_refuses_out_of_range_shards(
    tmp_path: Path, shard: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Return two and explain the refusal for an impossible shard."""
    _tree(tmp_path, ["tests/test_a.py", "tests/test_b.py"])
    argv = ["--root", str(tmp_path), "select", "--shard", shard, "--receipt", "r.json"]
    assert main(argv) == 2
    assert "shard must be K/N" in capsys.readouterr().err


def test_timing_plugin_records_nothing_without_its_option(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Leave no manifest when the option is absent."""
    monkeypatch.setenv("PYTHONPATH", str(_REPOSITORY))
    pytester.makepyfile(test_gamma="def test_three():\n    assert True\n")
    result = pytester.runpytest_subprocess("-p", "tools.pytest_timing_plugin")
    result.assert_outcomes(passed=1)
    assert not list(pytester.path.glob("*.json"))


def test_module_exports_are_the_documented_commands() -> None:
    """Keep the public surface to the scheduler commands."""
    assert pytest_shards.__all__ == [
        "discover",
        "main",
        "partition",
        "read_timings",
        "record",
        "verify",
    ]


def test_module_runs_as_a_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exit with the CLI status when executed as ``python -m tools.pytest_shards``."""
    _tree(tmp_path, ["tests/test_a.py"])
    receipt = tmp_path / "r.json"
    argv = ["pytest_shards", "--root", str(tmp_path), "select", "--shard", "1/1"]
    monkeypatch.setattr("sys.argv", [*argv, "--receipt", str(receipt)])
    with pytest.raises(SystemExit) as exit_status:
        runpy.run_path(
            str(_REPOSITORY / "tools" / "pytest_shards.py"), run_name="__main__"
        )
    assert exit_status.value.code == 0
    assert capsys.readouterr().out == "tests/test_a.py\n"
