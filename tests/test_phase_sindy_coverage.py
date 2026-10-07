# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Phase-SINDy coverage admission contracts

"""Exercise the admission CLI using captured real Python and LLVM exports.

Fixtures retain genuine branch/region vectors measured on the hash-pinned
source. Negative cases damage a copy of those records; they never invent
successful numerical execution or callback hits.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from tools.phase_sindy_coverage import verify_reports

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/phase_sindy_coverage"
BENCHMARK = "benchmarks/phase_sindy_benchmark.py"


def _read(path: Path) -> dict[str, object]:
    """Decode the recorded report for one controlled corruption case."""
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def _run(
    python: Path, native: Path, *, root: Path = ROOT
) -> subprocess.CompletedProcess[str]:
    """Invoke the actual public admission command in an isolated child."""
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/phase_sindy_coverage.py"),
            "--python-report",
            str(python),
            "--native-report",
            str(native),
            "--root",
            str(root),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_real_exports_are_admitted_with_disclosed_convergence_debt() -> None:
    """The actual captured pair retains eight untraced statements and native debt."""
    result = _run(FIXTURES / "python.json", FIXTURES / "native.json")
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["native_full_coverage"] is False
    assert "100000" in report["native_convergence_debt"]
    assert len(report["native_missing_regions"]) == 6
    exclusions = report["python_measurement_exclusions"]
    assert exclusions[BENCHMARK]["lines"] == [215, 216, 217, 218]
    assert exclusions["native-tests/helpers/sindy_cli_probe.py"]["lines"] == [
        57,
        58,
        59,
        64,
    ]
    assert exclusions["src/scpn_phase_orchestrator/autotune/sindy.py"]["lines"] == []


def test_public_api_and_cli_return_the_same_real_admission() -> None:
    """Library callers receive the same explicit debt and exclusions as operators."""
    direct = verify_reports(FIXTURES / "python.json", FIXTURES / "native.json")
    result = _run(FIXTURES / "python.json", FIXTURES / "native.json")
    assert result.returncode == 0
    assert json.loads(json.dumps(direct)) == json.loads(result.stdout)


def test_native_path_separator_does_not_change_admission(tmp_path: Path) -> None:
    """Equivalent export path spellings preserve the same recorded branch vectors."""
    document = _read(FIXTURES / "native.json")
    data = cast(list[dict[str, object]], document["data"])
    files = cast(list[dict[str, object]], data[0]["files"])
    filename = cast(str, files[0]["filename"])
    files[0]["filename"] = filename.replace("/", "\\")
    path = tmp_path / "native.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    result = _run(FIXTURES / "python.json", path)
    assert result.returncode == 0


@pytest.mark.parametrize(
    "fault",
    [
        "top-array",
        "meta-array",
        "line-only",
        "files-array",
        "missing-member",
        "line-string",
        "line-boolean",
        "statement-omitted",
        "statement-overlap",
        "statement-count",
        "branch-count",
        "hidden-statement",
        "unreviewed-line",
        "arc-size",
        "arc-duplicate",
        "branch-omitted",
        "branch-overlap",
        "unreviewed-arc",
    ],
)
def test_python_report_faults_refuse_without_interpreter_text(
    tmp_path: Path, fault: str
) -> None:
    """Malformed or incomplete copied measurements cannot clear admission."""
    document = _read(FIXTURES / "python.json")
    files = cast(dict[str, dict[str, object]], document["files"])
    member = files[BENCHMARK]
    summary = cast(dict[str, object], member["summary"])
    executed = cast(list[int], member["executed_lines"])
    missing = cast(list[int], member["missing_lines"])
    executed_arcs = cast(list[list[int]], member["executed_branches"])
    missing_arcs = cast(list[list[int]], member["missing_branches"])
    value: object = document
    if fault == "top-array":
        value = []
    elif fault == "meta-array":
        document["meta"] = []
    elif fault == "line-only":
        cast(dict[str, object], document["meta"])["branch_coverage"] = False
    elif fault == "files-array":
        document["files"] = []
    elif fault == "missing-member":
        del files[BENCHMARK]
    elif fault == "line-string":
        member["executed_lines"] = "executed"
    elif fault == "line-boolean":
        member["executed_lines"] = [True]
    elif fault == "statement-omitted":
        executed.pop()
    elif fault == "statement-overlap":
        missing.append(executed[0])
    elif fault == "statement-count":
        summary["num_statements"] = 0
    elif fault == "branch-count":
        summary["num_branches"] = 0
    elif fault == "hidden-statement":
        member["excluded_lines"] = [executed[0]]
    elif fault == "unreviewed-line":
        missing.append(executed.pop())
    elif fault == "arc-size":
        member["missing_branches"] = [[215]]
    elif fault == "arc-duplicate":
        missing_arcs.append(missing_arcs[0])
    elif fault == "branch-omitted":
        executed_arcs.pop()
    elif fault == "branch-overlap":
        missing_arcs.append(executed_arcs[0])
    else:
        missing_arcs.append(executed_arcs.pop())
    path = tmp_path / "python.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    result = _run(path, FIXTURES / "native.json")
    assert result.returncode == 1
    assert result.stdout.startswith("Phase-SINDy coverage refused:")
    assert "Traceback" not in result.stdout + result.stderr


@pytest.mark.parametrize(
    "fault",
    [
        "top-array",
        "data-string",
        "missing-member",
        "duplicate-member",
        "branches-omitted",
        "branch-size",
        "branch-missing",
        "branch-boolean",
        "segments-omitted",
        "segment-size",
        "segment-flag",
        "unreviewed-region",
    ],
)
def test_native_export_faults_refuse(tmp_path: Path, fault: str) -> None:
    """No damaged LLVM vector or additional uncovered region can clear the gate."""
    document = _read(FIXTURES / "native.json")
    data = cast(list[dict[str, object]], document["data"])
    files = cast(list[dict[str, object]], data[0]["files"])
    member = files[0]
    branches = cast(list[list[int]], member["branches"])
    segments = cast(list[list[object]], member["segments"])
    value: object = document
    if fault == "top-array":
        value = []
    elif fault == "data-string":
        document["data"] = "unmeasured"
    elif fault == "missing-member":
        files.clear()
    elif fault == "duplicate-member":
        files.append(member)
    elif fault == "branches-omitted":
        branches.clear()
    elif fault == "branch-size":
        branches[0].pop()
    elif fault == "branch-missing":
        branches[0][4] = 0
    elif fault == "branch-boolean":
        member["branches"] = [[True] * 9] + branches[1:]
    elif fault == "segments-omitted":
        segments.pop()
    elif fault == "segment-size":
        segments[0].pop()
    elif fault == "segment-flag":
        segments[0][3] = 1
    else:
        segment = next(row for row in segments if row[2] and row[3] and row[4])
        segment[2] = 0
    path = tmp_path / "native.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    result = _run(FIXTURES / "python.json", path)
    assert result.returncode == 1
    assert result.stdout.startswith("Phase-SINDy coverage refused:")
    assert "Traceback" not in result.stdout + result.stderr


@pytest.mark.parametrize("changed", ["python", "native", "lock", "policy-path"])
def test_source_or_dependency_drift_lapses_the_policy(
    tmp_path: Path, changed: str
) -> None:
    """An exact-source disposition cannot transfer to changed code or dependencies."""
    policy = _read(ROOT / "tools/phase_sindy_coverage_policy.json")
    paths = list(cast(dict[str, object], policy["python"])) + [
        "tools/phase_sindy_coverage_policy.json",
        "spo-kernel/crates/spo-engine/src/sindy.rs",
        "spo-kernel/Cargo.lock",
    ]
    for filename in paths:
        target = tmp_path / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / filename, target)
    if changed == "policy-path":
        cast(dict[str, object], policy["native"])["path"] = 5
        (tmp_path / "tools/phase_sindy_coverage_policy.json").write_text(
            json.dumps(policy), encoding="utf-8"
        )
    else:
        filename = (
            BENCHMARK
            if changed == "python"
            else "spo-kernel/crates/spo-engine/src/sindy.rs"
            if changed == "native"
            else "spo-kernel/Cargo.lock"
        )
        with (tmp_path / filename).open("a", encoding="utf-8") as stream:
            stream.write("\nchanged source binding\n")
    result = _run(FIXTURES / "python.json", FIXTURES / "native.json", root=tmp_path)
    assert result.returncode == 1


def test_unreadable_report_is_an_authored_refusal(tmp_path: Path) -> None:
    """A missing input is not accepted as zero uncovered code."""
    result = _run(tmp_path / "missing.json", FIXTURES / "native.json")
    assert result.returncode == 1
    assert result.stderr == ""
