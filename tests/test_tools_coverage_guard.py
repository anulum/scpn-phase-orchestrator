# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for coverage_guard.py

"""Unit tests for ``tools/coverage_guard.py``.

Coverage regression gate for the CI pipeline. Without tests a silent
regression (e.g. misparsed Cobertura attribute, wrong domain split)
would either let uncovered modules through or fail CI on healthy
runs. Cover the five behaviour surfaces:

* ``_validate_percent`` — bounds [0, 100], rejects NaN/Inf.
* ``_domain_for`` — ``src/scpn_phase_orchestrator/<domain>/...`` mapping;
  everything else falls into ``"other"``.
* ``load_coverage`` / ``load_thresholds`` — XML / JSON parsers,
  required-field validation, file-not-found surface.
* ``evaluate`` / ``main`` — global, per-domain, per-file threshold
  branches; exit codes on pass / fail.
* Branch axis — ``condition-coverage`` parsing (global / per-file /
  per-domain aggregation, branch-free files omitted, malformed
  attribute rejection), branch threshold validation, fail-closed
  evaluation when branch floors meet a line-only report, and the
  branch block in ``main`` output.
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_coverage_guard_test_mod", TOOLS_DIR / "coverage_guard.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load()


# ---------------------------------------------------------------------
# _validate_percent
# ---------------------------------------------------------------------


class TestValidatePercent:
    def test_accepts_boundary_zero(self) -> None:
        assert mod._validate_percent(0.0, label="x") == 0.0

    def test_accepts_boundary_hundred(self) -> None:
        assert mod._validate_percent(100.0, label="x") == 100.0

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match=r"must be in \[0, 100\]"):
            mod._validate_percent(-0.1, label="neg")

    def test_rejects_above_hundred(self) -> None:
        with pytest.raises(ValueError, match=r"must be in \[0, 100\]"):
            mod._validate_percent(100.1, label="over")

    def test_rejects_nan(self) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            mod._validate_percent(math.nan, label="nan")

    def test_rejects_inf(self) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            mod._validate_percent(math.inf, label="inf")


# ---------------------------------------------------------------------
# _domain_for
# ---------------------------------------------------------------------


class TestDomainFor:
    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("src/scpn_phase_orchestrator/upde/engine.py", "upde"),
            ("src/scpn_phase_orchestrator/monitor/pid.py", "monitor"),
            ("src/scpn_phase_orchestrator/binding/types.py", "binding"),
            ("src\\scpn_phase_orchestrator\\coupling\\base.py", "coupling"),
        ],
    )
    def test_known_domain(self, filename: str, expected: str) -> None:
        assert mod._domain_for(filename) == expected

    @pytest.mark.parametrize(
        "filename",
        [
            "tests/test_foo.py",
            "src/other_package/x.py",
            "tools/preflight.py",
            "README.md",
        ],
    )
    def test_falls_back_to_other(self, filename: str) -> None:
        assert mod._domain_for(filename) == "other"


# ---------------------------------------------------------------------
# _resolve
# ---------------------------------------------------------------------


def test_resolve_relative_path_uses_repo_root() -> None:
    path = mod._resolve("tools/foo.py")
    assert path.is_absolute()
    assert path.name == "foo.py"


def test_resolve_absolute_path_preserved(tmp_path: Path) -> None:
    target = tmp_path / "abs.json"
    resolved = mod._resolve(str(target))
    assert resolved == target


# ---------------------------------------------------------------------
# load_coverage
# ---------------------------------------------------------------------


def _write_cobertura(
    path: Path,
    *,
    line_rate: float,
    classes: list[tuple[str, list[tuple[int, int]]]],
) -> None:
    """Write a minimal Cobertura XML doc.

    classes is a list of ``(filename, [(line_no, hits), ...])`` tuples.
    """
    total_covered = sum(1 for _, lines in classes for _, h in lines if h > 0)
    total_lines = sum(len(lines) for _, lines in classes)
    parts = [
        f'<coverage line-rate="{line_rate}" '
        f'lines-covered="{total_covered}" lines-valid="{total_lines}">',
        "<packages><package><classes>",
    ]
    for filename, lines in classes:
        class_covered = sum(1 for _, h in lines if h > 0)
        class_total = len(lines) or 1
        class_rate = class_covered / class_total
        parts.append(f'<class filename="{filename}" line-rate="{class_rate}">')
        parts.append("<lines>")
        for line_no, hits in lines:
            parts.append(f'<line number="{line_no}" hits="{hits}"/>')
        parts.append("</lines></class>")
    parts.append("</classes></package></packages></coverage>")
    path.write_text("".join(parts), encoding="utf-8")


def test_load_coverage_parses_line_rate_and_domain(tmp_path: Path) -> None:
    xml = tmp_path / "cov.xml"
    _write_cobertura(
        xml,
        line_rate=0.90,
        classes=[
            (
                "src/scpn_phase_orchestrator/upde/engine.py",
                [(1, 1), (2, 1), (3, 0)],
            ),
            (
                "src/scpn_phase_orchestrator/monitor/pid.py",
                [(1, 1), (2, 1)],
            ),
        ],
    )
    summary = mod.load_coverage(xml)
    assert summary.line_rate_pct == 90.0
    assert summary.lines_covered == 4
    assert summary.lines_valid == 5
    # upde = 2/3 covered ≈ 66.67; monitor = 2/2 = 100
    assert summary.domain_line_rate_pct["upde"] == pytest.approx(2 / 3 * 100)
    assert summary.domain_line_rate_pct["monitor"] == 100.0
    assert summary.file_line_rate_pct[
        "src/scpn_phase_orchestrator/upde/engine.py"
    ] == pytest.approx(2 / 3 * 100)


def test_load_coverage_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        mod.load_coverage(tmp_path / "nope.xml")


def test_load_coverage_skips_empty_filename(tmp_path: Path) -> None:
    xml = tmp_path / "cov.xml"
    xml.write_text(
        '<coverage line-rate="1.0" lines-covered="1" lines-valid="1">'
        "<packages><package><classes>"
        '<class filename="" line-rate="1.0">'
        '<lines><line number="1" hits="1"/></lines></class>'
        "</classes></package></packages></coverage>",
        encoding="utf-8",
    )
    summary = mod.load_coverage(xml)
    assert summary.file_line_rate_pct == {}


# ---------------------------------------------------------------------
# load_thresholds
# ---------------------------------------------------------------------


def test_load_thresholds_happy_path(tmp_path: Path) -> None:
    config = tmp_path / "t.json"
    config.write_text(
        json.dumps(
            {
                "global_min_line_rate": 92.0,
                "domain_min_line_rate": {"upde": 95.0},
                "file_min_line_rate": {"src/foo.py": 100.0},
            }
        ),
        encoding="utf-8",
    )
    data = mod.load_thresholds(config)
    assert data["global_min_line_rate"] == 92.0
    assert data["domain_min_line_rate"] == {"upde": 95.0}


def test_load_thresholds_missing_global_key(tmp_path: Path) -> None:
    config = tmp_path / "t.json"
    config.write_text(json.dumps({"domain_min_line_rate": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="global_min_line_rate"):
        mod.load_thresholds(config)


def test_load_thresholds_non_dict_root(tmp_path: Path) -> None:
    config = tmp_path / "t.json"
    config.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        mod.load_thresholds(config)


def test_load_thresholds_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        mod.load_thresholds(tmp_path / "missing.json")


def test_load_thresholds_validates_domain_values(tmp_path: Path) -> None:
    config = tmp_path / "t.json"
    config.write_text(
        json.dumps(
            {
                "global_min_line_rate": 92.0,
                "domain_min_line_rate": {"upde": 120.0},  # invalid
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"domain_min_line_rate\[upde\]"):
        mod.load_thresholds(config)


# ---------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------


def _make_summary(
    line_rate: float = 95.0,
    domain: dict[str, float] | None = None,
    files: dict[str, float] | None = None,
    branch_rate: float | None = None,
    branch_domain: dict[str, float] | None = None,
    branch_files: dict[str, float] | None = None,
) -> object:
    return mod.CoverageSummary(
        line_rate_pct=line_rate,
        file_line_rate_pct=files or {},
        domain_line_rate_pct=domain or {},
        lines_covered=0,
        lines_valid=0,
        branch_rate_pct=branch_rate,
        file_branch_rate_pct=branch_files or {},
        domain_branch_rate_pct=branch_domain or {},
        branches_covered=0,
        branches_valid=0,
    )


def test_evaluate_passes_when_above_threshold() -> None:
    summary = _make_summary(line_rate=95.0)
    failures = mod.evaluate(summary, {"global_min_line_rate": 92.0})
    assert failures == []


def test_evaluate_flags_global_regression() -> None:
    summary = _make_summary(line_rate=90.0)
    failures = mod.evaluate(summary, {"global_min_line_rate": 92.0})
    assert len(failures) == 1
    assert "Global" in failures[0] and "90.00" in failures[0]


def test_evaluate_flags_domain_regression() -> None:
    summary = _make_summary(line_rate=95.0, domain={"upde": 80.0})
    failures = mod.evaluate(
        summary,
        {"global_min_line_rate": 92.0, "domain_min_line_rate": {"upde": 90.0}},
    )
    assert len(failures) == 1
    assert "upde" in failures[0]


def test_evaluate_ignores_missing_domain_in_summary() -> None:
    summary = _make_summary(line_rate=95.0, domain={"upde": 95.0})
    failures = mod.evaluate(
        summary,
        {
            "global_min_line_rate": 92.0,
            "domain_min_line_rate": {"missing_domain": 99.0},
        },
    )
    assert failures == []


def test_evaluate_flags_file_regression() -> None:
    summary = _make_summary(
        line_rate=95.0, files={"src/scpn_phase_orchestrator/upde/engine.py": 70.0}
    )
    failures = mod.evaluate(
        summary,
        {
            "global_min_line_rate": 92.0,
            "file_min_line_rate": {"src/scpn_phase_orchestrator/upde/engine.py": 90.0},
        },
    )
    assert len(failures) == 1
    assert "engine.py" in failures[0]


def test_evaluate_accumulates_multiple_failures() -> None:
    summary = _make_summary(
        line_rate=80.0,
        domain={"upde": 60.0},
        files={"src/x.py": 50.0},
    )
    failures = mod.evaluate(
        summary,
        {
            "global_min_line_rate": 92.0,
            "domain_min_line_rate": {"upde": 90.0},
            "file_min_line_rate": {"src/x.py": 100.0},
        },
    )
    assert len(failures) == 3


def test_evaluate_passes_domain_at_or_above_its_floor() -> None:
    summary = _make_summary(line_rate=95.0, domain={"upde": 95.0, "monitor": 90.0})
    failures = mod.evaluate(
        summary,
        {
            "global_min_line_rate": 92.0,
            "domain_min_line_rate": {"upde": 95.0, "monitor": 89.0},
        },
    )
    assert failures == []


def test_evaluate_ignores_non_dict_scoped_map() -> None:
    """A non-dict scoped map is inert when ``evaluate`` is called directly.

    ``load_thresholds`` rejects such configs before they reach ``evaluate``;
    the direct-call surface must not crash on them either.
    """
    summary = _make_summary(line_rate=95.0, domain={"upde": 10.0})
    failures = mod.evaluate(
        summary,
        {"global_min_line_rate": 92.0, "domain_min_line_rate": 5},
    )
    assert failures == []


# ---------------------------------------------------------------------
# main integration
# ---------------------------------------------------------------------


def test_main_returns_zero_on_pass(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    xml = tmp_path / "cov.xml"
    _write_cobertura(
        xml,
        line_rate=0.95,
        classes=[
            (
                "src/scpn_phase_orchestrator/upde/engine.py",
                [(1, 1), (2, 1), (3, 1)],
            ),
        ],
    )
    thresholds = tmp_path / "t.json"
    thresholds.write_text(json.dumps({"global_min_line_rate": 90.0}), encoding="utf-8")
    rc = mod.main(["--coverage-xml", str(xml), "--thresholds", str(thresholds)])
    assert rc == 0
    assert "Coverage guard passed" in capsys.readouterr().out


def test_main_returns_one_on_fail(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    xml = tmp_path / "cov.xml"
    _write_cobertura(
        xml,
        line_rate=0.80,
        classes=[
            (
                "src/scpn_phase_orchestrator/upde/engine.py",
                [(1, 1), (2, 1), (3, 0), (4, 0), (5, 0)],
            ),
        ],
    )
    thresholds = tmp_path / "t.json"
    thresholds.write_text(json.dumps({"global_min_line_rate": 95.0}), encoding="utf-8")
    rc = mod.main(["--coverage-xml", str(xml), "--thresholds", str(thresholds)])
    assert rc == 1
    assert "Coverage guard FAILED" in capsys.readouterr().out


# ---------------------------------------------------------------------
# Branch axis
# ---------------------------------------------------------------------


def _write_cobertura_branches(
    path: Path,
    *,
    line_rate: float,
    branch_rate: float,
    classes: list[tuple[str, list[tuple[int, int, tuple[int, int] | None]]]],
) -> None:
    """Write a minimal Cobertura XML doc carrying branch data.

    classes is a list of ``(filename, [(line_no, hits, condition), ...])``
    tuples where ``condition`` is ``(covered, total)`` for a branch line
    or ``None`` for a plain line.
    """
    total_line_covered = sum(1 for _, lines in classes for _, h, _ in lines if h > 0)
    total_lines = sum(len(lines) for _, lines in classes)
    total_branch_covered = sum(
        cond[0] for _, lines in classes for _, _, cond in lines if cond is not None
    )
    total_branches = sum(
        cond[1] for _, lines in classes for _, _, cond in lines if cond is not None
    )
    parts = [
        f'<coverage line-rate="{line_rate}" branch-rate="{branch_rate}" '
        f'lines-covered="{total_line_covered}" lines-valid="{total_lines}" '
        f'branches-covered="{total_branch_covered}" '
        f'branches-valid="{total_branches}">',
        "<packages><package><classes>",
    ]
    for filename, lines in classes:
        class_covered = sum(1 for _, h, _ in lines if h > 0)
        class_total = len(lines) or 1
        class_rate = class_covered / class_total
        parts.append(f'<class filename="{filename}" line-rate="{class_rate}">')
        parts.append("<lines>")
        for line_no, hits, condition in lines:
            if condition is None:
                parts.append(f'<line number="{line_no}" hits="{hits}"/>')
            else:
                covered, total = condition
                pct = round(100 * covered / total)
                parts.append(
                    f'<line number="{line_no}" hits="{hits}" branch="true" '
                    f'condition-coverage="{pct}% ({covered}/{total})"/>'
                )
        parts.append("</lines></class>")
    parts.append("</classes></package></packages></coverage>")
    path.write_text("".join(parts), encoding="utf-8")


class TestLoadCoverageBranch:
    def test_parses_global_file_and_domain_branch_rates(self, tmp_path: Path) -> None:
        xml = tmp_path / "cov.xml"
        _write_cobertura_branches(
            xml,
            line_rate=0.90,
            branch_rate=5 / 6,
            classes=[
                (
                    "src/scpn_phase_orchestrator/upde/engine.py",
                    [(1, 1, (1, 2)), (2, 1, None), (3, 0, (2, 2))],
                ),
                (
                    "src/scpn_phase_orchestrator/monitor/pid.py",
                    [(1, 1, (2, 2))],
                ),
            ],
        )
        summary = mod.load_coverage(xml)
        assert summary.branch_rate_pct == pytest.approx(5 / 6 * 100)
        assert summary.branches_covered == 5
        assert summary.branches_valid == 6
        # upde/engine.py: (1 + 2) / (2 + 2); monitor/pid.py: 2/2
        assert summary.file_branch_rate_pct[
            "src/scpn_phase_orchestrator/upde/engine.py"
        ] == pytest.approx(75.0)
        assert summary.domain_branch_rate_pct["upde"] == pytest.approx(75.0)
        assert summary.domain_branch_rate_pct["monitor"] == 100.0

    def test_branch_free_file_omitted_from_branch_maps(self, tmp_path: Path) -> None:
        xml = tmp_path / "cov.xml"
        _write_cobertura_branches(
            xml,
            line_rate=1.0,
            branch_rate=1.0,
            classes=[
                (
                    "src/scpn_phase_orchestrator/upde/engine.py",
                    [(1, 1, (2, 2))],
                ),
                (
                    "src/scpn_phase_orchestrator/binding/types.py",
                    [(1, 1, None), (2, 1, None)],
                ),
            ],
        )
        summary = mod.load_coverage(xml)
        assert (
            "src/scpn_phase_orchestrator/binding/types.py"
            not in summary.file_branch_rate_pct
        )
        assert "binding" not in summary.domain_branch_rate_pct
        # The branch-free file still participates in the line maps.
        assert summary.domain_line_rate_pct["binding"] == 100.0

    def test_line_only_report_has_no_branch_data(self, tmp_path: Path) -> None:
        xml = tmp_path / "cov.xml"
        _write_cobertura(
            xml,
            line_rate=0.95,
            classes=[
                (
                    "src/scpn_phase_orchestrator/upde/engine.py",
                    [(1, 1), (2, 1)],
                ),
            ],
        )
        summary = mod.load_coverage(xml)
        assert summary.branch_rate_pct is None
        assert summary.file_branch_rate_pct == {}
        assert summary.domain_branch_rate_pct == {}
        assert summary.branches_valid == 0

    def test_malformed_condition_coverage_rejected(self, tmp_path: Path) -> None:
        xml = tmp_path / "cov.xml"
        xml.write_text(
            '<coverage line-rate="1.0" branch-rate="1.0" '
            'lines-covered="1" lines-valid="1" '
            'branches-covered="1" branches-valid="2">'
            "<packages><package><classes>"
            '<class filename="src/scpn_phase_orchestrator/upde/engine.py" '
            'line-rate="1.0">'
            '<lines><line number="1" hits="1" branch="true" '
            'condition-coverage="garbage"/></lines></class>'
            "</classes></package></packages></coverage>",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Malformed condition-coverage"):
            mod.load_coverage(xml)


class TestLoadThresholdsBranch:
    def test_accepts_branch_keys(self, tmp_path: Path) -> None:
        config = tmp_path / "t.json"
        config.write_text(
            json.dumps(
                {
                    "global_min_line_rate": 92.0,
                    "global_min_branch_rate": 85.0,
                    "domain_min_branch_rate": {"upde": 90.0},
                    "file_min_branch_rate": {"src/foo.py": 100.0},
                }
            ),
            encoding="utf-8",
        )
        data = mod.load_thresholds(config)
        assert data["global_min_branch_rate"] == 85.0
        assert data["domain_min_branch_rate"] == {"upde": 90.0}

    def test_rejects_out_of_range_global_branch_rate(self, tmp_path: Path) -> None:
        config = tmp_path / "t.json"
        config.write_text(
            json.dumps({"global_min_line_rate": 92.0, "global_min_branch_rate": 120.0}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="global_min_branch_rate"):
            mod.load_thresholds(config)

    def test_rejects_non_dict_branch_map(self, tmp_path: Path) -> None:
        config = tmp_path / "t.json"
        config.write_text(
            json.dumps({"global_min_line_rate": 92.0, "domain_min_branch_rate": 5}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="must be a JSON object"):
            mod.load_thresholds(config)

    def test_rejects_out_of_range_domain_branch_value(self, tmp_path: Path) -> None:
        config = tmp_path / "t.json"
        config.write_text(
            json.dumps(
                {
                    "global_min_line_rate": 92.0,
                    "domain_min_branch_rate": {"upde": 130.0},
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=r"domain_min_branch_rate\[upde\]"):
            mod.load_thresholds(config)


class TestEvaluateBranch:
    def test_fails_closed_when_branch_floors_meet_line_only_report(self) -> None:
        summary = _make_summary(line_rate=95.0, branch_rate=None)
        failures = mod.evaluate(
            summary,
            {"global_min_line_rate": 90.0, "global_min_branch_rate": 80.0},
        )
        assert len(failures) == 1
        assert "no branch data" in failures[0]

    def test_fails_closed_on_scoped_branch_floors_alone(self) -> None:
        summary = _make_summary(line_rate=95.0, branch_rate=None)
        failures = mod.evaluate(
            summary,
            {"global_min_line_rate": 90.0, "domain_min_branch_rate": {"upde": 80.0}},
        )
        assert len(failures) == 1
        assert "no branch data" in failures[0]

    def test_passes_when_branch_above_threshold(self) -> None:
        summary = _make_summary(line_rate=95.0, branch_rate=88.0)
        failures = mod.evaluate(
            summary,
            {"global_min_line_rate": 90.0, "global_min_branch_rate": 85.0},
        )
        assert failures == []

    def test_flags_global_branch_regression(self) -> None:
        summary = _make_summary(line_rate=95.0, branch_rate=80.0)
        failures = mod.evaluate(
            summary,
            {"global_min_line_rate": 90.0, "global_min_branch_rate": 85.0},
        )
        assert len(failures) == 1
        assert "Global branch coverage" in failures[0] and "80.00" in failures[0]

    def test_flags_domain_branch_regression(self) -> None:
        summary = _make_summary(
            line_rate=95.0, branch_rate=90.0, branch_domain={"upde": 70.0}
        )
        failures = mod.evaluate(
            summary,
            {"global_min_line_rate": 90.0, "domain_min_branch_rate": {"upde": 85.0}},
        )
        assert len(failures) == 1
        assert "upde" in failures[0] and "branch" in failures[0]

    def test_flags_file_branch_regression(self) -> None:
        summary = _make_summary(
            line_rate=95.0,
            branch_rate=90.0,
            branch_files={"src/scpn_phase_orchestrator/upde/engine.py": 60.0},
        )
        failures = mod.evaluate(
            summary,
            {
                "global_min_line_rate": 90.0,
                "file_min_branch_rate": {
                    "src/scpn_phase_orchestrator/upde/engine.py": 90.0
                },
            },
        )
        assert len(failures) == 1
        assert "engine.py" in failures[0] and "branch" in failures[0]

    def test_skips_domain_missing_from_branch_map(self) -> None:
        summary = _make_summary(
            line_rate=95.0, branch_rate=90.0, branch_domain={"upde": 95.0}
        )
        failures = mod.evaluate(
            summary,
            {
                "global_min_line_rate": 90.0,
                "domain_min_branch_rate": {"missing_domain": 99.0},
            },
        )
        assert failures == []

    def test_rejects_non_numeric_global_branch_rate(self) -> None:
        summary = _make_summary(line_rate=95.0, branch_rate=90.0)
        with pytest.raises(TypeError, match="global_min_branch_rate"):
            mod.evaluate(
                summary,
                {"global_min_line_rate": 90.0, "global_min_branch_rate": "high"},
            )

    def test_line_and_branch_failures_accumulate(self) -> None:
        summary = _make_summary(
            line_rate=80.0,
            branch_rate=70.0,
            branch_domain={"upde": 60.0},
        )
        failures = mod.evaluate(
            summary,
            {
                "global_min_line_rate": 92.0,
                "global_min_branch_rate": 85.0,
                "domain_min_branch_rate": {"upde": 90.0},
            },
        )
        assert len(failures) == 3


def test_main_prints_branch_block_and_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    xml = tmp_path / "cov.xml"
    _write_cobertura_branches(
        xml,
        line_rate=1.0,
        branch_rate=0.9,
        classes=[
            (
                "src/scpn_phase_orchestrator/upde/engine.py",
                [(1, 1, (9, 10)), (2, 1, None)],
            ),
        ],
    )
    thresholds = tmp_path / "t.json"
    thresholds.write_text(
        json.dumps({"global_min_line_rate": 90.0, "global_min_branch_rate": 85.0}),
        encoding="utf-8",
    )
    rc = mod.main(["--coverage-xml", str(xml), "--thresholds", str(thresholds)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Coverage branch rate: 90.00% (9/10)" in out
    assert "Coverage guard passed" in out


def test_main_fails_on_branch_regression(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    xml = tmp_path / "cov.xml"
    _write_cobertura_branches(
        xml,
        line_rate=1.0,
        branch_rate=0.5,
        classes=[
            (
                "src/scpn_phase_orchestrator/upde/engine.py",
                [(1, 1, (1, 2))],
            ),
        ],
    )
    thresholds = tmp_path / "t.json"
    thresholds.write_text(
        json.dumps({"global_min_line_rate": 90.0, "global_min_branch_rate": 85.0}),
        encoding="utf-8",
    )
    rc = mod.main(["--coverage-xml", str(xml), "--thresholds", str(thresholds)])
    assert rc == 1
    assert "Global branch coverage 50.00%" in capsys.readouterr().out


def test_script_entry_point_runs_as_subprocess(tmp_path: Path) -> None:
    """The tool is a real CLI: ``__main__`` wiring exits 0 on a passing report."""
    xml = tmp_path / "cov.xml"
    _write_cobertura(
        xml,
        line_rate=0.95,
        classes=[("src/scpn_phase_orchestrator/upde/engine.py", [(1, 1)])],
    )
    thresholds = tmp_path / "t.json"
    thresholds.write_text(json.dumps({"global_min_line_rate": 90.0}), encoding="utf-8")
    result = subprocess.run(  # noqa: S603 — fixed argv, no shell
        [
            sys.executable,
            str(TOOLS_DIR / "coverage_guard.py"),
            "--coverage-xml",
            str(xml),
            "--thresholds",
            str(thresholds),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Coverage guard passed." in result.stdout
