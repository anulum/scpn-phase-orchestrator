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
runs. Cover the four behaviour surfaces:

* ``_validate_percent`` — bounds [0, 100], rejects NaN/Inf.
* ``_domain_for`` — ``src/scpn_phase_orchestrator/<domain>/...`` mapping;
  everything else falls into ``"other"``.
* ``load_coverage`` / ``load_thresholds`` — XML / JSON parsers,
  required-field validation, file-not-found surface.
* ``evaluate`` / ``main`` — global, per-domain, per-file threshold
  branches; exit codes on pass / fail.
"""

from __future__ import annotations

import importlib.util
import json
import math
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
    assert (
        summary.file_line_rate_pct["src/scpn_phase_orchestrator/upde/engine.py"]
        == pytest.approx(2 / 3 * 100)
    )


def test_load_coverage_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        mod.load_coverage(tmp_path / "nope.xml")


def test_load_coverage_skips_empty_filename(tmp_path: Path) -> None:
    xml = tmp_path / "cov.xml"
    xml.write_text(
        '<coverage line-rate="1.0" lines-covered="1" lines-valid="1">'
        '<packages><package><classes>'
        '<class filename="" line-rate="1.0">'
        '<lines><line number="1" hits="1"/></lines></class>'
        '</classes></package></packages></coverage>',
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
) -> object:
    return mod.CoverageSummary(
        line_rate_pct=line_rate,
        file_line_rate_pct=files or {},
        domain_line_rate_pct=domain or {},
        lines_covered=0,
        lines_valid=0,
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
            "file_min_line_rate": {
                "src/scpn_phase_orchestrator/upde/engine.py": 90.0
            },
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
    thresholds.write_text(
        json.dumps({"global_min_line_rate": 90.0}), encoding="utf-8"
    )
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
    thresholds.write_text(
        json.dumps({"global_min_line_rate": 95.0}), encoding="utf-8"
    )
    rc = mod.main(["--coverage-xml", str(xml), "--thresholds", str(thresholds)])
    assert rc == 1
    assert "Coverage guard FAILED" in capsys.readouterr().out
