#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Coverage regression guard

"""Coverage regression guard for scpn-phase-orchestrator.

Parses a Cobertura coverage XML report and enforces no-decrease floors
from a JSON thresholds file. Two independent gate axes are supported:

* **Line coverage** — always evaluated; ``global_min_line_rate`` is
  required, ``domain_min_line_rate`` / ``file_min_line_rate`` optional.
* **Branch coverage** — evaluated when any ``*_min_branch_rate`` key is
  configured. The branch axis fails closed: configured branch floors
  against an XML produced without branch instrumentation are a gate
  failure, not a silent skip.

The line lane (CI ``coverage-guard`` job) uses
``tools/coverage_guard_thresholds.json``; the perf-isolated branch lane
(CI ``branch-coverage`` job) uses
``tools/coverage_guard_branch_thresholds.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COVERAGE_XML = REPO_ROOT / "coverage-python.xml"
DEFAULT_THRESHOLDS = REPO_ROOT / "tools" / "coverage_guard_thresholds.json"

_BRANCH_THRESHOLD_KEYS = (
    "global_min_branch_rate",
    "domain_min_branch_rate",
    "file_min_branch_rate",
)
_CONDITION_COVERAGE_PATTERN = re.compile(r"\((\d+)/(\d+)\)")


@dataclass(frozen=True)
class CoverageSummary:
    """Parsed line and branch rates from one Cobertura XML report.

    Branch fields default to the no-branch-data state so line-only
    reports (the main coverage matrix runs without ``--cov-branch``)
    parse into the same summary type: ``branch_rate_pct`` is ``None``
    and the per-file / per-domain branch maps are empty.
    """

    line_rate_pct: float
    file_line_rate_pct: dict[str, float]
    domain_line_rate_pct: dict[str, float]
    lines_covered: int
    lines_valid: int
    branch_rate_pct: float | None = None
    file_branch_rate_pct: dict[str, float] = field(default_factory=dict)
    domain_branch_rate_pct: dict[str, float] = field(default_factory=dict)
    branches_covered: int = 0
    branches_valid: int = 0


def _resolve(path_value: str) -> Path:
    """Resolve ``path_value`` against the repository root when relative."""
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _validate_percent(value: float, *, label: str) -> float:
    """Return ``value`` as a float after checking it is a finite percentage."""
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite.")
    if numeric < 0.0 or numeric > 100.0:
        raise ValueError(f"{label} must be in [0, 100].")
    return numeric


def _domain_for(filename: str) -> str:
    """Map a source filename to its top-level package domain.

    ``src/scpn_phase_orchestrator/<domain>/...`` maps to ``<domain>``;
    anything outside that layout falls into ``"other"``.
    """
    normalized = filename.replace("\\", "/")
    parts = normalized.split("/")
    if len(parts) >= 3 and parts[0] == "src" and parts[1] == "scpn_phase_orchestrator":
        return parts[2]
    return "other"


def _branch_counts(cls: ET.Element, *, filename: str) -> tuple[int, int]:
    """Sum covered/total branch conditions over a class's line elements.

    Cobertura marks branch lines with ``condition-coverage="P% (c/t)"``;
    a malformed attribute raises rather than silently under-counting.
    Returns ``(0, 0)`` for a file with no branch lines.
    """
    covered = 0
    total = 0
    for line in cls.findall("./lines/line"):
        condition = line.get("condition-coverage")
        if condition is None:
            continue
        match = _CONDITION_COVERAGE_PATTERN.search(condition)
        if match is None:
            raise ValueError(
                f"Malformed condition-coverage {condition!r} in {filename}."
            )
        covered += int(match.group(1))
        total += int(match.group(2))
    return covered, total


def load_coverage(path: Path) -> CoverageSummary:
    """Parse a Cobertura XML report into a :class:`CoverageSummary`.

    Line rates come from the root / per-class ``line-rate`` attributes
    with per-domain rates aggregated from line hit counts. Branch rates
    are populated only when the report carries branch data
    (``branches-valid`` > 0): the global rate from the root
    ``branch-rate`` attribute, per-file and per-domain rates aggregated
    from ``condition-coverage`` counts, with branch-free files omitted
    from the per-file map so they cannot masquerade as 0 %-covered.
    """
    if not path.exists():
        raise FileNotFoundError(f"Coverage XML not found: {path}")
    root = ET.parse(path).getroot()  # noqa: S314 — input is local CI artifact

    line_rate = _validate_percent(
        float(root.get("line-rate", "0.0")) * 100.0, label="line_rate"
    )
    lines_covered = int(root.get("lines-covered", "0"))
    lines_valid = int(root.get("lines-valid", "0"))
    branches_covered = int(root.get("branches-covered", "0"))
    branches_valid = int(root.get("branches-valid", "0"))
    has_branch_data = branches_valid > 0
    branch_rate: float | None = None
    if has_branch_data:
        branch_rate = _validate_percent(
            float(root.get("branch-rate", "0.0")) * 100.0, label="branch_rate"
        )

    file_line_rate_pct: dict[str, float] = {}
    domain_hits: dict[str, tuple[int, int]] = {}
    file_branch_rate_pct: dict[str, float] = {}
    domain_branch_hits: dict[str, tuple[int, int]] = {}
    for cls in root.findall(".//class"):
        filename = cls.get("filename", "").replace("\\", "/")
        if not filename:
            continue
        class_rate = _validate_percent(
            float(cls.get("line-rate", "0.0")) * 100.0,
            label=f"line_rate[{filename}]",
        )
        file_line_rate_pct[filename] = class_rate

        covered = 0
        total = 0
        for line in cls.findall("./lines/line"):
            total += 1
            if int(line.get("hits", "0")) > 0:
                covered += 1
        domain = _domain_for(filename)
        existing = domain_hits.get(domain, (0, 0))
        domain_hits[domain] = (existing[0] + covered, existing[1] + total)

        if has_branch_data:
            branch_covered, branch_total = _branch_counts(cls, filename=filename)
            if branch_total > 0:
                file_branch_rate_pct[filename] = _validate_percent(
                    100.0 * branch_covered / branch_total,
                    label=f"branch_rate[{filename}]",
                )
                existing_branch = domain_branch_hits.get(domain, (0, 0))
                domain_branch_hits[domain] = (
                    existing_branch[0] + branch_covered,
                    existing_branch[1] + branch_total,
                )

    domain_line_rate_pct: dict[str, float] = {}
    for domain, (covered, total) in domain_hits.items():
        pct = 100.0 * covered / total if total > 0 else 0.0
        domain_line_rate_pct[domain] = _validate_percent(
            pct, label=f"domain_line_rate[{domain}]"
        )

    domain_branch_rate_pct: dict[str, float] = {}
    for domain, (covered, total) in domain_branch_hits.items():
        domain_branch_rate_pct[domain] = _validate_percent(
            100.0 * covered / total, label=f"domain_branch_rate[{domain}]"
        )

    return CoverageSummary(
        line_rate_pct=line_rate,
        file_line_rate_pct=file_line_rate_pct,
        domain_line_rate_pct=domain_line_rate_pct,
        lines_covered=lines_covered,
        lines_valid=lines_valid,
        branch_rate_pct=branch_rate,
        file_branch_rate_pct=file_branch_rate_pct,
        domain_branch_rate_pct=domain_branch_rate_pct,
        branches_covered=branches_covered,
        branches_valid=branches_valid,
    )


def load_thresholds(path: Path) -> dict[str, object]:
    """Load and validate the thresholds JSON.

    ``global_min_line_rate`` is required; ``global_min_branch_rate`` is
    optional and activates the branch axis. The four scoped maps
    (``domain_min_line_rate``, ``file_min_line_rate``,
    ``domain_min_branch_rate``, ``file_min_branch_rate``) are optional
    JSON objects whose values must be valid percentages.
    """
    if not path.exists():
        raise FileNotFoundError(f"Coverage threshold config not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Threshold config must be a JSON object.")
    if "global_min_line_rate" not in data:
        raise ValueError("Threshold config must define global_min_line_rate.")
    _validate_percent(float(data["global_min_line_rate"]), label="global_min_line_rate")
    if "global_min_branch_rate" in data:
        _validate_percent(
            float(data["global_min_branch_rate"]), label="global_min_branch_rate"
        )
    for key in (
        "domain_min_line_rate",
        "file_min_line_rate",
        "domain_min_branch_rate",
        "file_min_branch_rate",
    ):
        value = data.get(key, {})
        if not isinstance(value, dict):
            raise ValueError(f"{key} must be a JSON object when provided.")
        for sub_key, sub_value in value.items():
            _validate_percent(float(sub_value), label=f"{key}[{sub_key}]")
    return data


def _global_minimum(thresholds: dict[str, object], key: str) -> float:
    """Return the numeric global floor stored under ``key``."""
    raw = thresholds[key]
    if not isinstance(raw, (int, float)):
        raise TypeError(f"{key} must be numeric, got {type(raw).__name__}")
    return float(raw)


def _scoped_failures(
    thresholds: dict[str, object],
    *,
    key: str,
    observed: dict[str, float],
    scope: str,
    metric: str,
) -> list[str]:
    """Compare one scoped threshold map against observed rates.

    Names configured in the thresholds but absent from ``observed`` are
    skipped: a domain or file can legitimately be missing from a given
    lane's report (optional backends, lane-specific test selection).
    """
    failures: list[str] = []
    configured = thresholds.get(key, {})
    if isinstance(configured, dict):
        for name, threshold in configured.items():
            target = float(threshold)
            value = observed.get(name)
            if value is None:
                continue
            if value < target:
                failures.append(
                    f"{scope} '{name}' {metric} coverage {value:.2f}% "
                    f"< threshold {target:.2f}%."
                )
    return failures


def evaluate(summary: CoverageSummary, thresholds: dict[str, object]) -> list[str]:
    """Return the list of gate failures for ``summary`` under ``thresholds``.

    The line axis is always evaluated. The branch axis is evaluated when
    any branch threshold key is configured, and fails closed when the
    report carries no branch data — a branch-gated lane that silently
    ran without branch instrumentation must not pass.
    """
    failures: list[str] = []

    global_line_min = _global_minimum(thresholds, "global_min_line_rate")
    if summary.line_rate_pct < global_line_min:
        failures.append(
            f"Global line coverage {summary.line_rate_pct:.2f}% "
            f"< threshold {global_line_min:.2f}%."
        )
    failures.extend(
        _scoped_failures(
            thresholds,
            key="domain_min_line_rate",
            observed=summary.domain_line_rate_pct,
            scope="Domain",
            metric="line",
        )
    )
    failures.extend(
        _scoped_failures(
            thresholds,
            key="file_min_line_rate",
            observed=summary.file_line_rate_pct,
            scope="File",
            metric="line",
        )
    )

    branch_configured = any(key in thresholds for key in _BRANCH_THRESHOLD_KEYS)
    branch_rate = summary.branch_rate_pct
    if branch_configured and branch_rate is None:
        failures.append(
            "Branch thresholds are configured but the coverage XML carries "
            "no branch data; run coverage with branch instrumentation."
        )
    elif branch_configured and branch_rate is not None:
        if "global_min_branch_rate" in thresholds:
            global_branch_min = _global_minimum(thresholds, "global_min_branch_rate")
            if branch_rate < global_branch_min:
                failures.append(
                    f"Global branch coverage {branch_rate:.2f}% "
                    f"< threshold {global_branch_min:.2f}%."
                )
        failures.extend(
            _scoped_failures(
                thresholds,
                key="domain_min_branch_rate",
                observed=summary.domain_branch_rate_pct,
                scope="Domain",
                metric="branch",
            )
        )
        failures.extend(
            _scoped_failures(
                thresholds,
                key="file_min_branch_rate",
                observed=summary.file_branch_rate_pct,
                scope="File",
                metric="branch",
            )
        )

    return failures


def main(argv: list[str] | None = None) -> int:
    """Run the guard from the command line; return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coverage-xml",
        default=str(DEFAULT_COVERAGE_XML),
        help="Path to coverage XML report (Cobertura format).",
    )
    parser.add_argument(
        "--thresholds",
        default=str(DEFAULT_THRESHOLDS),
        help="JSON file containing coverage thresholds.",
    )
    args = parser.parse_args(argv)

    coverage_path = _resolve(args.coverage_xml)
    thresholds_path = _resolve(args.thresholds)
    summary = load_coverage(coverage_path)
    thresholds = load_thresholds(thresholds_path)

    print(
        f"Coverage line rate: {summary.line_rate_pct:.2f}% "
        f"({summary.lines_covered}/{summary.lines_valid})"
    )
    for domain, pct in sorted(summary.domain_line_rate_pct.items()):
        print(f"  {domain:20s}: {pct:6.2f}%")
    if summary.branch_rate_pct is not None:
        print(
            f"Coverage branch rate: {summary.branch_rate_pct:.2f}% "
            f"({summary.branches_covered}/{summary.branches_valid})"
        )
        for domain, pct in sorted(summary.domain_branch_rate_pct.items()):
            print(f"  {domain:20s}: {pct:6.2f}%")

    failures = evaluate(summary, thresholds)
    if failures:
        print("Coverage guard FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("Coverage guard passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
