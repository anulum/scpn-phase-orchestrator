# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Source-bound Phase-SINDy coverage admission

"""Admit measured SINDy coverage with exact, reviewed residuals.

Observer callbacks execute with Python tracing suspended, including under
coverage.py's monitoring core. Their named measurement exclusions do not
excuse missing numerical computation. Bounded native SVD nonconvergence remains
uncovered debt; this command never reports native statement coverage as 100%.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import cast

from coverage import Coverage

ROOT = Path(__file__).resolve().parents[1]


def _object(value: object) -> dict[str, object]:
    """Require a JSON object at a report or policy boundary."""
    if not isinstance(value, dict):
        raise ValueError("expected an object")
    return cast(dict[str, object], value)


def _list(value: object) -> list[object]:
    """Require a JSON array without accepting string iteration."""
    if not isinstance(value, list):
        raise ValueError("expected an array")
    return cast(list[object], value)


def _read(path: Path) -> dict[str, object]:
    """Read one UTF-8 JSON object; file and decoding faults refuse admission."""
    value: object = json.loads(path.read_text(encoding="utf-8"))
    return _object(value)


def _integers(value: object) -> list[int]:
    """Require exact integer line numbers or counters, excluding boolean aliases."""
    items = _list(value)
    if any(type(item) is not int for item in items):
        raise ValueError("expected integer counters")
    return cast(list[int], items)


def _arcs(value: object) -> set[tuple[int, int]]:
    """Decode distinct branch outcomes without accepting partial vectors."""
    result: set[tuple[int, int]] = set()
    for raw in _list(value):
        arc = _integers(raw)
        if len(arc) != 2 or tuple(arc) in result:
            raise ValueError("expected distinct branch pairs")
        result.add((arc[0], arc[1]))
    return result


def _digest(path: Path) -> str:
    """Bind the reviewed policy to actual current source bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_reports(
    python_report: Path, native_report: Path, *, root: Path = ROOT
) -> dict[str, object]:
    """Refuse unreviewed exclusions, incomplete measurements and source drift.

    Parameters
    ----------
    python_report : Path
        Combined coverage.py branch report for the six owning Python surfaces.
    native_report : Path
        LLVM branch export from actual ``spo-engine`` SINDy tests.
    root : Path, default=ROOT
        Checkout whose current bytes the policy names.

    Returns
    -------
    dict[str, object]
        Explicit measurement exclusions and remaining native convergence debt.

    Raises
    ------
    ValueError
        If source identity, report membership or coverage violates the policy.
    OSError
        If a required input cannot be read.
    """
    policy = _read(root / "tools/phase_sindy_coverage_policy.json")
    python = _read(python_report)
    if _object(python.get("meta")).get("branch_coverage") is not True:
        raise ValueError("Python branch measurement is required")
    files = _object(python.get("files"))
    exclusions: dict[str, object] = {}
    analysis = Coverage.current() or Coverage()
    for filename, raw in _object(policy.get("python")).items():
        expected = _object(raw)
        source = root / filename
        if _digest(source) != expected.get("sha256"):
            raise ValueError(f"source changed: {filename}")
        measured = _object(files.get(filename))
        summary = _object(measured.get("summary"))
        statements = set(analysis.analysis2(str(source))[1])
        executed = set(_integers(measured.get("executed_lines")))
        missing = set(_integers(measured.get("missing_lines")))
        if executed & missing or executed | missing != statements:
            raise ValueError(f"incomplete statement membership: {filename}")
        if (
            summary.get("num_statements") != expected.get("statements")
            or summary.get("num_branches") != expected.get("branches")
            or measured.get("excluded_lines") != expected.get("excluded_lines")
        ):
            raise ValueError(f"measurement configuration changed: {filename}")
        allowed_lines = set(_integers(expected.get("measurement_exclusions")))
        arcs = _arcs(measured.get("missing_branches"))
        allowed_arcs = _arcs(expected.get("measurement_arcs"))
        executed_arcs = _arcs(measured.get("executed_branches"))
        if executed_arcs & arcs or len(executed_arcs | arcs) != expected.get(
            "branches"
        ):
            raise ValueError(f"incomplete branch membership: {filename}")
        if not missing <= allowed_lines or not arcs <= allowed_arcs:
            raise ValueError(f"unreviewed Python coverage gap: {filename}")
        exclusions[filename] = {"lines": sorted(missing), "arcs": sorted(arcs)}

    expected_native = _object(policy.get("native"))
    native_name = expected_native.get("path")
    if not isinstance(native_name, str):
        raise ValueError("native source path is required")
    if _digest(root / native_name) != expected_native.get("sha256") or _digest(
        root / "spo-kernel/Cargo.lock"
    ) != expected_native.get("lock_sha256"):
        raise ValueError("native source or locked dependency changed")
    exports = _list(_read(native_report).get("data"))
    members = [
        _object(member)
        for export in exports
        for member in _list(_object(export).get("files"))
        if str(_object(member).get("filename", ""))
        .replace("\\", "/")
        .endswith(native_name)
    ]
    if len(members) != 1:
        raise ValueError("exactly one native SINDy measurement is required")
    native = members[0]
    branches = _list(native.get("branches"))
    if len(branches) != expected_native.get("branch_records"):
        raise ValueError("native branch membership changed")
    for raw in branches:
        branch = _integers(raw)
        if len(branch) != 9 or branch[4] <= 0 or branch[5] <= 0:
            raise ValueError("every measured native branch outcome must execute")
    missing_regions: list[list[object]] = []
    segments = _list(native.get("segments"))
    if len(segments) != expected_native.get("segment_records"):
        raise ValueError("native region membership changed")
    for raw in segments:
        segment = _list(raw)
        if (
            len(segment) != 6
            or any(type(item) is not int for item in segment[:3])
            or any(type(item) is not bool for item in segment[3:])
        ):
            raise ValueError("invalid native region vector")
        if segment[2] == 0 and segment[3] is True and segment[4] is True:
            missing_regions.append(segment)
    allowed_regions = _list(expected_native.get("residual_regions"))
    if any(region not in allowed_regions for region in missing_regions):
        raise ValueError("unreviewed native coverage gap")
    return {
        "python_measurement_exclusions": exclusions,
        "native_missing_regions": missing_regions,
        "native_convergence_debt": expected_native.get("convergence_debt"),
        "native_full_coverage": False,
    }


def main(argv: list[str] | None = None) -> int:
    """Apply source-bound admission to supplied real Python and LLVM exports.

    Parameters
    ----------
    argv : list[str] or None
        Command arguments; None uses the process command line.

    Returns
    -------
    int
        Zero for an admitted report pair, one for an authored refusal.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-report", type=Path, required=True)
    parser.add_argument("--native-report", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        result = verify_reports(args.python_report, args.native_report, root=args.root)
    except (OSError, ValueError, TypeError):
        print(
            "Phase-SINDy coverage refused: input, source or measurement violates policy"
        )
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
