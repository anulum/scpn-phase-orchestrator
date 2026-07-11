# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — coverage-report docs drift guard

"""Keep the published coverage figures in step with the enforced ratchet.

The V&V report and testing guide publish the coverage posture: the measured rate
and the per-domain no-decrease ratchet floor. This drift guard reads the global
ratchet floors from the ``coverage_guard`` threshold files and fails if either
document stops citing them, so a raised floor forces the docs to be updated rather
than silently going stale (the SPO-G04 finding: the docs must publish the measured
coverage and the raise path, not a misleading historical floor).
"""

from __future__ import annotations

import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_LINE_THRESHOLDS = _REPO / "tools" / "coverage_guard_thresholds.json"
_BRANCH_THRESHOLDS = _REPO / "tools" / "coverage_guard_branch_thresholds.json"
_VALIDATION_REPORT = _REPO / "docs" / "VALIDATION_REPORT.md"
_TESTING_GUIDE = _REPO / "docs" / "guide" / "testing.md"


def _global_line_floor() -> int:
    data = json.loads(_LINE_THRESHOLDS.read_text(encoding="utf-8"))
    return int(data["global_min_line_rate"])


def _global_branch_floor() -> int:
    data = json.loads(_BRANCH_THRESHOLDS.read_text(encoding="utf-8"))
    return int(data["global_min_branch_rate"])


def test_validation_report_cites_the_ratchet_floors() -> None:
    doc = _VALIDATION_REPORT.read_text(encoding="utf-8")
    assert f"{_global_line_floor()}%" in doc
    assert f"{_global_branch_floor()}%" in doc


def test_validation_report_documents_the_ratchet_gate() -> None:
    doc = _VALIDATION_REPORT.read_text(encoding="utf-8")
    assert "ratchet" in doc
    assert "coverage_guard_thresholds.json" in doc
    assert "coverage_guard_branch_thresholds.json" in doc


def test_testing_guide_cites_the_ratchet_floors() -> None:
    doc = _TESTING_GUIDE.read_text(encoding="utf-8")
    assert f"{_global_line_floor()}%" in doc
    assert f"{_global_branch_floor()}%" in doc


def test_testing_guide_documents_the_ratchet_not_a_flat_gate() -> None:
    doc = _TESTING_GUIDE.read_text(encoding="utf-8")
    assert "ratchet" in doc
    assert "coverage_guard" in doc
