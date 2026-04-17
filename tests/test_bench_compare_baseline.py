# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for the benchmark regression guard

"""Defensive tests for ``bench/compare_baseline.py``.

The script is the CI benchmark-regression guard. A silent bug in it
would let real regressions through or, conversely, break the pipeline
on well-behaved builds. Cover the four behavioural branches:

* Exit 0 when no config regresses past the 20% threshold.
* Exit 1 with a human-readable summary when at least one does.
* Exit 2 with a usage message on the wrong argv length.
* Baseline entries keyed by (n_osc, method, backend); missing keys in
  ``current`` are silently ignored (partial runs allowed).

Exercised both the list-form and dict-form baseline JSON layouts.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPARE_SCRIPT = REPO_ROOT / "bench" / "compare_baseline.py"


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj), encoding="utf-8")


def _run(
    baseline_path: Path, current_path: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(COMPARE_SCRIPT), str(baseline_path), str(current_path)],
        capture_output=True,
        text=True,
        check=False,
    )


def _entry(
    n_osc: int,
    method: str,
    backend: str,
    us_per_step: float,
) -> dict[str, Any]:
    return {
        "n_osc": n_osc,
        "method": method,
        "backend": backend,
        "us_per_step": us_per_step,
        "steps": 1000,
    }


# ---------------------------------------------------------------------
# PASS cases
# ---------------------------------------------------------------------


def test_exact_match_is_pass(tmp_path: Path) -> None:
    """Identical baseline and current yield exit 0 and a no-regression
    summary."""
    baseline = [_entry(8, "euler", "rust", 7.5)]
    current = [_entry(8, "euler", "rust", 7.5)]
    _write_json(tmp_path / "b.json", baseline)
    _write_json(tmp_path / "c.json", current)
    proc = _run(tmp_path / "b.json", tmp_path / "c.json")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "within regression threshold" in proc.stdout


def test_improvement_is_pass(tmp_path: Path) -> None:
    """Faster than baseline is clearly a pass; the script must not error
    on negative percentage deltas."""
    baseline = [_entry(16, "rk4", "rust", 25.0)]
    current = [_entry(16, "rk4", "rust", 10.0)]  # 60% faster
    _write_json(tmp_path / "b.json", baseline)
    _write_json(tmp_path / "c.json", current)
    proc = _run(tmp_path / "b.json", tmp_path / "c.json")
    assert proc.returncode == 0
    assert "-60.0%" in proc.stdout


def test_boundary_exactly_20_percent_is_pass(tmp_path: Path) -> None:
    """20% slower is the boundary; threshold is `pct <= 20`."""
    baseline = [_entry(8, "euler", "rust", 10.0)]
    current = [_entry(8, "euler", "rust", 12.0)]  # exactly +20%
    _write_json(tmp_path / "b.json", baseline)
    _write_json(tmp_path / "c.json", current)
    proc = _run(tmp_path / "b.json", tmp_path / "c.json")
    assert proc.returncode == 0
    assert "+20.0%" in proc.stdout


# ---------------------------------------------------------------------
# FAIL cases
# ---------------------------------------------------------------------


def test_regression_above_threshold_fails(tmp_path: Path) -> None:
    """>20% slower triggers exit 1 and the regression is listed."""
    baseline = [_entry(32, "euler", "rust", 10.0)]
    current = [_entry(32, "euler", "rust", 15.0)]  # +50%
    _write_json(tmp_path / "b.json", baseline)
    _write_json(tmp_path / "c.json", current)
    proc = _run(tmp_path / "b.json", tmp_path / "c.json")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "FAIL" in proc.stdout
    assert "+50.0%" in proc.stdout
    assert "1 regression(s)" in proc.stdout


def test_multiple_regressions_all_reported(tmp_path: Path) -> None:
    """All failing configs appear in the summary; count matches."""
    baseline = [
        _entry(8, "euler", "rust", 10.0),
        _entry(16, "rk4", "rust", 20.0),
        _entry(32, "rk45", "rust", 40.0),
    ]
    current = [
        _entry(8, "euler", "rust", 15.0),  # +50%
        _entry(16, "rk4", "rust", 20.0),  # 0%
        _entry(32, "rk45", "rust", 70.0),  # +75%
    ]
    _write_json(tmp_path / "b.json", baseline)
    _write_json(tmp_path / "c.json", current)
    proc = _run(tmp_path / "b.json", tmp_path / "c.json")
    assert proc.returncode == 1
    assert "2 regression(s)" in proc.stdout


# ---------------------------------------------------------------------
# Baseline formats
# ---------------------------------------------------------------------


def test_dict_form_baseline_accepted(tmp_path: Path) -> None:
    """The script also accepts ``{"results": [...]}`` dict layouts."""
    baseline = {"results": [_entry(8, "euler", "rust", 10.0)]}
    current = {"results": [_entry(8, "euler", "rust", 11.0)]}
    _write_json(tmp_path / "b.json", baseline)
    _write_json(tmp_path / "c.json", current)
    proc = _run(tmp_path / "b.json", tmp_path / "c.json")
    assert proc.returncode == 0


def test_missing_current_key_silently_skipped(tmp_path: Path) -> None:
    """Current entries without a baseline counterpart are ignored: a
    partial benchmark run does not fail the guard."""
    baseline = [_entry(8, "euler", "rust", 10.0)]
    current = [_entry(64, "rk4", "python", 100.0)]  # completely different key
    _write_json(tmp_path / "b.json", baseline)
    _write_json(tmp_path / "c.json", current)
    proc = _run(tmp_path / "b.json", tmp_path / "c.json")
    assert proc.returncode == 0
    assert "within regression threshold" in proc.stdout


def test_zero_baseline_skipped(tmp_path: Path) -> None:
    """Entries with a zero or negative baseline are skipped to avoid
    division-by-zero; the run still passes overall."""
    baseline = [_entry(8, "euler", "rust", 0.0)]
    current = [_entry(8, "euler", "rust", 5.0)]
    _write_json(tmp_path / "b.json", baseline)
    _write_json(tmp_path / "c.json", current)
    proc = _run(tmp_path / "b.json", tmp_path / "c.json")
    assert proc.returncode == 0


# ---------------------------------------------------------------------
# CLI validation
# ---------------------------------------------------------------------


def test_wrong_argv_length_prints_usage(tmp_path: Path) -> None:
    """argv != 3 → exit 2 with a usage line on stderr."""
    proc = subprocess.run(
        [sys.executable, str(COMPARE_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Usage:" in proc.stderr


def test_missing_baseline_file_raises(tmp_path: Path) -> None:
    """Missing files surface as FileNotFoundError (non-zero exit)."""
    missing = tmp_path / "nope.json"
    current = tmp_path / "c.json"
    _write_json(current, [])
    proc = _run(missing, current)
    assert proc.returncode != 0


def test_malformed_json_raises(tmp_path: Path) -> None:
    """Invalid JSON surfaces as JSONDecodeError (non-zero exit)."""
    bad = tmp_path / "b.json"
    bad.write_text("{not-json", encoding="utf-8")
    good = tmp_path / "c.json"
    _write_json(good, [])
    proc = _run(bad, good)
    assert proc.returncode != 0


@pytest.mark.parametrize("pct_regression", [21.0, 50.0, 100.0])
def test_regression_percentages_are_formatted(
    tmp_path: Path, pct_regression: float
) -> None:
    """Regression report includes the ``+NN.N%`` formatted delta."""
    base_val = 10.0
    cur_val = base_val * (1.0 + pct_regression / 100.0)
    baseline = [_entry(8, "euler", "rust", base_val)]
    current = [_entry(8, "euler", "rust", cur_val)]
    _write_json(tmp_path / "b.json", baseline)
    _write_json(tmp_path / "c.json", current)
    proc = _run(tmp_path / "b.json", tmp_path / "c.json")
    assert proc.returncode == 1
    assert f"+{pct_regression:.1f}%" in proc.stdout
