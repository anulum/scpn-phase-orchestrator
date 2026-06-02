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
* Baseline entries keyed by (n_osc, method, backend); missing baseline keys in
  ``current`` fail by default, with an explicit partial-run flag for local use.

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
    baseline_path: Path,
    current_path: Path,
    *extra_args: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(COMPARE_SCRIPT),
            str(baseline_path),
            str(current_path),
            *extra_args,
        ],
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


def test_small_absolute_ci_noise_is_pass(tmp_path: Path) -> None:
    """Large percentages on tiny timings stay below the absolute floor.

    CI runners can add tens of microseconds of fixed overhead to very small
    oscillator cases. The benchmark gate should report the movement but not
    fail unless the slowdown is also materially large in absolute terms.
    """
    baseline = [_entry(8, "euler", "rust", 7.5)]
    current = [_entry(8, "euler", "rust", 60.8)]
    _write_json(tmp_path / "b.json", baseline)
    _write_json(tmp_path / "c.json", current)
    proc = _run(tmp_path / "b.json", tmp_path / "c.json")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "tolerated" in proc.stdout
    assert "+53.3 us" in proc.stdout


# ---------------------------------------------------------------------
# FAIL cases
# ---------------------------------------------------------------------


def test_regression_above_threshold_fails(tmp_path: Path) -> None:
    """>20% and materially slower triggers exit 1 and is listed."""
    baseline = [_entry(32, "euler", "rust", 100.0)]
    current = [_entry(32, "euler", "rust", 250.0)]  # +150%, +150 us
    _write_json(tmp_path / "b.json", baseline)
    _write_json(tmp_path / "c.json", current)
    proc = _run(tmp_path / "b.json", tmp_path / "c.json")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "FAIL" in proc.stdout
    assert "+150.0%" in proc.stdout
    assert "+150.0 us" in proc.stdout
    assert "1 regression(s)" in proc.stdout


def test_multiple_regressions_all_reported(tmp_path: Path) -> None:
    """All failing configs appear in the summary; count matches."""
    baseline = [
        _entry(8, "euler", "rust", 10.0),
        _entry(16, "rk4", "rust", 20.0),
        _entry(32, "rk45", "rust", 40.0),
    ]
    current = [
        _entry(8, "euler", "rust", 150.0),  # +1400%, +140 us
        _entry(16, "rk4", "rust", 20.0),  # 0%
        _entry(32, "rk45", "rust", 170.0),  # +325%, +130 us
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


def test_category_form_baseline_accepted(tmp_path: Path) -> None:
    """Checked-in baselines may group benchmark entries by science kernel."""
    baseline = {
        "meta": {"generated": "fixture"},
        "upde": [_entry(8, "euler", "rust", 10.0)],
        "stuart_landau": [
            {
                "n_osc": 8,
                "method": "euler",
                "steps": 500,
                "us_per_step": 27.7,
            }
        ],
    }
    current = {"results": [_entry(8, "euler", "rust", 11.0)]}
    _write_json(tmp_path / "b.json", baseline)
    _write_json(tmp_path / "c.json", current)
    proc = _run(tmp_path / "b.json", tmp_path / "c.json")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "within regression threshold" in proc.stdout


def test_missing_current_baseline_key_fails_closed(tmp_path: Path) -> None:
    """Current runs missing a checked-in baseline key fail closed."""
    baseline = [_entry(8, "euler", "rust", 10.0)]
    current = [_entry(64, "rk4", "python", 100.0)]  # completely different key
    _write_json(tmp_path / "b.json", baseline)
    _write_json(tmp_path / "c.json", current)
    proc = _run(tmp_path / "b.json", tmp_path / "c.json")
    assert proc.returncode == 1
    assert "missing baseline configurations" in proc.stdout


def test_missing_current_key_allowed_only_with_partial_run_flag(tmp_path: Path) -> None:
    """Deliberate local partial runs must opt in explicitly."""
    baseline = [_entry(8, "euler", "rust", 10.0)]
    current = [_entry(64, "rk4", "python", 100.0)]
    _write_json(tmp_path / "b.json", baseline)
    _write_json(tmp_path / "c.json", current)
    proc = _run(
        tmp_path / "b.json",
        tmp_path / "c.json",
        "--allow-missing-current",
    )
    assert proc.returncode == 1
    assert "No overlapping benchmark entries found" in proc.stdout


def test_malformed_benchmark_records_are_ignored(tmp_path: Path) -> None:
    """Non-comparable records do not crash the guard, but no baseline
    comparisons still fail closed."""
    baseline = {
        "meta": {"generated": "fixture"},
        "results": [
            {"method": "euler", "backend": "rust", "us_per_step": 10.0},
            "not-a-record",
        ],
    }
    current = {"results": [_entry(8, "euler", "rust", 11.0)]}
    _write_json(tmp_path / "b.json", baseline)
    _write_json(tmp_path / "c.json", current)
    proc = _run(tmp_path / "b.json", tmp_path / "c.json")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "No comparable baseline benchmark entries found" in proc.stdout


def test_zero_baseline_fails_closed(tmp_path: Path) -> None:
    """Zero baselines are invalid because they hide division-by-zero and
    remove the regression budget."""
    baseline = [_entry(8, "euler", "rust", 0.0)]
    current = [_entry(8, "euler", "rust", 5.0)]
    _write_json(tmp_path / "b.json", baseline)
    _write_json(tmp_path / "c.json", current)
    proc = _run(tmp_path / "b.json", tmp_path / "c.json")
    assert proc.returncode != 0
    assert "finite and > 0" in proc.stderr


def test_duplicate_baseline_key_fails_closed(tmp_path: Path) -> None:
    """Duplicate baseline keys must not silently overwrite each other."""
    baseline = [
        _entry(8, "euler", "rust", 10.0),
        _entry(8, "euler", "rust", 11.0),
    ]
    current = [_entry(8, "euler", "rust", 10.0)]
    _write_json(tmp_path / "b.json", baseline)
    _write_json(tmp_path / "c.json", current)
    proc = _run(tmp_path / "b.json", tmp_path / "c.json")
    assert proc.returncode == 1
    assert "Duplicate baseline benchmark entry" in proc.stdout


def test_duplicate_current_key_fails_closed(tmp_path: Path) -> None:
    """Duplicate current keys must not double-count or mask a regression."""
    baseline = [_entry(8, "euler", "rust", 10.0)]
    current = [
        _entry(8, "euler", "rust", 10.0),
        _entry(8, "euler", "rust", 12.0),
    ]
    _write_json(tmp_path / "b.json", baseline)
    _write_json(tmp_path / "c.json", current)
    proc = _run(tmp_path / "b.json", tmp_path / "c.json")
    assert proc.returncode == 1
    assert "Duplicate current benchmark entry" in proc.stdout


def test_duplicate_json_object_key_fails_closed(tmp_path: Path) -> None:
    """JSON duplicate keys are rejected before benchmark extraction."""
    baseline = tmp_path / "b.json"
    baseline.write_text(
        '{"results": [{"n_osc": 8, "method": "euler", "backend": "rust", '
        '"us_per_step": 10.0}], "results": []}',
        encoding="utf-8",
    )
    current = tmp_path / "c.json"
    _write_json(current, [_entry(8, "euler", "rust", 10.0)])
    proc = _run(baseline, current)
    assert proc.returncode == 1
    assert "duplicate key" in proc.stderr


def test_non_finite_json_token_fails_closed(tmp_path: Path) -> None:
    """Benchmark evidence JSON must not contain NaN/Infinity tokens."""
    baseline = tmp_path / "b.json"
    baseline.write_text(
        '[{"n_osc": 8, "method": "euler", "backend": "rust", "us_per_step": NaN}]',
        encoding="utf-8",
    )
    current = tmp_path / "c.json"
    _write_json(current, [_entry(8, "euler", "rust", 10.0)])
    proc = _run(baseline, current)
    assert proc.returncode == 1
    assert "non-finite token" in proc.stderr


@pytest.mark.parametrize(
    "bad_entry, expected",
    [
        (
            {"n_osc": True, "method": "euler", "backend": "rust", "us_per_step": 10.0},
            "positive integer",
        ),
        (
            {"n_osc": 8.0, "method": "euler", "backend": "rust", "us_per_step": 10.0},
            "positive integer",
        ),
        (
            {"n_osc": 8, "method": "", "backend": "rust", "us_per_step": 10.0},
            "non-empty string",
        ),
        (
            {"n_osc": 8, "method": "euler", "backend": "", "us_per_step": 10.0},
            "non-empty string",
        ),
    ],
)
def test_invalid_benchmark_identity_fields_fail_closed(
    tmp_path: Path,
    bad_entry: dict[str, object],
    expected: str,
) -> None:
    """Benchmark identity fields are part of the regression contract."""
    baseline = [bad_entry]
    current = [_entry(8, "euler", "rust", 10.0)]
    _write_json(tmp_path / "b.json", baseline)
    _write_json(tmp_path / "c.json", current)
    proc = _run(tmp_path / "b.json", tmp_path / "c.json")
    assert proc.returncode == 1
    assert expected in proc.stderr


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
    assert "usage:" in proc.stderr


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
    base_val = 1000.0
    cur_val = base_val * (1.0 + pct_regression / 100.0)
    baseline = [_entry(8, "euler", "rust", base_val)]
    current = [_entry(8, "euler", "rust", cur_val)]
    _write_json(tmp_path / "b.json", baseline)
    _write_json(tmp_path / "c.json", current)
    proc = _run(tmp_path / "b.json", tmp_path / "c.json")
    assert proc.returncode == 1
    assert f"+{pct_regression:.1f}%" in proc.stdout
