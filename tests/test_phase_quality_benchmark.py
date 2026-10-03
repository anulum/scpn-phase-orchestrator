# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Real phase quality measurement CLI tests
"""Exercise the numerical benchmark API and its actual child-process entry point."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from benchmarks.phase_quality_benchmark import main, measure_quality


@pytest.mark.parametrize("n", [1, 10, 100])
def test_measurement_uses_real_available_operations(n: int) -> None:
    """Measurements contain checked real results and finite elapsed samples.

    Parameters
    ----------
    n : int
        Number of public extraction records to score.
    """
    report = measure_quality(n, calls=2, repeats=2)
    operations = cast("dict[str, dict[str, object]]", report["operations"])
    expected_names = {
        "public_score",
        "public_score_huge",
        "public_mask",
        "public_override_mask",
    }
    available = importlib.util.find_spec("spo_kernel") is not None
    if available:
        expected_names |= {"native_score", "native_score_huge", "native_mask"}
    assert set(operations) == expected_names
    assert report["public_backend"] == ("native" if available else "python")
    assert (report["native_binary"] is not None) is available
    for operation in operations.values():
        samples = cast("list[float]", operation["seconds_per_call"])
        assert len(samples) == 2
        assert all(math.isfinite(x) and x >= 0.0 for x in samples)
    normal = operations["public_score"]["checked_result"]
    assert operations["public_score_huge"]["checked_result"] == pytest.approx(normal)


@pytest.mark.parametrize("argument", ["n", "calls", "repeats"])
@pytest.mark.parametrize("invalid", [0, -1, True, 1.5])
def test_invalid_controls_refuse_before_real_recovery(
    argument: str, invalid: object
) -> None:
    """Invalid measurement controls are refused without poisoning valid calls.

    Parameters
    ----------
    argument : str
        Public measurement control to violate.
    invalid : object
        Value outside the positive non-boolean integer contract.
    """
    controls = {"n": 10, "calls": 1, "repeats": 1}
    controls[argument] = cast("int", invalid)
    with pytest.raises(ValueError, match="positive non-boolean integers"):
        measure_quality(**controls)
    assert measure_quality(10, calls=1, repeats=1)["n"] == 10


def test_public_cli_function_emits_checked_strict_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI argument parsing reaches real measurements and serialisable output.

    Parameters
    ----------
    capsys : pytest.CaptureFixture[str]
        Pytest capture of the producer's actual standard output.
    """
    assert main(["--sizes", "10", "--calls", "1", "--repeats", "1"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report[0]["n"] == 10
    assert report[0]["operations"]["public_score_huge"][
        "checked_result"
    ] == pytest.approx(0.45)


@pytest.mark.parametrize("child_instrumentation", [False, True])
def test_real_cli_child_returns_finite_observations_and_refuses_bad_controls(
    child_instrumentation: bool, tmp_path: Path
) -> None:
    """Actual module processes check numerical outputs and propagate refusal.

    Parameters
    ----------
    child_instrumentation : bool
        Launch the real child directly or with the real coverage CLI.
    tmp_path : Path
        Isolated child data destination when the caller supplied no data path.

    Notes
    -----
    Both actual process modes run in each interpreter; coverage children retain
    parallel files for the owning union. Neither a coverage object nor a kernel
    is replaced. Child exit codes remain authoritative.
    """
    argv = [sys.executable, "-m"]
    if child_instrumentation:
        argv += [
            "coverage",
            "run",
            "--parallel-mode",
            "--branch",
            "--data-file="
            + os.environ.get("COVERAGE_FILE", str(tmp_path / "child.coverage")),
            "-m",
        ]
    argv += [
        "benchmarks.phase_quality_benchmark",
        "--sizes",
        "10",
        "--calls",
        "2",
        "--repeats",
        "2",
    ]
    result = subprocess.run(
        argv, check=True, capture_output=True, text=True, timeout=30
    )
    report = json.loads(result.stdout)
    assert len(report) == 1
    assert report[0]["operations"]["public_score_huge"][
        "checked_result"
    ] == pytest.approx(0.45)
    assert report[0]["calls"] == report[0]["repeats"] == 2
    refusal = subprocess.run(
        [sys.executable, "-m", "benchmarks.phase_quality_benchmark", "--calls", "0"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert refusal.returncode != 0
    assert "positive non-boolean integers" in refusal.stderr
