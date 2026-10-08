# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Actual ethical-cost comparison controls

"""Exercise genuine installed benchmark workers, arithmetic and CLI failures."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from benchmarks.ethical_cost_benchmark import benchmark_size, measure_current
from benchmarks.ethical_cost_reference import reference_cost
from scpn_phase_orchestrator.ssgf import ethical


@pytest.mark.parametrize(
    "controls",
    [
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
        (-1, 1, 1),
        (True, 1, 1),
        (1, False, 1),
        (1, 1, True),
        (1.5, 1, 1),
    ],
)
def test_invalid_benchmark_counts_are_refused(
    controls: tuple[object, object, object],
) -> None:
    """Actual control validation rejects aliases and nonpositive work counts."""
    n, calls, batches = (cast("int", v) for v in controls)
    with pytest.raises(ValueError, match="positive"):
        measure_current(n, calls, batches, "rust")


@pytest.mark.parametrize("owner", ["", "Python", "invalid"])
def test_unknown_benchmark_owner_is_refused(owner: str) -> None:
    """An unsupported requested owner cannot produce a timing record."""
    with pytest.raises(ValueError, match="valid owner"):
        measure_current(3, 1, 1, owner)


def test_comparison_refuses_actual_non_python_executable_before_execution() -> None:
    """A real Git binary cannot impersonate a selected Python runtime."""
    raw = shutil.which("git")
    assert raw is not None
    with pytest.raises(ValueError, match="trusted running Python"):
        benchmark_size(
            3, 1, 1, python_profile=Path(raw), rust_profile=Path(sys.executable)
        )


def _profile(owner: str) -> Path:
    """Resolve an explicitly qualified installed runtime without modifying it."""
    raw = os.environ.get("SPO_ETHICAL_" + owner.upper() + "_PROFILE")
    assert raw is not None, "Qualified original installed profiles are required"
    result = Path(raw)
    assert result.is_file()
    return result


@pytest.mark.native_runtime
def test_comparison_measures_two_actual_owners_and_independent_fixture() -> None:
    """Same-source workers report real samples and independent numerical parity."""
    row = benchmark_size(
        3, 3, 2, python_profile=_profile("python"), rust_profile=_profile("rust")
    )
    rng = np.random.default_rng(42)
    phases = rng.uniform(0.0, 2.0 * np.pi, 3)
    matrix = rng.uniform(0.0, 0.5, (3, 3))
    np.fill_diagonal(matrix, 0.0)
    expected = reference_cost(phases, matrix)
    assert row["parity_passed"] is True
    assert cast("float", row["python_over_rust"]) > 0.0
    for owner in ("python", "rust"):
        observed = cast("dict[str, object]", row[owner])
        assert observed["owner"] == owner
        assert observed["native_calls"] == int(owner == "rust")
        np.testing.assert_allclose(
            cast("list[float]", observed["cost"]), expected[:3], rtol=1e-10, atol=1e-10
        )
        assert observed["violations"] == expected[3]
        means = cast("list[float]", observed["batch_mean_us"])
        timings = cast("list[float]", observed["latency_us"])
        assert len(means) == 2 and len(timings) == 6
        assert all(value > 0.0 for value in means + timings)
        for percentile in (50, 95, 99):
            assert observed[f"p{percentile}_us"] == float(
                np.percentile(timings, percentile)
            )
        assert (
            observed["source_sha256"]
            == hashlib.sha256(Path(ethical.__file__).read_bytes()).hexdigest()
        )


@pytest.mark.native_runtime
def test_actual_comparison_cli_emits_only_completed_profile_records() -> None:
    """The real module CLI completes both installed workers before printing data."""
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "benchmarks.ethical_cost_benchmark",
            "--sizes",
            "3",
            "--calls",
            "2",
            "--batches",
            "2",
            "--python-profile",
            str(_profile("python")),
            "--rust-profile",
            str(_profile("rust")),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    observed = cast("dict[str, object]", json.loads(result.stdout))
    rows = cast("list[dict[str, object]]", observed["results"])
    assert len(rows) == 1 and rows[0]["parity_passed"] is True
    assert rows[0]["n"] == 3


@pytest.mark.native_runtime
def test_kernel_absent_worker_refuses_required_rust_timing() -> None:
    """An actual absent native installation returns failure and no timing JSON."""
    script = (
        Path(__file__).resolve().parents[1] / "benchmarks/ethical_cost_benchmark.py"
    )
    result = subprocess.run(
        [
            str(_profile("python")),
            "-I",
            "-B",
            str(script),
            "--worker",
            "rust",
            "--sizes",
            "3",
            "--calls",
            "1",
            "--batches",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode != 0 and not result.stdout.strip()
    assert "original compiled ethical-cost owner is required" in result.stderr


@pytest.mark.native_runtime
def test_native_installed_worker_cannot_claim_absent_python_timing() -> None:
    """An actually present native installation refuses a Python-only label."""
    script = (
        Path(__file__).resolve().parents[1] / "benchmarks/ethical_cost_benchmark.py"
    )
    result = subprocess.run(
        [
            str(_profile("rust")),
            "-I",
            "-B",
            str(script),
            "--worker",
            "python",
            "--sizes",
            "3",
            "--calls",
            "1",
            "--batches",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode != 0 and not result.stdout.strip()
    assert "genuinely kernel-absent install" in result.stderr
