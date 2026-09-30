# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Sparse benchmark equation and CLI contracts

"""Verify real sparse benchmark trajectories, reproducibility and CLI workloads."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from benchmarks.sparse_benchmark import SparseMeasurement, run_sparse_bench


@pytest.mark.parametrize("n,density", [(1, 0.0), (8, 0.25), (4, 1.0)])
def test_real_benchmark_preserves_seeded_equation(n: int, density: float) -> None:
    """Repeat real Euler calls with identical inputs and bounded equation error."""
    first = run_sparse_bench(n, density, n_steps=4, repeats=2)
    second = run_sparse_bench(n, density, n_steps=4, repeats=1)
    assert first["input_sha256"] == second["input_sha256"]
    assert first["order_parameter"] == second["order_parameter"]
    assert first["edges"] == second["edges"]
    assert first["steps"] == 4
    assert first["repeats"] == 2
    assert len(first["seconds"]) == 2
    assert all(sample > 0.0 for sample in first["seconds"])
    assert first["median_step_us"] == float(np.median(first["seconds"])) * 1e6 / 4
    assert first["max_phase_error"] <= 1e-11
    assert 0.0 <= first["order_parameter"] <= 1.0 + 1e-15
    assert first["kernel_available"] == (
        importlib.util.find_spec("spo_kernel") is not None
    )
    if density == 0.0:
        assert first["edges"] == 0
        assert first["order_parameter"] == pytest.approx(1.0, abs=1e-15)
    else:
        assert 0 < first["edges"] <= n * n


@pytest.mark.parametrize("n", [0, -1, True])
def test_invalid_node_count_refuses(n: int) -> None:
    """Refuse nonpositive and boolean node counts before generating inputs."""
    with pytest.raises(ValueError, match="n must be positive"):
        run_sparse_bench(n=n)


@pytest.mark.parametrize("density", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_density_refuses(density: float) -> None:
    """Refuse densities that cannot describe a finite sparse workload."""
    with pytest.raises(ValueError, match="density must be finite"):
        run_sparse_bench(density=density)


@pytest.mark.parametrize("steps", [0, -1, True])
def test_invalid_step_count_refuses(steps: int) -> None:
    """Refuse workloads without a positive integral timing denominator."""
    with pytest.raises(ValueError, match="n_steps must be positive"):
        run_sparse_bench(n_steps=steps)


@pytest.mark.parametrize("repeats", [0, -1, True])
def test_invalid_repeat_count_refuses(repeats: int) -> None:
    """Refuse repetition counts that cannot produce timing samples."""
    with pytest.raises(ValueError, match="repeats must be positive"):
        run_sparse_bench(repeats=repeats)


@pytest.mark.parametrize("custom", [False, True])
def test_real_cli_reports_default_and_requested_workloads(custom: bool) -> None:
    """Execute the actual CLI, including its original 1,000/10,000-node cases."""
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        "-m",
        "benchmarks.sparse_benchmark",
        "--steps",
        "1",
        "--repeats",
        "1",
    ]
    if custom:
        command += ["--n", "8", "--density", "0.25"]
    completed = subprocess.run(
        command,
        cwd=root,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    measurements = cast(list[SparseMeasurement], json.loads(completed.stdout))
    assert [(row["n"], row["density"]) for row in measurements] == (
        [(8, 0.25)] if custom else [(1000, 0.01), (10000, 0.001)]
    )
    for row in measurements:
        assert row["steps"] == row["repeats"] == 1
        assert row["max_phase_error"] <= 1e-11
        assert len(row["input_sha256"]) == 64
        assert row["seconds"][0] > 0.0
        assert row["python"] and row["numpy"] and row["scipy"]
