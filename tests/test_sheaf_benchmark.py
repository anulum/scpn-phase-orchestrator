# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Sheaf diagnostic contracts

"""Verify actual public and CLI sheaf diagnostic equations and repeat provenance."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from typing import cast

import pytest

from benchmarks.sheaf_benchmark import SheafMeasurement, run_sheaf_bench


@pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
def test_real_seeded_methods_match_independent_equation(method: str) -> None:
    """Real trajectories satisfy method accuracy and genuine backend metadata.

    The reference-failure guard stays visible: the bounded seeded smooth ODE
    succeeds in both actual SciPy environments. No genuine failed reference is
    available within these measured workloads; substituting solve_ivp would
    manufacture that branch. CLI tests exercise the same independent reference.
    """
    record = run_sheaf_bench(3, 2, 5, 2, method)
    again = run_sheaf_bench(3, 2, 5, 1, method)
    assert record["input_sha256"] == again["input_sha256"]
    assert record["max_phase_error"] < (0.003 if method == "euler" else 1e-9)
    assert record["max_phase_error"] == pytest.approx(
        again["max_phase_error"], abs=1e-15
    )
    assert len(record["seconds"]) == 2 and all(value > 0 for value in record["seconds"])
    assert record["kernel_available"] == (
        importlib.util.find_spec("spo_kernel") is not None
    )
    assert 0.0 < record["last_dt"] <= record["dt"]
    assert "no production latency claim" in record["diagnostic_scope"]


@pytest.mark.parametrize("field", ["n", "d", "n_steps", "repeats"])
@pytest.mark.parametrize("value", [True, 0, -1, 1.5])
def test_diagnostic_counts_refuse_without_running(field: str, value: object) -> None:
    """Malformed workload controls refuse through the public benchmark function."""
    counts: dict[str, int] = {"n": 2, "d": 2, "n_steps": 2, "repeats": 1}
    counts[field] = cast(
        int, value
    )  # Preserve the deliberately malformed runtime object.
    with pytest.raises(ValueError, match=field):
        run_sheaf_bench(counts["n"], counts["d"], counts["n_steps"], counts["repeats"])


def test_diagnostic_unknown_method_refuses() -> None:
    """Unsupported integration methods cannot produce a timing record."""
    with pytest.raises(ValueError, match="method"):
        run_sheaf_bench(2, 2, 2, 1, "heun")


@pytest.mark.parametrize("method", [None, "rk45"])
def test_real_cli_emits_current_trajectory_evidence(method: str | None) -> None:
    """The actual module CLI emits reproducible equation residuals and raw repeats."""
    argv = [
        sys.executable,
        "-m",
        "benchmarks.sheaf_benchmark",
        "--n",
        "3",
        "--d",
        "2",
        "--steps",
        "5",
        "--repeats",
        "2",
    ]
    if method is not None:
        argv += ["--method", method]
    completed = subprocess.run(
        argv,
        env=os.environ.copy(),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    records = cast(list[SheafMeasurement], json.loads(completed.stdout))
    assert [record["method"] for record in records] == (
        [method] if method else ["euler", "rk4", "rk45"]
    )
    for record in records:
        reference = run_sheaf_bench(3, 2, 5, 1, record["method"])
        assert record["input_sha256"] == reference["input_sha256"]
        assert record["max_phase_error"] == pytest.approx(
            reference["max_phase_error"], abs=1e-15
        )
        assert len(record["seconds"]) == 2 and all(
            value > 0 for value in record["seconds"]
        )
        assert record["kernel_available"] == reference["kernel_available"]
