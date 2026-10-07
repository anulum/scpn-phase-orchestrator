# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Original-owner integration benchmark contracts

"""Exercise actual benchmark API and CLI outputs, refusal and reproducibility."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from benchmarks import attnres_modulation_benchmark as benchmark
from benchmarks.attnres_modulation_benchmark import BenchmarkRow, bench_one
from benchmarks.attnres_reference import FloatArray
from scpn_phase_orchestrator.coupling.attention_residuals import (
    AVAILABLE_BACKENDS,
    attnres_modulate,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("size", [1, 3])
def test_actual_owner_loops_have_paired_samples_and_independent_trajectory_errors(
    size: int,
) -> None:
    """Observe every available owner on paired isolated and connected graphs."""
    report = bench_one(size, n_steps=2, repeats=2)
    assert set(report["backends"]) == set(AVAILABLE_BACKENDS)
    for result in (report["baseline"], *report["backends"].values()):
        assert len(result["durations_seconds"]) == 2
        assert all(duration > 0.0 for duration in result["durations_seconds"])
        assert result["oracle_max_abs_error"] < 2e-11
        assert result["median_ms_per_step"] > 0.0
        assert len(result["final_phases"]) == size
    assert "_python_fallback" in report["backends"]["python"]["observed_calls"]


@pytest.mark.parametrize("field", ["n", "n_steps", "repeats"])
@pytest.mark.parametrize("invalid", [0, -1, True, 1.5])
def test_invalid_count_controls_fail_before_any_timing(
    field: str, invalid: object
) -> None:
    """Reject zero, negative, boolean and fractional benchmark counts."""
    with pytest.raises(ValueError, match=field):
        bench_one(
            cast(int, invalid) if field == "n" else 2,
            n_steps=cast(int, invalid) if field == "n_steps" else 1,
            repeats=cast(int, invalid) if field == "repeats" else 1,
            backends=("python",),
        )


@pytest.mark.parametrize("interval", [0.0, -0.1, np.inf, np.nan, True])
def test_invalid_time_intervals_fail_before_any_timing(interval: float) -> None:
    """Refuse non-positive, non-finite and boolean Euler time intervals."""
    with pytest.raises(ValueError, match="dt"):
        bench_one(2, dt=interval, n_steps=1, repeats=1, backends=("python",))


@pytest.mark.parametrize("owners", [(), ("unknown",), ("python", "python")])
def test_invalid_owner_sets_cannot_silently_omit_a_runtime(
    owners: tuple[str, ...],
) -> None:
    """Refuse empty, misspelled and duplicate comparison owner sets."""
    with pytest.raises(ValueError, match="backends"):
        bench_one(2, n_steps=1, repeats=1, backends=owners)


@pytest.mark.parametrize("entry", ["module", "script"])
@pytest.mark.parametrize("write_file", [True, False])
def test_real_cli_records_actual_source_hashes_and_paired_json_results(
    tmp_path: Path, entry: str, write_file: bool
) -> None:
    """Consume the real console JSON and optional file from each supported entry."""
    arguments = (
        ["-m", "benchmarks.attnres_modulation_benchmark"]
        if entry == "module"
        else [str(ROOT / "benchmarks/attnres_modulation_benchmark.py")]
    )
    output = tmp_path / "results.json"
    arguments += [
        "--sizes",
        "2",
        "--steps",
        "1",
        "--repeats",
        "2",
        "--backends",
        "python",
    ]
    if write_file:
        arguments += ["--output", str(output)]
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.run(
        [sys.executable, "-B", *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads(process.stdout)
    assert report["schema"] == "spo.phase-attention-diagnostics.v2"
    assert report["artifacts"] == []
    assert "no isolated speedup" in report["claim_boundary"]
    for source in report["sources"]:
        actual = Path(source["path"])
        assert source["sha256"] == hashlib.sha256(actual.read_bytes()).hexdigest()
    row = cast(BenchmarkRow, report["results"][0])
    assert row["n"] == 2 and row["n_steps"] == 1 and row["repetitions"] == 2
    assert list(row["backends"]) == ["python"]
    assert len(row["backends"]["python"]["durations_seconds"]) == 2
    if write_file:
        assert json.loads(output.read_text()) == report
    else:
        assert not output.exists()


def test_benchmark_refuses_numerically_correct_results_from_the_wrong_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real Python computation cannot qualify a requested compiled owner."""

    def route_to_python(
        coupling: FloatArray,
        phases: FloatArray,
        *,
        block_size: int,
        lambda_: float,
        backend: str,
    ) -> FloatArray:
        """Inject a wrong owner route while retaining original numerical computation."""
        return attnres_modulate(
            coupling, phases, block_size=block_size, lambda_=lambda_, backend="python"
        )

    monkeypatch.setattr(benchmark, "attnres_modulate", route_to_python)
    with pytest.raises(RuntimeError, match="named owner rust did not execute"):
        bench_one(3, n_steps=1, repeats=1, backends=("rust",))
