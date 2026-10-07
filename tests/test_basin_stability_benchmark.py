# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Original consumer benchmark contracts

"""Actual benchmark API/CLI outputs and rejected omission/oracle mismatches."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from benchmarks.basin_stability_benchmark import _measure, bench_at

ROOT = Path(__file__).resolve().parents[1]


def test_actual_benchmark_exercises_all_four_public_consumers() -> None:
    """An original Python owner produces raw timings and independent scalar values."""
    row = bench_at(2, 1, 2, 2, backends=("python",))
    assert row["N"] == 2 and row["calls"] == 2
    assert set(row["baseline"]) == {"trial", "sampling", "sweep", "search"}
    for name, timing in row["backends"]["python"].items():
        assert len(timing["durations_seconds"]) == 2
        assert all(value >= 0 for value in timing["durations_seconds"])
        assert timing["oracle_max_abs_error"] < 2e-14
        assert "_python_steady_state_r" in timing["observed_calls"]
        assert len(timing["values"]) == len(row["baseline"][name]["values"])


@pytest.mark.parametrize(
    ("n", "transient", "measure", "calls"),
    [(0, 0, 1, 1), (True, 0, 1, 1), (2, -1, 1, 1), (2, 0, 0, 1), (2, 0, 1, 0)],
)
def test_benchmark_refuses_invalid_work_counts(
    n: int, transient: int, measure: int, calls: int
) -> None:
    """Zero-work identities cannot be reported as measured runtime execution."""
    with pytest.raises(ValueError, match="must be an integer"):
        bench_at(n, transient, measure, calls, backends=("python",))


@pytest.mark.parametrize("owners", [(), ("unknown",), ("python", "python")])
def test_benchmark_refuses_ambiguous_or_unknown_owner_sets(
    owners: tuple[str, ...],
) -> None:
    """A required comparison cannot silently omit or duplicate a requested owner."""
    with pytest.raises(ValueError, match="distinct supported"):
        bench_at(2, 0, 1, 1, backends=owners)


def test_omitted_original_owner_is_a_rejected_negative_control() -> None:
    """An injected precomputed output is rejected even when numerically identical."""
    with pytest.raises(RuntimeError, match="owner not observed"):
        _measure(
            lambda: np.array([0.5]),
            np.array([0.5]),
            1,
            owner="python",
            consumer="trial",
        )


def test_actual_python_computation_cannot_be_credited_to_rust() -> None:
    """Reject a different original owner even when its actual number agrees."""
    from scpn_phase_orchestrator.upde.basin_stability import steady_state_r

    phases = np.array([0.0, np.pi / 2])
    graph = np.array([[0.0, 5e-31], [5e-31, 0.0]])

    def original_python_trial() -> NDArray[np.float64]:
        """Compute the original public tiny-edge law with explicit Python ownership."""
        return np.array(
            [
                steady_state_r(
                    phases,
                    np.zeros(2),
                    graph,
                    dt=1e30,
                    n_transient=0,
                    n_measure=1,
                    backend="python",
                )
            ]
        )

    expected = np.array([np.cos((np.pi / 2 - 1) / 2)])
    with pytest.raises(RuntimeError, match="original named rust owner not observed"):
        _measure(original_python_trial, expected, 1, owner="rust", consumer="trial")


def test_actual_partial_rust_exports_keep_original_trial_diagnostics() -> None:
    """A child measures real partial exports without replacing collected classes."""
    from scpn_phase_orchestrator.upde.basin_stability import AVAILABLE_BACKENDS

    if "rust" not in AVAILABLE_BACKENDS:
        with pytest.raises(ImportError, match="requested basin backend 'rust'"):
            bench_at(2, 0, 1, 1, backends=("rust",))
        return
    code = r"""
import json,spo_kernel
if hasattr(spo_kernel,'trace_sync_transition_rust'):
    del spo_kernel.trace_sync_transition_rust
from benchmarks.basin_stability_benchmark import bench_at
row=bench_at(2,0,1,1,backends=('rust',))
for consumer in('sweep','search'):
    observation=row['backends']['rust'][consumer]
    assert observation['oracle_max_abs_error']<2e-14
    assert any('spo_kernel.spo_kernel.steady_state_r_rust'in name
        for name in observation['observed_calls'])
    assert '_python_steady_state_r'not in observation['observed_calls']
print(json.dumps({'original_rust':True,'consumers':['sweep','search']}))
"""
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads(process.stdout.splitlines()[-1])
    assert report == {"original_rust": True, "consumers": ["sweep", "search"]}


def test_independent_oracle_mismatch_is_a_rejected_negative_control() -> None:
    """An intentionally incorrect oracle cannot pass numerical qualification."""
    with pytest.raises(AssertionError):
        _measure(
            lambda: np.array([0.5]), np.array([0.6]), 1, owner=None, consumer="trial"
        )


def test_corrupted_actual_sampling_record_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A damaged original result is an explicit negative output-contract control."""
    from benchmarks import basin_stability_benchmark as diagnostic
    from scpn_phase_orchestrator.upde.basin_stability import (
        BasinStabilityResult,
        basin_stability,
    )

    def damaged_record(
        omegas: NDArray[np.float64],
        graph: NDArray[np.float64],
        alpha: NDArray[np.float64],
        *,
        n_transient: int,
        n_measure: int,
        n_samples: int,
        R_threshold: float,
        seed: int,
        backend: str,
    ) -> BasinStabilityResult:
        """Compute first, then deliberately damage only the classification record."""
        result = basin_stability(
            omegas,
            graph,
            alpha,
            n_transient=n_transient,
            n_measure=n_measure,
            n_samples=n_samples,
            R_threshold=R_threshold,
            seed=seed,
            backend=backend,
        )
        result.n_converged += 1
        return result

    monkeypatch.setattr(diagnostic, "basin_stability", damaged_record)
    with pytest.raises(RuntimeError, match="sampling classification disagrees"):
        diagnostic.bench_at(2, 0, 1, 1, backends=("python",))


def test_changed_source_identity_prevents_cli_report_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An explicit damaged hash record refuses an otherwise actual CLI comparison."""
    from benchmarks import basin_stability_benchmark as diagnostic

    original = diagnostic._artifact
    seen: set[Path] = set()

    def damaged_identity(path: Path) -> dict[str, str]:
        """Read actual file bytes before injecting the negative metadata mismatch."""
        record = original(path)
        if path in seen:
            record["sha256"] = "0" * 64
        seen.add(path)
        return record

    output = tmp_path / "refused-comparison.json"
    monkeypatch.setattr(diagnostic, "_artifact", damaged_identity)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "basin_stability_benchmark.py",
            "--sizes",
            "2",
            "--calls",
            "1",
            "--backends",
            "python",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(RuntimeError, match="sources changed during measurement"):
        diagnostic.main()
    assert not output.exists()


def test_real_cli_records_sources_and_raw_samples(tmp_path: Path) -> None:
    """Run the actual command and inspect its strict JSON provenance and values."""
    output = tmp_path / "consumer-diagnostics.json"
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.run(
        [
            sys.executable,
            "-B",
            "benchmarks/basin_stability_benchmark.py",
            "--sizes",
            "2",
            "--n-transient",
            "0",
            "--n-measure",
            "1",
            "--calls",
            "1",
            "--backends",
            "python",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads(output.read_text())
    assert report["schema"] == "spo.basin-coupling-consumer-diagnostics.v1"
    assert report["started_unix"] <= report["finished_unix"]
    assert report["sources"] and all(
        len(item["sha256"]) == 64 for item in report["sources"]
    )
    assert report["results"][0]["backends"]["python"]["trial"]["values"]
    assert report["artifacts"] == []


@pytest.mark.parametrize(
    "failure", ("population", "matrix", "finite", "count", "weight", "angle", "step")
)
def test_independent_scalar_oracle_refuses_invalid_domains(failure: str) -> None:
    """The separate oracle refuses malformed inputs and unrepresentable arithmetic."""
    from benchmarks.kuramoto_trial_reference import scalar_trial

    phases = [0.0, 1.0]
    omega = [0.0, 0.0]
    graph = [[0.0, 0.3], [0.2, 0.0]]
    lag = [[0.0, 0.0], [0.0, 0.0]]
    scale = 1.0
    dt = 0.01
    transient = 0
    if failure == "population":
        phases = []
    elif failure == "matrix":
        graph[0] = [0.0]
    elif failure == "finite":
        omega[0] = float("nan")
    elif failure == "count":
        transient = -1
    elif failure == "weight":
        graph[0][1] = 1e308
        scale = 2.0
    elif failure == "angle":
        phases = [1e308, -1e308]
    else:
        omega = [1e308, 1e308]
        dt = 2.0
    with pytest.raises(ValueError):
        scalar_trial(phases, omega, graph, lag, scale=scale, dt=dt, transient=transient)


def test_oracle_empty_window_validates_without_integrating_unused_steps() -> None:
    """The independent zero-window convention distinguishes invalid counts."""
    from benchmarks.kuramoto_trial_reference import scalar_trial

    assert (
        scalar_trial([0.0], [1e308], [[0.0]], [[0.0]], dt=2.0, transient=100, measure=0)
        == 0.0
    )
    with pytest.raises(ValueError, match="step counts"):
        scalar_trial([0.0], [0.0], [[0.0]], [[0.0]], measure=-1)


def test_real_cli_coverage_exercises_finite_window_search_cases(tmp_path: Path) -> None:
    """Original CLI records subthreshold and bisection cases under instrumentation."""
    commands = [(97, 0, 1), (97, 2, 3), (2, 0, 1), (2, 0, 1), (2, 0, 1)]
    code = r"""
import sys,runpy,importlib.util
if importlib.util.find_spec('juliacall') is not None:
    from juliacall import Main
from coverage import Coverage
arguments=sys.argv[2:]
measurement=Coverage(data_file=sys.argv[1],data_suffix=False,branch=True,
    source=['benchmarks'])
measurement.start()
try:
    sys.argv=['benchmarks/basin_stability_benchmark.py',*arguments]
    runpy.run_path('benchmarks/basin_stability_benchmark.py',run_name='__main__')
finally:
    measurement.stop();measurement.save()
"""
    expected = [None, 17.5, 2.5, 2.5, 2.5]
    from benchmarks.kuramoto_trial_reference import scalar_trial

    phases = np.random.default_rng(23).uniform(0, 2 * np.pi, 97)
    omega = np.linspace(-0.4, 0.6, 97)
    graph = np.full((97, 97), 2 / 97)
    np.fill_diagonal(graph, 0.0)
    zero_lags = np.zeros((97, 97))
    readings = [
        scalar_trial(
            phases.tolist(),
            omega.tolist(),
            graph.tolist(),
            zero_lags.tolist(),
            scale=scale,
            dt=0.01,
            transient=2,
            measure=3,
        )
        for scale in (10.0, 15.0, 20.0)
    ]
    assert readings[0] < 0.1 and readings[1] < 0.1 <= readings[2]
    for index, (n, transient, measure) in enumerate(commands):
        output = tmp_path / f"cli-{index}.json"
        raw = tmp_path / f".coverage.cli-{index}"
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        args = [
            sys.executable,
            "-B",
            "-c",
            code,
            str(raw),
            "--sizes",
            str(n),
            "--n-transient",
            str(transient),
            "--n-measure",
            str(measure),
            "--calls",
            "1",
            "--backends",
            "python",
            "--output",
            str(output),
        ]
        if index == 3:
            # Discovery is recorded, while every returned owner must compute.
            option = args.index("--backends")
            del args[option : option + 2]
        if index == 4:
            option = args.index("--output")
            del args[option : option + 2]
        process = subprocess.run(
            args,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert process.returncode == 0, process.stdout + process.stderr
        report = json.loads(process.stdout if index == 4 else output.read_text())
        values = report["results"][0]["backends"]["python"]["search"]["values"]
        assert values == [expected[index]]
        assert raw.is_file()
