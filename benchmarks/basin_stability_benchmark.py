# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Actual finite-horizon consumer diagnostics

"""Paired public trial, sampling, sweep and search timings for original owners.

Every named owner must execute and match an independent scalar Euler oracle.
Shared-host timings are diagnostics, not production-budget or speedup acceptance.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import importlib
import json
import math
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
from numpy.typing import NDArray

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.kuramoto_trial_reference import scalar_trial
from scpn_phase_orchestrator.upde.basin_stability import (
    AVAILABLE_BACKENDS,
    basin_stability,
    steady_state_r,
)
from scpn_phase_orchestrator.upde.bifurcation import (
    find_critical_coupling,
    trace_sync_transition,
)

FloatArray = NDArray[np.float64]
Operation = Callable[[], FloatArray]


class Timing(TypedDict):
    """Raw wall times, actual observed calls, values and independent errors."""

    durations_seconds: list[float]
    median_ms_per_call: float
    observed_calls: list[str]
    oracle_max_abs_error: float
    values: list[float | None]


class BenchmarkRow(TypedDict):
    """Identical inputs and four complete consumer operations for every owner."""

    N: int
    n_transient: int
    n_measure: int
    calls: int
    available: list[str]
    baseline: dict[str, Timing]
    backends: dict[str, dict[str, Timing]]


def _all_to_all(n: int, strength: float = 2.0) -> FloatArray:
    """Return the explicitly normalized off-diagonal trial graph."""
    matrix = np.full((n, n), strength / n)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def _reference(
    n: int,
    omegas: FloatArray,
    graph: FloatArray,
    alpha: FloatArray,
    transient: int,
    measure: int,
) -> dict[str, Operation]:
    """Construct independent scalar laws and the documented threshold algorithms."""
    phases = np.random.default_rng(11).uniform(0, 2 * np.pi, n).tolist()
    frequencies = omegas.tolist()
    coupling = graph.tolist()
    lag = alpha.tolist()

    def trial() -> FloatArray:
        """Evaluate the independent post-step scalar Euler window."""
        return np.array(
            [
                scalar_trial(
                    phases,
                    frequencies,
                    coupling,
                    lag,
                    dt=0.01,
                    transient=transient,
                    measure=measure,
                )
            ]
        )

    def sample() -> FloatArray:
        """Replay the public NumPy seed while independently integrating both trials."""
        rng = np.random.default_rng(19)
        return np.array(
            [
                scalar_trial(
                    rng.uniform(0, 2 * np.pi, n).tolist(),
                    frequencies,
                    coupling,
                    lag,
                    dt=0.01,
                    transient=transient,
                    measure=measure,
                )
                for _ in range(2)
            ]
        )

    initial = np.random.default_rng(23).uniform(0, 2 * np.pi, n).tolist()

    def sweep() -> FloatArray:
        """Start each grid trajectory from the same independent seeded phases."""
        return np.array(
            [
                scalar_trial(
                    initial,
                    frequencies,
                    coupling,
                    lag,
                    scale=scale,
                    dt=0.01,
                    transient=transient,
                    measure=measure,
                )
                for scale in (0.0, 1.0, 2.0)
            ]
        )

    def search() -> FloatArray:
        """Apply the specified binary search using only scalar oracle measurements."""
        zero_lags = np.zeros((n, n)).tolist()

        def read(scale: float) -> float:
            """Evaluate a single independent zero-lag search trajectory."""
            return scalar_trial(
                initial,
                frequencies,
                coupling,
                zero_lags,
                scale=scale,
                dt=0.01,
                transient=transient,
                measure=measure,
            )

        low = 0.0
        high = 20.0
        if read(high) < 0.1:
            return np.array([float("nan")])
        # The benchmark fixes tol=6; halving [0,20] needs two iterations.
        while high - low >= 6.0:
            middle = (low + high) / 2
            if read(middle) < 0.1:
                low = middle
            else:
                high = middle
        return np.array([(low + high) / 2])

    return {"trial": trial, "sampling": sample, "sweep": sweep, "search": search}


def _operations(
    owner: str,
    n: int,
    omegas: FloatArray,
    graph: FloatArray,
    alpha: FloatArray,
    transient: int,
    measure: int,
) -> dict[str, Operation]:
    """Construct actual public calls with explicit numerical ownership."""
    phases = np.random.default_rng(11).uniform(0, 2 * np.pi, n)

    def trial() -> FloatArray:
        """Execute the selected original deterministic primitive."""
        return np.array(
            [
                steady_state_r(
                    phases,
                    omegas,
                    graph,
                    alpha,
                    n_transient=transient,
                    n_measure=measure,
                    backend=owner,
                )
            ]
        )

    def sample() -> FloatArray:
        """Execute public paired sampling and verify inclusive classification."""
        result = basin_stability(
            omegas,
            graph,
            alpha,
            n_transient=transient,
            n_measure=measure,
            n_samples=2,
            R_threshold=0.5,
            seed=19,
            backend=owner,
        )
        count = int(np.count_nonzero(result.R_final >= 0.5))
        if result.n_converged != count or count / 2 != result.S_B:
            raise RuntimeError(
                "public sampling classification disagrees with its trials"
            )
        return result.R_final

    def sweep() -> FloatArray:
        """Execute the selected batched or delegated independent coupling grid."""
        result = trace_sync_transition(
            omegas,
            graph,
            alpha,
            K_range=(0.0, 2.0),
            n_points=3,
            n_transient=transient,
            n_measure=measure,
            seed=23,
            backend=owner,
        )
        np.testing.assert_array_equal(result.K_values, [0.0, 1.0, 2.0])
        return result.R_values

    def search() -> FloatArray:
        """Execute the selected original zero-lag critical-threshold search."""
        return np.array(
            [
                find_critical_coupling(
                    omegas,
                    graph,
                    n_transient=transient,
                    n_measure=measure,
                    tol=6.0,
                    seed=23,
                    backend=owner,
                )
            ]
        )

    return {"trial": trial, "sampling": sample, "sweep": sweep, "search": search}


def _measure(
    operation: Operation,
    expected: FloatArray,
    repetitions: int,
    *,
    owner: str | None,
    consumer: str,
) -> Timing:
    """Observe the original owner and numerically qualify every timed repetition."""
    with cProfile.Profile() as profile:
        actual = operation()
    profile.create_stats()
    calls = sorted({key[2] for key in profile.stats})
    if owner is not None:
        needle = {
            "python": "_python_steady_state_r",
            "go": "steady_state_r_go",
            "julia": "steady_state_r_julia",
            "mojo": "steady_state_r_mojo",
            "rust": "<built-in method spo_kernel.spo_kernel.steady_state_r_rust>",
        }[owner]
        if owner == "rust" and consumer in ("sweep", "search"):
            native = importlib.import_module("spo_kernel")
            if all(
                callable(getattr(native, name, None))
                for name in (
                    "trace_sync_transition_rust",
                    "find_critical_coupling_bif_rust",
                )
            ):
                needle = (
                    "<built-in method spo_kernel.spo_kernel."
                    + (
                        "trace_sync_transition_rust"
                        if consumer == "sweep"
                        else "find_critical_coupling_bif_rust"
                    )
                    + ">"
                )
        if needle not in calls or (
            owner != "python" and "_python_steady_state_r" in calls
        ):
            raise RuntimeError(
                f"original named {owner} owner not observed for {consumer}: {calls}"
            )
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12, equal_nan=True)
    durations = []
    for _ in range(repetitions):
        started = time.perf_counter()
        actual = operation()
        durations.append(time.perf_counter() - started)
        np.testing.assert_allclose(
            actual, expected, rtol=2e-12, atol=2e-12, equal_nan=True
        )
    finite = np.isfinite(expected)
    error = (
        float(np.max(np.abs(actual[finite] - expected[finite])))
        if np.any(finite)
        else 0.0
    )
    return {
        "durations_seconds": durations,
        "median_ms_per_call": 1000 * statistics.median(durations),
        "observed_calls": calls,
        "oracle_max_abs_error": error,
        "values": [float(value) if math.isfinite(value) else None for value in actual],
    }


def bench_at(
    n: int,
    n_transient: int,
    n_measure: int,
    calls: int,
    *,
    backends: tuple[str, ...] | None = None,
) -> BenchmarkRow:
    """Compare all four consumers on identical independently verified inputs.

    Parameters
    ----------
    n : int
        Positive oscillator count.
    n_transient, n_measure : int
        Nonnegative transient and positive post-step measurement counts.
    calls : int
        Positive paired repetition count, defaulted to twenty by the CLI.
    backends : tuple of str or None
        Required numerical owners, or the actual discovered owner set.

    Returns
    -------
    BenchmarkRow
        Raw timings, source-observed owner calls and independent numerical errors.

    Raises
    ------
    ValueError
        If counts or owner names are invalid.
    ImportError
        If a required original owner is unavailable.
    RuntimeError
        If an owner was not observed or sampling classification is inconsistent.
    AssertionError
        If actual consumer output differs from the independent oracle.
    """
    for name, value, minimum in (
        ("n", n, 1),
        ("n_transient", n_transient, 0),
        ("n_measure", n_measure, 1),
        ("calls", calls, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    owners = tuple(AVAILABLE_BACKENDS) if backends is None else backends
    if (
        not owners
        or len(set(owners)) != len(owners)
        or any(
            owner not in ("python", "rust", "go", "julia", "mojo") for owner in owners
        )
    ):
        raise ValueError("backends must be distinct supported owner names")
    omegas = np.linspace(-0.4, 0.6, n)
    graph = _all_to_all(n)
    alpha = np.full((n, n), 0.07)
    np.fill_diagonal(alpha, 0)
    reference = _reference(n, omegas, graph, alpha, n_transient, n_measure)
    expected = {name: operation() for name, operation in reference.items()}
    baseline = {
        name: _measure(operation, expected[name], calls, owner=None, consumer=name)
        for name, operation in reference.items()
    }
    measured = {}
    for owner in owners:
        operations = _operations(owner, n, omegas, graph, alpha, n_transient, n_measure)
        measured[owner] = {
            name: _measure(operation, expected[name], calls, owner=owner, consumer=name)
            for name, operation in operations.items()
        }
    return {
        "N": n,
        "n_transient": n_transient,
        "n_measure": n_measure,
        "calls": calls,
        "available": list(AVAILABLE_BACKENDS),
        "baseline": baseline,
        "backends": measured,
    }


def _artifact(path: Path) -> dict[str, str]:
    """Pin an actual source or runtime file without inferring its build identity."""
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> int:
    """Run the real CLI and record raw comparisons plus host/source identities."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 16, 64])
    parser.add_argument("--n-transient", type=int, default=2)
    parser.add_argument("--n-measure", type=int, default=3)
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument(
        "--backends", nargs="+", choices=("python", "rust", "go", "julia", "mojo")
    )
    args = parser.parse_args()
    owners = None if args.backends is None else tuple(args.backends)
    root = Path(__file__).resolve().parents[1]
    start = time.time()
    load_before = os.getloadavg() if hasattr(os, "getloadavg") else None
    sources = [
        "src/scpn_phase_orchestrator/upde/basin_stability.py",
        "src/scpn_phase_orchestrator/upde/bifurcation.py",
        "src/scpn_phase_orchestrator/upde/_basin_stability_validation.py",
        *[
            "src/scpn_phase_orchestrator/experimental/accelerators/upde/_basin_stability_"
            + name
            + ".py"
            for name in ("go", "julia", "mojo")
        ],
        "spo-kernel/crates/spo-engine/src/basin_stability.rs",
        "spo-kernel/crates/spo-engine/src/bifurcation.rs",
        "spo-kernel/crates/spo-ffi/src/stability_boundary.rs",
        "go/basin_stability.go",
        "julia/basin_stability.jl",
        "mojo/basin_stability.mojo",
        "benchmarks/basin_stability_benchmark.py",
        "benchmarks/kuramoto_trial_reference.py",
    ]
    before = [_artifact(root / name) for name in sources]
    rows = [
        bench_at(n, args.n_transient, args.n_measure, args.calls, backends=owners)
        for n in args.sizes
    ]
    after = [_artifact(root / name) for name in sources]
    if before != after:
        raise RuntimeError("benchmark sources changed during measurement")
    artifacts = []
    if any("rust" in row["backends"] for row in rows):
        module = importlib.import_module("spo_kernel.spo_kernel")
        artifacts.append(_artifact(Path(cast(str, module.__file__))))
    for name, owner in (
        ("go/libbasin_stability.so", "go"),
        ("mojo/basin_stability_mojo", "mojo"),
    ):
        if any(owner in row["backends"] for row in rows):
            artifacts.append(_artifact(root / name))
    governor = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    report = {
        "schema": "spo.basin-coupling-consumer-diagnostics.v1",
        "claim_boundary": "shared-host diagnostics; no production-budget "
        "or isolated speedup acceptance",
        "host": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "command": sys.argv,
        "started_unix": start,
        "finished_unix": time.time(),
        "isolation": "none; shared workstation",
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "governor": governor.read_text().strip() if governor.is_file() else None,
        "load_before": load_before,
        "load_after": os.getloadavg() if hasattr(os, "getloadavg") else None,
        "sources": before,
        "artifacts": artifacts,
        "results": rows,
    }
    serialized = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    else:
        # Julia can make inherited POSIX stdio nonblocking during initialisation.
        # This CLI owns stdout and emits one complete synchronous JSON document.
        if os.name == "posix":
            os.set_blocking(sys.stdout.fileno(), True)
        print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
