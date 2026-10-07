# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Public Phase-SINDy runtime diagnostics

"""Measure actual phase regression with independent numerical references.

Run each installation in its own process. Shared-workstation timings qualify
the measured local workload and never establish an isolated speedup.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType, FunctionType
from typing import Literal, TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.autotune.sindy import PhaseSINDy

FloatArray = NDArray[np.float64]
Backend = Literal["auto", "native", "python"]


@dataclass(frozen=True)
class _Case:
    """One independently defined trajectory and its expected coefficient layout."""

    name: str
    phases: FloatArray
    dt: float
    threshold: float
    reference: FloatArray


class Measurement(TypedDict):
    """Actual numerical result, invocation and repeated durations for one workload."""

    name: str
    samples: int
    nodes: int
    dt_seconds: float
    threshold: float
    dtype: str
    coefficients: list[list[float]]
    reference: list[list[float]]
    maximum_absolute_error: float
    tolerance: float
    observed_calls: list[str]
    durations_seconds: list[float]
    median_seconds: float
    p95_seconds: float
    equations: list[str]


class DiagnosticReport(TypedDict):
    """Bind runtime and source evidence to explicitly non-isolated timings."""

    schema_version: int
    recorded_at: str
    classification: str
    interpreter: str
    python: str
    numpy: str
    scipy: str
    backend: str
    native_artifact: dict[str, str] | None
    estimator_artifact: dict[str, str]
    source_sha256: dict[str, str]
    repeats: int
    host: dict[str, object]
    cases: list[Measurement]


def _cases() -> list[_Case]:
    """Define Euler, arbitrary-turn and analytical dependent-feature workloads."""
    omega = np.array([1.1, 1.8, 2.6], dtype=np.float64)
    coupling = np.array([[0.0, 0.2, 0.35], [-0.1, 0.0, 0.13], [0.07, -0.18, 0.0]])
    phases = np.empty((400, 3), dtype=np.float64)
    phases[0] = [0.0, 0.7, 2.1]
    for sample in range(1, 400):
        previous = phases[sample - 1]
        derivative = omega.copy()
        for target in range(3):
            for source in range(3):
                if source != target:
                    derivative[target] += coupling[target, source] * math.sin(
                        float(previous[source] - previous[target])
                    )
        phases[sample] = np.remainder(previous + 0.02 * derivative, math.tau)
    directed_reference = np.array(
        [[1.1, 0.2, 0.35], [1.8, -0.1, 0.13], [2.6, 0.07, -0.18]]
    )
    times = np.arange(40, dtype=np.float64) * 0.01
    sine = math.sin(0.4)
    inverse_norm = 1.0 / (1.0 + sine * sine)
    return [
        _Case("directed_euler", phases, 0.02, 0.0, directed_reference),
        _Case(
            "multiple_turn_alias",
            np.arange(12, dtype=np.float64).reshape(-1, 1) * (0.1 + 3 * math.tau),
            0.1,
            0.0,
            np.array([[1.0]], dtype=np.float64),
        ),
        _Case(
            "dependent_features",
            np.column_stack((times, times + 0.4)),
            0.01,
            0.0,
            np.array(
                [
                    [inverse_norm, sine * inverse_norm],
                    [inverse_norm, -sine * inverse_norm],
                ]
            ),
        ),
        _Case(
            "empty_support",
            np.array([[0.0], [0.1], [0.2]]),
            1.0,
            1.0,
            np.array([[0.0]]),
        ),
    ]


def benchmark_phase_sindy(
    *, repeats: int = 20, expect_backend: Backend = "auto"
) -> DiagnosticReport:
    """Observe public fits, reject incorrect results, and time repeated actual fits.

    Parameters
    ----------
    repeats : int, default=20
        Positive number of timed fits per workload following an observed warm-up.
    expect_backend : {"auto", "native", "python"}, default="auto"
        Require the actual installation to select the requested backend. No
        flags, import tables or production functions are replaced.

    Returns
    -------
    DiagnosticReport
        Numerical references, source/binary hashes, actual call observations,
        repeated timings in seconds and host/load/thread context.

    Raises
    ------
    ValueError
        If the repeat count or expected backend is invalid.
    RuntimeError
        If the installation, observed computation or numerical result violates
        the requested backend and independent reference contracts.
    """
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if expect_backend not in ("auto", "native", "python"):
        raise ValueError("expect_backend must be auto, native or python")
    present = importlib.util.find_spec("spo_kernel") is not None
    backend = "native" if present else "python"
    if expect_backend != "auto" and expect_backend != backend:
        raise RuntimeError(
            f"Actual installation selects {backend}, expected {expect_backend}"
        )
    owner: object = None
    native_artifact: dict[str, str] | None = None
    if present:
        kernel = importlib.import_module("spo_kernel")
        owner = kernel.sindy_fit_rust
        native_module = importlib.import_module("spo_kernel.spo_kernel")
        binary = Path(inspect.getfile(native_module))
        native_artifact = {
            "path": str(binary),
            "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "version": importlib.metadata.version("spo-kernel"),
        }
    scipy_owner: object = importlib.import_module("scipy.linalg").lstsq
    if not isinstance(scipy_owner, FunctionType):
        raise RuntimeError(
            "The installed SciPy least-squares owner has no Python call identity"
        )
    load_before = list(os.getloadavg()) if hasattr(os, "getloadavg") else []
    measurements: list[Measurement] = []
    for case in _cases():
        model = PhaseSINDy(threshold=case.threshold, max_iter=3)
        observed: set[str] = set()

        def observe_call(
            frame: FrameType,
            event: str,
            argument: object,
            observed_calls: set[str] = observed,
        ) -> None:
            """Identify actual installed owners without substituting numerical work."""
            if event == "c_call" and argument is owner:
                observed_calls.add("spo_kernel.spo_kernel.sindy_fit_rust")
            if event == "call" and frame.f_code is scipy_owner.__code__:
                observed_calls.add("scipy.linalg.lstsq")

        previous_profile = sys.getprofile()
        try:
            sys.setprofile(observe_call)
            coefficients = np.stack(model.fit(case.phases, case.dt))
        finally:
            sys.setprofile(previous_profile)
        required_call = (
            "spo_kernel.spo_kernel.sindy_fit_rust" if present else "scipy.linalg.lstsq"
        )
        if required_call not in observed:
            raise RuntimeError(f"No actual {required_call} invocation for {case.name}")
        error = float(np.max(np.abs(coefficients - case.reference)))
        tolerance = 2e-8
        if not np.isfinite(coefficients).all() or error > tolerance:
            raise RuntimeError(
                f"{case.name}: coefficient error {error} exceeds {tolerance}"
            )
        durations: list[float] = []
        for _ in range(repeats):
            started = time.perf_counter()
            model.fit(case.phases, case.dt)
            durations.append(time.perf_counter() - started)
        measurements.append(
            {
                "name": case.name,
                "samples": case.phases.shape[0],
                "nodes": case.phases.shape[1],
                "dt_seconds": case.dt,
                "threshold": case.threshold,
                "dtype": str(coefficients.dtype),
                "coefficients": coefficients.tolist(),
                "reference": case.reference.tolist(),
                "maximum_absolute_error": error,
                "tolerance": tolerance,
                "observed_calls": sorted(observed),
                "durations_seconds": durations,
                "median_seconds": statistics.median(durations),
                "p95_seconds": float(np.percentile(durations, 95)),
                "equations": model.get_equations(),
            }
        )
    repository = Path(__file__).resolve().parents[1]
    estimator_source = Path(inspect.getfile(PhaseSINDy))
    source_paths = (
        "benchmarks/phase_sindy_benchmark.py",
        "src/scpn_phase_orchestrator/autotune/sindy.py",
        "spo-kernel/crates/spo-engine/src/sindy.rs",
        "spo-kernel/crates/spo-ffi/src/lib.rs",
        "spo-kernel/Cargo.lock",
    )
    return {
        "schema_version": 1,
        "recorded_at": datetime.now(UTC).isoformat(),
        "classification": (
            "shared-workstation functional and timing diagnostic; "
            "no isolation or speedup claim"
        ),
        "interpreter": sys.executable,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": importlib.metadata.version("scipy"),
        "backend": backend,
        "native_artifact": native_artifact,
        "estimator_artifact": {
            "path": str(estimator_source),
            "sha256": hashlib.sha256(estimator_source.read_bytes()).hexdigest(),
        },
        "source_sha256": {
            name: hashlib.sha256((repository / name).read_bytes()).hexdigest()
            for name in source_paths
        },
        "repeats": repeats,
        "host": {
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else [],
            "load_before": load_before,
            "load_after": list(os.getloadavg()) if hasattr(os, "getloadavg") else [],
            "threads": {
                name: os.environ.get(name)
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "RAYON_NUM_THREADS",
                )
            },
            "isolation": "none; shared workstation",
        },
        "cases": measurements,
    }


def main(argv: list[str] | None = None) -> int:
    """Print one JSON diagnostic report from the actual installed runtime.

    Parameters
    ----------
    argv : list[str] or None
        CLI arguments without the programme name; None reads process arguments.

    Returns
    -------
    int
        Zero after every workload has exercised its expected real computation.

    Raises
    ------
    SystemExit
        With code two for invalid arguments or code one for a runtime mismatch.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--expect-backend", choices=("auto", "native", "python"), default="auto"
    )
    arguments = parser.parse_args(argv)
    try:
        report = benchmark_phase_sindy(
            repeats=int(arguments.repeats),
            expect_backend=cast(Backend, arguments.expect_backend),
        )
    except ValueError as error:
        parser.error(str(error))
    except RuntimeError as error:
        parser.exit(1, f"Phase-SINDy diagnostic failed: {error}\n")
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
