# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Symbolic extraction diagnostics

"""Measure real public symbolic extraction, with observed native-call provenance."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import TypedDict

import numpy as np

from scpn_phase_orchestrator.oscillators.symbolic import (
    SymbolicArray,
    SymbolicExtractor,
)


class SymbolicMeasurement(TypedDict):
    """One timed workload and its actual native-call and numerical observations."""

    name: str
    mode: str
    dtype: str
    samples: int
    n_states: str
    observed_backend: str
    native_calls: list[str]
    median_seconds: float
    p95_seconds: float
    sampled_theta: list[float]
    sampled_quality: list[float]


class SymbolicBenchmarkReport(TypedDict):
    """Environment-bound public-extraction diagnostics, not isolated speed claims."""

    format_version: int
    recorded_at: str
    classification: str
    python: str
    numpy: str
    kernel_present: bool
    native_artifact: dict[str, str] | None
    repeats: int
    host: dict[str, object]
    source_sha256: dict[str, str]
    cases: list[SymbolicMeasurement]


def _cases() -> list[tuple[str, str, SymbolicArray, int]]:
    """Return repeatable public workloads covering mapping, distance and capacity."""
    sequence = np.tile(np.arange(16, dtype=np.int64), 63)[:1000]
    return [
        ("ring_1000", "ring", sequence, 16),
        ("graph_1000", "graph", sequence, 16),
        (
            "signed_full_span",
            "graph",
            np.tile(np.array([-(2**63), 2**63 - 1], dtype=np.int64), 129)[:-1],
            4,
        ),
        (
            "unsigned_full_span",
            "graph",
            np.tile(np.array([0, 2**64 - 1], dtype=np.uint64), 129)[:-1],
            4,
        ),
        (
            "strided_graph",
            "graph",
            np.array([0, 0, 2, 2, 5, 5, 1, 1], dtype=np.int64)[::2],
            8,
        ),
        (
            "vocabulary_beyond_native_capacity",
            "ring",
            np.array([0, 1, 0], dtype=np.int64),
            int(np.iinfo(np.uintp).max) + 1,
        ),
    ]


def benchmark_symbolic_extraction(*, repeats: int = 20) -> SymbolicBenchmarkReport:
    """Time public extraction and record real C-call identities outside timed loops.

    Parameters
    ----------
    repeats : int
        Positive number of timed calls per workload, following one warm-up.

    Returns
    -------
    SymbolicBenchmarkReport
        Non-isolated timings in seconds, observed implementation, numerical
        samples, source hashes and host/runtime context for each workload.

    Raises
    ------
    ValueError
        If repeats is boolean, non-integral or less than one.
    """
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    kernel_present = importlib.util.find_spec("spo_kernel") is not None
    owners: dict[str, object] = {}
    native_artifact: dict[str, str] | None = None
    if kernel_present:
        kernel = importlib.import_module("spo_kernel")
        native_module = importlib.import_module("spo_kernel.spo_kernel")
        artifact = Path(inspect.getfile(native_module))
        native_artifact = {
            "filename": artifact.name,
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }
        owners = {
            name: getattr(kernel, name)
            for name in (
                "ring_phases_rust",
                "graph_walk_phases_rust",
                "transition_qualities_rust",
            )
        }
    load_before = list(os.getloadavg()) if hasattr(os, "getloadavg") else []
    measurements: list[SymbolicMeasurement] = []
    for name, mode, signal, n_states in _cases():
        extractor = SymbolicExtractor(n_states=n_states, mode=mode)
        executed: set[str] = set()

        def observe_call(
            frame: FrameType,
            event: str,
            argument: object,
            observed: set[str] = executed,
        ) -> None:
            """Identify calls to installed symbolic owners without replacing them."""
            if event == "c_call":
                for owner_name, owner in owners.items():
                    if argument is owner:
                        observed.add(owner_name)

        previous_profile = sys.getprofile()
        try:
            sys.setprofile(observe_call)
            states = extractor.extract(signal, 100.0)
        finally:
            sys.setprofile(previous_profile)
        durations = []
        for _repeat in range(repeats):
            started = time.perf_counter()
            extractor.extract(signal, 100.0)
            durations.append(time.perf_counter() - started)
        samples = [states[0], states[len(states) // 2], states[-1]]
        measurements.append(
            {
                "name": name,
                "mode": mode,
                "dtype": str(signal.dtype),
                "samples": len(signal),
                "n_states": str(n_states),
                "observed_backend": "native" if executed else "python",
                "native_calls": sorted(executed),
                "median_seconds": float(np.median(durations)),
                "p95_seconds": float(np.percentile(durations, 95)),
                "sampled_theta": [state.theta for state in samples],
                "sampled_quality": [state.quality for state in samples],
            }
        )
    repository = Path(__file__).resolve().parents[1]
    sources = (
        "benchmarks/bench_symbolic.py",
        "src/scpn_phase_orchestrator/oscillators/symbolic.py",
        "spo-kernel/crates/spo-oscillators/src/symbolic.rs",
        "spo-kernel/crates/spo-ffi/src/symbolic_bindings.rs",
        "spo-kernel/crates/spo-ffi/src/lib.rs",
    )
    return {
        "format_version": 1,
        "recorded_at": datetime.now(UTC).isoformat(),
        "classification": "non-isolated functional and local timing diagnostic",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "kernel_present": kernel_present,
        "native_artifact": native_artifact,
        "repeats": repeats,
        "host": {
            "machine": platform.machine(),
            "processor": platform.processor() or "not reported",
            "cpu_count": os.cpu_count(),
            "affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else [],
            "load_before": load_before,
            "load_after": list(os.getloadavg()) if hasattr(os, "getloadavg") else [],
            "isolation": "none; shared workstation",
        },
        "source_sha256": {
            source: hashlib.sha256((repository / source).read_bytes()).hexdigest()
            for source in sources
        },
        "cases": measurements,
    }


def main(argv: list[str] | None = None) -> int:
    """Emit the public extraction diagnostic as JSON from the command line.

    Parameters
    ----------
    argv : list[str] or None
        Command arguments without the program name; None reads process arguments.

    Returns
    -------
    int
        Zero after printing one finite JSON report to standard output.

    Raises
    ------
    SystemExit
        If argument parsing fails or the repeat count is invalid; the exit code
        is two and the usage error goes to standard error.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args(argv)
    try:
        report = benchmark_symbolic_extraction(repeats=args.repeats)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
