# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Real geometry projection benchmark

"""Measure checked public Python and installed native coupling projections."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable
from numbers import Integral
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.coupling import (
    NonNegativeConstraint,
    SymmetryConstraint,
    project_knm,
    validate_knm,
)


def measure_projection(n: int, calls: int = 10, repeats: int = 3) -> dict[str, object]:
    """Measure projection calls with correctness checks outside the timed intervals.

    Parameters
    ----------
    n : int
        Positive number of rows and columns in the reproducible input matrix.
    calls : int
        Positive number of calls in each measured sample.
    repeats : int
        Positive number of measured samples after warm-up.

    Returns
    -------
    dict[str, object]
        Input hash, runtime metadata, samples and verified Python/native results.

    Raises
    ------
    ValueError
        If a size or repetition count is not a positive non-boolean integer.
    AssertionError
        If a measured projection violates the numerical or ownership contract.

    Notes
    -----
    Shared-host samples provide local regression evidence, not isolated latency
    or causal speed-up measurements. Python projection does not dispatch to Rust.
    """
    if any(
        isinstance(v, bool) or not isinstance(v, Integral) or v < 1
        for v in (n, calls, repeats)
    ):
        raise ValueError(
            "sizes and repetition counts must be positive non-boolean integers"
        )
    n, calls, repeats = int(n), int(calls), int(repeats)
    raw = ((np.arange(n * n) % 31) / 31.0 - 0.5).reshape(n, n)
    original = raw.copy()
    expected = np.maximum(0.5 * (raw + raw.T), 0.0)
    np.fill_diagonal(expected, 0.0)
    constraints = [SymmetryConstraint(), NonNegativeConstraint()]
    native = (
        importlib.import_module("spo_kernel")
        if importlib.util.find_spec("spo_kernel")
        else None
    )

    def measure(fn: Callable[[], NDArray[np.float64]]) -> dict[str, object]:
        """Warm up and verify every sampled result without timing validation work."""
        np.testing.assert_array_equal(fn(), expected)
        samples: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(calls):
                result = fn()
            samples.append((time.perf_counter() - start) * 1e6 / calls)
            np.testing.assert_array_equal(result, expected)
            validate_knm(result)
            np.testing.assert_array_equal(raw, original)
        return {"samples_us": samples, "median_us": statistics.median(samples)}

    row: dict[str, object] = {
        "n": n,
        "calls": calls,
        "repeats": repeats,
        "input_sha256": hashlib.sha256(raw.astype("<f8").tobytes()).hexdigest(),
        "input_encoding": "little-endian IEEE754 binary64, row-major",
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "load_before": os.getloadavg(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "isolated": False,
        "native_installation": native.__file__ if native else None,
        "public_python": measure(lambda: project_knm(raw, constraints)),
    }
    if native:
        row["direct_native"] = measure(
            lambda: np.asarray(
                native.PyCouplingBuilder.project(raw.ravel(), n), dtype=np.float64
            ).reshape(n, n)
        )
    row["load_after"] = os.getloadavg()
    return row


def main() -> None:
    """Print strict JSON from the actual interpreter and kernel installation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[16, 64, 256])
    parser.add_argument("--calls", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if any(value < 1 for value in [*args.sizes, args.calls, args.repeats]):
        parser.error("sizes and repetition counts must be positive integers")
    root = Path(__file__).resolve().parents[1]
    owners = [
        "src/scpn_phase_orchestrator/coupling/geometry_constraints.py",
        "spo-kernel/crates/spo-engine/src/coupling.rs",
        "spo-kernel/crates/spo-ffi/src/coupling_builder.rs",
        "benchmarks/geometry_projection_benchmark.py",
    ]
    print(
        json.dumps(
            {
                "source_sha256": {
                    name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                    for name in owners
                },
                "rows": [
                    measure_projection(n, args.calls, args.repeats) for n in args.sizes
                ],
            },
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
