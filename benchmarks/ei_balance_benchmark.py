# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Real E/I runtime diagnostic comparison

"""Measure actual public E/I operations with reproducible source inputs.

Run each maintained Python installation separately. Timings on this shared
host are local diagnostics, never production performance or speedup evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from functools import partial
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.coupling import adjust_ei_ratio, compute_ei_balance
from scpn_phase_orchestrator.coupling.ei_balance import EIBalance


def native_binary_provenance(module_name: str) -> tuple[str | None, str | None]:
    """Resolve and hash a real extension module or its same-name package member.

    Parameters
    ----------
    module_name : str
        Import name of the installed extension or its packaging wrapper.

    Returns
    -------
    tuple[str or None, str or None]
        Actual binary origin and SHA-256, or two ``None`` values when absent.

    Raises
    ------
    RuntimeError
        If a present candidate resolves to no native extension binary.
    """
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return None, None
    if spec.submodule_search_locations is not None:
        member = module_name.rsplit(".", 1)[-1]
        spec = importlib.util.find_spec(f"{module_name}.{member}")
    if (
        spec is None
        or spec.origin is None
        or not spec.origin.endswith(tuple(EXTENSION_SUFFIXES))
    ):
        raise RuntimeError("installed candidate has no resolvable native binary")
    binary = Path(spec.origin)
    return str(binary), hashlib.sha256(binary.read_bytes()).hexdigest()


def validate_ei_measurement(
    matrix: NDArray[np.float64],
    excitatory_indices: list[int],
    inhibitory_indices: list[int],
    summary: EIBalance,
    adjusted: NDArray[np.float64],
    target_ratio: float,
) -> None:
    """Check real runtime outputs against their declared benchmark source.

    Use this before timing or associating an independently obtained result with
    an input. The diagnostic contract uses moderate positive coupling and a
    disjoint, nonempty source partition, as in the timing driver; it does not
    promise attainment for arbitrary signed, silent or overlapping groups.

    Parameters
    ----------
    matrix : NDArray[np.float64]
        Original finite square benchmark coupling, including its diagonal.
    excitatory_indices : list[int]
        Excitatory source rows in the disjoint partition.
    inhibitory_indices : list[int]
        Inhibitory source rows in the disjoint partition.
    summary : EIBalance
        Actual public summary to associate with this input.
    adjusted : NDArray[np.float64]
        Actual public adjustment to associate with this input and target.
    target_ratio : float
        Declared finite positive target outside the silent-summary regime.

    Raises
    ------
    AssertionError
        If means, attained ratio or adjusted rows disagree with the reference.
    """
    expected_e = float(matrix[excitatory_indices].mean())
    expected_i = float(matrix[inhibitory_indices].mean())
    if not (
        np.isclose(summary.excitatory_strength, expected_e, rtol=1e-12, atol=0.0)
        and np.isclose(summary.inhibitory_strength, expected_i, rtol=1e-12, atol=0.0)
    ):
        raise AssertionError("public E/I means disagree with independent matrix means")
    if not np.isclose(
        compute_ei_balance(adjusted, excitatory_indices, inhibitory_indices).ratio,
        target_ratio,
        rtol=1e-12,
        atol=0.0,
    ):
        raise AssertionError("public E/I adjustment did not attain the declared target")
    np.testing.assert_allclose(
        adjusted[excitatory_indices], matrix[excitatory_indices], rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        adjusted[inhibitory_indices],
        matrix[inhibitory_indices] * expected_e / expected_i / target_ratio,
        rtol=1e-12,
        atol=0.0,
    )


def benchmark_ei_balance(
    sizes: Sequence[int] = (16, 64, 256), *, calls: int = 10, repeats: int = 3
) -> dict[str, object]:
    """Measure real summary and adjustment with deterministic dense coupling.

    Parameters
    ----------
    sizes : Sequence[int]
        Oscillator counts of at least two.
    calls : int
        Calls per timing repetition, excluding one initial warm-up.
    repeats : int
        Number of retained timing repetitions.

    Returns
    -------
    dict[str, object]
        JSON-safe runtime provenance, correctness evidence and raw timings.

    Raises
    ------
    ValueError
        If dimensions or timing counts are invalid.
    AssertionError
        If real runtime results violate the independent matrix expectations.
    """
    if (
        isinstance(calls, bool)
        or not isinstance(calls, int)
        or calls < 1
        or isinstance(repeats, bool)
        or not isinstance(repeats, int)
        or repeats < 1
        or not sizes
        or any(isinstance(n, bool) or not isinstance(n, int) or n < 2 for n in sizes)
    ):
        raise ValueError(
            "sizes must be integers >= 2; calls and repeats positive integers"
        )
    binary, binary_sha = native_binary_provenance("spo_kernel")
    rows: list[dict[str, object]] = []
    for n in sizes:
        rng = np.random.default_rng(20261001 + n)
        matrix = rng.uniform(0.1, 1.0, (n, n))
        np.fill_diagonal(matrix, 0.0)
        exc, inh = list(range(n // 2)), list(range(n // 2, n))
        summary = compute_ei_balance(matrix, exc, inh)
        adjusted = adjust_ei_ratio(matrix, exc, inh, 1.5)
        validate_ei_measurement(matrix, exc, inh, summary, adjusted, 1.5)
        operations: dict[str, Callable[[], object]] = {
            "compute": partial(compute_ei_balance, matrix, exc, inh),
            "adjust": partial(adjust_ei_ratio, matrix, exc, inh, 1.5),
        }
        for operation, function in operations.items():
            function()
            timings: list[float] = []
            for _ in range(repeats):
                start = time.perf_counter_ns()
                for _ in range(calls):
                    function()
                timings.append((time.perf_counter_ns() - start) / calls)
            rows.append(
                {
                    "n": n,
                    "operation": operation,
                    "calls": calls,
                    "repeats": repeats,
                    "ns_per_call": timings,
                    "median_ns_per_call": statistics.median(timings),
                    "input_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
                    "summary": asdict(summary),
                    "adjusted_ratio": compute_ei_balance(adjusted, exc, inh).ratio,
                }
            )
    return {
        "scope": (
            "non-isolated shared-host functional/parity and local regression evidence"
        ),
        "production_performance_claim": False,
        "speedup_claim": False,
        "actual_native": binary is not None,
        "native_binary": binary,
        "native_binary_sha256": binary_sha,
        "python": sys.version,
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the real diagnostic CLI and emit strict JSON to standard output.

    Parameters
    ----------
    argv : Sequence[str] or None
        Explicit arguments, or the actual process arguments.

    Returns
    -------
    int
        Zero after all real numerical checks complete.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--calls", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    record = benchmark_ei_balance(args.sizes, calls=args.calls, repeats=args.repeats)
    record["command"] = [
        sys.executable,
        str(Path(__file__).resolve()),
        *(list(argv) if argv is not None else sys.argv[1:]),
    ]
    print(json.dumps(record, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
