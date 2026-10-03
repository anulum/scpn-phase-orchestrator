# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Actual phase quality runtime benchmark
"""Measure real public and PyO3 quality operations with source provenance."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import statistics
import struct
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from numbers import Integral
from pathlib import Path

import numpy as np

from scpn_phase_orchestrator.oscillators.base import PhaseState
from scpn_phase_orchestrator.oscillators.quality import PhaseQualityScorer

try:
    from spo_kernel import PyPhaseQualityScorer as _NativeScorer
except ImportError:
    _NativeScorer = None


def measure_quality(n: int, calls: int = 100, repeats: int = 5) -> dict[str, object]:
    """Measure identical quality data through the actual available runtimes.

    Parameters
    ----------
    n : int
        Positive number of extracted states.
    calls : int
        Positive number of calls in each timed sample.
    repeats : int
        Positive number of samples after two warm-up calls.

    Returns
    -------
    dict[str, object]
        Real output checks, individual seconds-per-call samples, source and
        installed extension hashes, scalar bits and instrumentation/host context.

    Raises
    ------
    ValueError
        If a control is not a positive non-boolean integer.
    AssertionError
        If a real measured operation violates its numerical contract.

    Notes
    -----
    Public calls include state validation and list conversion; direct PyO3
    calls include source validation, while the standalone Rust producer times
    typed slices. These different boundaries and a shared host do not establish
    isolated latency, a speedup or a control deadline. Existing profilers are
    left in place and their presence is recorded rather than labelled absent.
    """
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) or value < 1
        for value in (n, calls, repeats)
    ):
        raise ValueError("controls must be positive non-boolean integers")
    n, calls, repeats = int(n), int(calls), int(repeats)
    qualities = [(i % 10) / 10.0 for i in range(n)]
    amplitudes = [1.0] * n
    huge = [1e308] * n
    states = [
        PhaseState(0.0, 1.0, amplitude, quality, "P", f"n{i}")
        for i, (quality, amplitude) in enumerate(
            zip(qualities, amplitudes, strict=True)
        )
    ]
    huge_states = [
        PhaseState(0.0, 1.0, amplitude, quality, "P", f"n{i}")
        for i, (quality, amplitude) in enumerate(zip(qualities, huge, strict=True))
    ]
    cycles, remainder = divmod(n, 10)
    expected_mean = (45 * cycles + remainder * (remainder - 1) // 2) / (10 * n)
    expected_mask = np.array([q if q >= 0.3 else 0.0 for q in qualities])
    expected_override = np.array([q if q >= 0.35 else 0.0 for q in qualities])
    scorer = PhaseQualityScorer()
    operations: list[tuple[str, Callable[[], object], object]] = [
        ("public_score", lambda: scorer.score(states), expected_mean),
        ("public_score_huge", lambda: scorer.score(huge_states), expected_mean),
        ("public_mask", lambda: scorer.downweight_mask(states), expected_mask),
        (
            "public_override_mask",
            lambda: scorer.downweight_mask(states, min_quality=0.35),
            expected_override,
        ),
    ]
    native_info: dict[str, object] | None = None
    if _NativeScorer is not None:
        native = _NativeScorer()
        operations.extend(
            [
                (
                    "native_score",
                    lambda: native.score(qualities, amplitudes),
                    expected_mean,
                ),
                (
                    "native_score_huge",
                    lambda: native.score(qualities, huge),
                    expected_mean,
                ),
                (
                    "native_mask",
                    lambda: native.downweight_mask(qualities),
                    expected_mask,
                ),
            ]
        )
        module = importlib.import_module("spo_kernel.spo_kernel")
        binary = Path(str(module.__file__)).resolve()
        native_info = {
            "path": str(binary),
            "bytes": binary.stat().st_size,
            "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        }
    results: dict[str, dict[str, object]] = {}
    for name, operation, expected in operations:
        for _ in range(2):
            np.testing.assert_allclose(
                np.asarray(operation(), dtype=np.float64),
                np.asarray(expected, dtype=np.float64),
                rtol=2e-13,
                atol=1e-15,
            )
        samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            for _ in range(calls):
                operation()
            samples.append((time.perf_counter() - started) / calls)
        observed = np.asarray(operation(), dtype=np.float64)
        np.testing.assert_allclose(
            observed, np.asarray(expected, dtype=np.float64), rtol=2e-13, atol=1e-15
        )
        checked_result: float | list[float] = (
            float(observed) if observed.ndim == 0 else [float(x) for x in observed]
        )
        results[name] = {
            "seconds_per_call": samples,
            "median_seconds": statistics.median(samples),
            "expected_mean": expected_mean,
            "checked_result": checked_result,
        }
    root = Path(__file__).resolve().parents[1]
    source_paths = [
        "src/scpn_phase_orchestrator/oscillators/quality.py",
        "spo-kernel/crates/spo-ffi/src/phase_quality.rs",
        "spo-kernel/crates/spo-oscillators/src/quality.rs",
        "spo-kernel/crates/spo-oscillators/benches/quality_bench.rs",
        "benchmarks/phase_quality_benchmark.py",
    ]
    return {
        "recorded_utc": datetime.now(UTC).isoformat(),
        "n": n,
        "calls": calls,
        "repeats": repeats,
        "python": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "host_load": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "profile_present": sys.getprofile() is not None,
        "trace_present": sys.gettrace() is not None,
        "public_backend": "native" if _NativeScorer is not None else "python",
        "native_binary": native_info,
        "normal_weight_bits": struct.unpack(">Q", struct.pack(">d", 1.0))[0],
        "huge_weight_bits": struct.unpack(">Q", struct.pack(">d", 1e308))[0],
        "source_sha256": {
            relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
            for relative in source_paths
        },
        "operations": results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark CLI and emit strict JSON to the caller's output.

    Parameters
    ----------
    argv : Sequence[str] or None
        Actual CLI arguments, or the current process arguments when omitted.

    Returns
    -------
    int
        Zero after all real output checks and measurements complete.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[10, 100, 1000])
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)
    report = [measure_quality(n, args.calls, args.repeats) for n in args.sizes]
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
