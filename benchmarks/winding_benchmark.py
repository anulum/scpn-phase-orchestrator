# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Phase winding multi-backend benchmark

"""Per-backend wall-clock benchmark for
``monitor.winding.winding_numbers``."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor import winding as w_mod
from scpn_phase_orchestrator.monitor.winding import (
    ACTIVE_BACKEND,
    AVAILABLE_BACKENDS,
    winding_numbers,
)

TWO_PI = 2.0 * np.pi
BACKEND_ORDER = ("rust", "mojo", "julia", "go", "python")


def _bench(backend: str, traj, calls: int) -> float:
    saved = w_mod.ACTIVE_BACKEND
    try:
        w_mod.ACTIVE_BACKEND = backend
        winding_numbers(traj)
        t0 = time.perf_counter()
        for _ in range(calls):
            winding_numbers(traj)
        return time.perf_counter() - t0
    finally:
        w_mod.ACTIVE_BACKEND = saved


def _problem(t: int, n: int, seed: int = 42) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    omegas = np.linspace(-2.4, 2.4, n, dtype=np.float64)
    omegas += rng.normal(0.0, 0.05, n)
    dt = 0.05
    hist = np.zeros((t, n), dtype=np.float64)
    hist[0] = rng.uniform(0, TWO_PI, n)
    for i in range(1, t):
        hist[i] = (hist[i - 1] + omegas * dt) % TWO_PI
    return np.ascontiguousarray(hist, dtype=np.float64)


def bench_at(t: int, n: int, calls: int) -> dict:
    hist = _problem(t, n)
    row: dict = {
        "T": t,
        "N": n,
        "calls": calls,
        "available": AVAILABLE_BACKENDS,
        "boundary_contract": "exact_numpy_wrapped_increment_validated",
    }
    for backend in AVAILABLE_BACKENDS:
        tm = _bench(backend, hist, calls)
        row[f"{backend}_ms_per_call"] = (tm / calls) * 1000.0
    return row


def _validate_int_control(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be an integer")
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return parsed


def _array_sha256(values: NDArray[np.integer]) -> str:
    contiguous = np.ascontiguousarray(values, dtype=np.int64)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _backend_status(backend: str) -> tuple[bool, str]:
    if backend in AVAILABLE_BACKENDS:
        return True, ""
    return False, f"{backend} backend was not resolved by monitor.winding"


def _direct_winding(
    backend: str,
    traj: NDArray[np.float64],
) -> NDArray[np.int64]:
    t, n = int(traj.shape[0]), int(traj.shape[1])
    expected = w_mod._winding_reference(traj)
    if backend == "python":
        return expected
    flat = np.ascontiguousarray(traj.ravel(), dtype=np.float64)
    backend_fn = w_mod._load_backend(backend)
    return w_mod._validate_backend_winding(
        backend_fn(flat, t, n),
        n=n,
        t=t,
        expected=expected,
    )


def _bench_with_output(
    backend: str,
    traj: NDArray[np.float64],
    calls: int,
) -> tuple[float, NDArray[np.int64]]:
    _direct_winding(backend, traj)
    t0 = time.perf_counter()
    output: NDArray[np.int64] | None = None
    for _ in range(calls):
        output = _direct_winding(backend, traj)
    if output is None:
        raise RuntimeError("benchmark calls must be positive")
    return time.perf_counter() - t0, output


def benchmark_winding_polyglot_parity_gate(
    *,
    t: int = 256,
    n: int = 8,
    calls: int = 1,
    seed: int = 2026,
) -> dict[str, object]:
    """Record exact winding parity across every declared backend slot.

    Unlike the public ``winding_numbers`` API, this gate does not allow the
    production fallback path to hide backend drift. Each resolved backend is
    called directly and its integer vector must equal the exact NumPy
    wrapped-increment reference.
    """

    t = _validate_int_control(t, name="t", minimum=2)
    n = _validate_int_control(n, name="n", minimum=1)
    calls = _validate_int_control(calls, name="calls", minimum=1)
    seed = _validate_int_control(seed, name="seed", minimum=0)

    traj = _problem(t, n, seed=seed)
    reference = w_mod._winding_reference(traj)

    records: list[dict[str, object]] = []
    parity_checked_count = 0
    parity_pass_count = 0
    available_backend_count = 0
    t0 = time.perf_counter()

    for backend in BACKEND_ORDER:
        available, reason = _backend_status(backend)
        if not available:
            records.append(
                {
                    "backend": backend,
                    "status": "unavailable",
                    "ms_per_call": None,
                    "winding_sha256": None,
                    "max_abs_error": None,
                    "exact_match": False,
                    "parity_passed": False,
                    "tolerance": 0,
                    "unavailable_reason": reason,
                }
            )
            continue

        available_backend_count += 1
        elapsed, winding = _bench_with_output(backend, traj, calls)
        diff = np.abs(winding.astype(np.int64) - reference.astype(np.int64))
        max_abs_error = int(np.max(diff)) if diff.size else 0
        exact_match = bool(np.array_equal(winding, reference))
        parity_checked_count += 1
        parity_pass_count += int(exact_match)
        records.append(
            {
                "backend": backend,
                "status": "available",
                "ms_per_call": (elapsed / calls) * 1000.0,
                "winding_sha256": _array_sha256(winding),
                "max_abs_error": max_abs_error,
                "exact_match": exact_match,
                "parity_passed": exact_match,
                "tolerance": 0,
                "unavailable_reason": "",
            }
        )

    wall_time = time.perf_counter() - t0
    thresholds = {
        "backend_order": list(BACKEND_ORDER),
        "require_all_available_parity": True,
        "require_all_declared_backend_records": True,
        "require_exact_integer_winding": True,
        "require_exact_wrapped_increment_reference": True,
        "require_python_reference": True,
        "tolerance": 0,
    }
    acceptance_passed = (
        len(records) == len(BACKEND_ORDER)
        and any(
            record["backend"] == "python" and record["status"] == "available"
            for record in records
        )
        and parity_pass_count == parity_checked_count
        and np.issubdtype(reference.dtype, np.integer)
    )
    benchmark_payload = {
        "T": t,
        "N": n,
        "calls": calls,
        "seed": seed,
        "records": records,
        "thresholds": thresholds,
        "reference_winding_sha256": _array_sha256(reference),
    }
    benchmark_sha = hashlib.sha256(
        json.dumps(benchmark_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    return {
        "suite": "winding_polyglot_parity_gate",
        "backend_count": len(records),
        "available_backend_count": available_backend_count,
        "unavailable_backend_count": len(records) - available_backend_count,
        "parity_checked_count": parity_checked_count,
        "parity_pass_count": parity_pass_count,
        "all_available_passed": int(parity_pass_count == parity_checked_count),
        "python_reference_present": 1,
        "T": t,
        "N": n,
        "calls": calls,
        "seed": seed,
        "reference_min_winding": int(np.min(reference)) if reference.size else 0,
        "reference_max_winding": int(np.max(reference)) if reference.size else 0,
        "reference_winding_sha256": _array_sha256(reference),
        "benchmark_sha256": benchmark_sha,
        "wall_time_s": wall_time,
        "steps_per_second": parity_checked_count / wall_time if wall_time else 0.0,
        "acceptance_passed": int(acceptance_passed),
        "acceptance_thresholds_json": json.dumps(thresholds, sort_keys=True),
        "backend_records_json": json.dumps(records, sort_keys=True),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--T-list", type=int, nargs="+", default=[500, 2000, 10000])
    parser.add_argument("--N", type=int, default=16)
    parser.add_argument("--calls", type=int, default=5)
    parser.add_argument(
        "--parity-gate",
        action="store_true",
        help="emit deterministic all-backend parity-gate JSON",
    )
    args = parser.parse_args()

    if args.parity_gate:
        result = benchmark_winding_polyglot_parity_gate(
            t=args.T_list[0],
            n=args.N,
            calls=args.calls,
        )
        text = json.dumps(result, indent=2, sort_keys=True)
        print(text)
        if args.output:
            args.output.write_text(text + "\n", encoding="utf-8")
        return 0

    print(f"Active: {ACTIVE_BACKEND}  Available: {AVAILABLE_BACKENDS}")
    print("Boundary contract: exact NumPy wrapped-increment reference validated\n")
    header = f"{'T':>6} {'N':>4} {'calls':>6}"
    for b in AVAILABLE_BACKENDS:
        header += f" {b + '_ms':>12}"
    print(header)
    print("-" * len(header))
    results: list[dict] = []
    for t in args.T_list:
        row = bench_at(t, args.N, args.calls)
        results.append(row)
        line = f"{t:>6} {args.N:>4} {args.calls:>6}"
        for b in AVAILABLE_BACKENDS:
            line += f" {row[f'{b}_ms_per_call']:>12.4f}"
        print(line)
    if args.output:
        args.output.write_text(
            json.dumps({"results": results}, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
