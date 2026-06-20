# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Twin-confidence divergence multi-backend benchmark

"""Per-backend wall-clock benchmark for the twin-confidence divergence kernel.

Times ``monitor.twin_confidence.phase_order_divergence`` across the
Rust → Mojo → Julia → Go → Python fallback chain at increasing phase counts so
the cost profile of each backend is visible on the same axis, and provides a
parity gate that records every declared backend against the NumPy reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor import twin_confidence as tc
from scpn_phase_orchestrator.monitor.twin_confidence import (
    ACTIVE_BACKEND,
    AVAILABLE_BACKENDS,
    TwinDivergence,
    phase_order_divergence,
)

TWO_PI = 2.0 * np.pi
BACKEND_ORDER = ("rust", "mojo", "julia", "go", "python")
PARITY_TOLERANCES = {
    "rust": 1e-12,
    "julia": 1e-12,
    "go": 1e-12,
    "mojo": 1e-8,
    "python": 0.0,
}


def _problem(
    n: int, w: int, seed: int = 42
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    rng = np.random.default_rng(seed)
    model_phases = rng.uniform(0.0, TWO_PI, size=n)
    observed_phases = model_phases + rng.normal(0.0, 0.1, size=n)
    model_order = rng.uniform(0.0, 1.0, size=w)
    observed_order = np.clip(model_order + rng.normal(0.0, 0.02, size=w), 0.0, 1.0)
    return model_phases, observed_phases, model_order, observed_order


def _bench_with_result(
    backend: str,
    model_phases: NDArray[np.float64],
    observed_phases: NDArray[np.float64],
    model_order: NDArray[np.float64],
    observed_order: NDArray[np.float64],
    n_bins: int,
    calls: int,
) -> tuple[float, TwinDivergence]:
    saved = tc.ACTIVE_BACKEND
    try:
        tc.ACTIVE_BACKEND = backend
        # Warm-up — first call covers JIT / library init / FFI cache.
        phase_order_divergence(
            model_phases, observed_phases, model_order, observed_order, n_bins=n_bins
        )
        t0 = time.perf_counter()
        result: TwinDivergence | None = None
        for _ in range(calls):
            result = phase_order_divergence(
                model_phases,
                observed_phases,
                model_order,
                observed_order,
                n_bins=n_bins,
            )
        if result is None:
            raise RuntimeError("benchmark calls must be positive")
        return time.perf_counter() - t0, result
    finally:
        tc.ACTIVE_BACKEND = saved


def bench_at(n: int, w: int, n_bins: int, calls: int) -> dict[str, object]:
    """Time every available backend at one problem size.

    Parameters
    ----------
    n : int
        Phase-vector length.
    w : int
        Order-parameter window length.
    n_bins : int
        Number of phase histogram bins.
    calls : int
        Timed calls per backend (after one warm-up call).

    Returns
    -------
    dict[str, object]
        Per-backend milliseconds per call plus the problem parameters.
    """
    model_phases, observed_phases, model_order, observed_order = _problem(n, w)
    row: dict[str, object] = {
        "n": n,
        "w": w,
        "n_bins": n_bins,
        "calls": calls,
        "available": list(AVAILABLE_BACKENDS),
    }
    for backend in AVAILABLE_BACKENDS:
        elapsed, _result = _bench_with_result(
            backend,
            model_phases,
            observed_phases,
            model_order,
            observed_order,
            n_bins,
            calls,
        )
        row[f"{backend}_ms_per_call"] = (elapsed / calls) * 1000.0
    return row


def _pair_sha256(divergence: TwinDivergence) -> str:
    payload = np.asarray(
        [divergence.phase_js_divergence, divergence.order_wasserstein],
        dtype=np.float64,
    )
    return hashlib.sha256(payload.tobytes()).hexdigest()


def benchmark_twin_confidence_polyglot_parity_gate(
    *,
    n: int = 200,
    w: int = 64,
    n_bins: int = 36,
    calls: int = 50,
) -> dict[str, object]:
    """Benchmark all declared backends against the NumPy reference.

    Every declared language slot produces a record. Available backends are timed
    and compared against the same reference divergence pair; unavailable backends
    remain explicit records with a reason, so the gate is portable to hosts that
    lack an auxiliary toolchain.

    Parameters
    ----------
    n : int
        Phase-vector length.
    w : int
        Order-parameter window length.
    n_bins : int
        Number of phase histogram bins.
    calls : int
        Timed calls per backend.

    Returns
    -------
    dict[str, object]
        A deterministic, JSON-safe gate summary with per-backend timing, parity
        error, and acceptance flags.
    """
    model_phases, observed_phases, model_order, observed_order = _problem(
        n, w, seed=2026
    )
    reference_elapsed, reference = _bench_with_result(
        "python",
        model_phases,
        observed_phases,
        model_order,
        observed_order,
        n_bins,
        calls,
    )
    reference_pair = np.asarray(
        [reference.phase_js_divergence, reference.order_wasserstein], dtype=np.float64
    )
    records: list[dict[str, object]] = []
    for backend in BACKEND_ORDER:
        tolerance = PARITY_TOLERANCES[backend]
        if backend != "python" and backend not in AVAILABLE_BACKENDS:
            records.append(
                {
                    "backend": backend,
                    "status": "unavailable",
                    "tolerance": tolerance,
                    "ms_per_call": None,
                    "max_abs_error": None,
                    "parity_passed": False,
                    "pair_sha256": None,
                    "unavailable_reason": f"{backend} not resolved by twin_confidence",
                }
            )
            continue
        if backend == "python":
            elapsed, result = reference_elapsed, reference
        else:
            elapsed, result = _bench_with_result(
                backend,
                model_phases,
                observed_phases,
                model_order,
                observed_order,
                n_bins,
                calls,
            )
        pair = np.asarray(
            [result.phase_js_divergence, result.order_wasserstein], dtype=np.float64
        )
        max_abs_error = float(np.max(np.abs(pair - reference_pair)))
        records.append(
            {
                "backend": backend,
                "status": "available",
                "tolerance": tolerance,
                "ms_per_call": (elapsed / calls) * 1000.0,
                "max_abs_error": max_abs_error,
                "parity_passed": bool(max_abs_error <= tolerance),
                "pair_sha256": _pair_sha256(result),
                "unavailable_reason": "",
            }
        )

    available = [record for record in records if record["status"] == "available"]
    parity_pass_count = sum(int(bool(record["parity_passed"])) for record in records)
    acceptance_passed = int(
        len(records) == len(BACKEND_ORDER)
        and all(bool(record["parity_passed"]) for record in available)
        and any(record["backend"] == "python" for record in available)
    )
    return {
        "suite": "twin_confidence_polyglot_parity_gate",
        "backend_count": len(records),
        "available_backend_count": len(available),
        "parity_pass_count": parity_pass_count,
        "n": n,
        "w": w,
        "n_bins": n_bins,
        "calls": calls,
        "reference_pair_sha256": _pair_sha256(reference),
        "acceptance_passed": acceptance_passed,
        "backend_records_json": json.dumps(records, sort_keys=True),
    }


def main() -> int:
    """Run the twin-confidence benchmark from the command line.

    Returns
    -------
    int
        Process exit status (``0`` on success).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 256, 1024])
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--n-bins", type=int, default=36)
    parser.add_argument("--calls", type=int, default=200)
    parser.add_argument("--parity-gate", action="store_true")
    args = parser.parse_args()

    if args.parity_gate:
        result = benchmark_twin_confidence_polyglot_parity_gate(
            n=args.sizes[0], w=args.window, n_bins=args.n_bins, calls=args.calls
        )
        text = json.dumps(result, indent=2)
        print(text)
        if args.output:
            args.output.write_text(text + "\n", encoding="utf-8")
        return 0

    print(f"Active: {ACTIVE_BACKEND}  Available: {AVAILABLE_BACKENDS}\n")
    header = f"{'N':>6}{'W':>6}{'calls':>7}"
    for backend in AVAILABLE_BACKENDS:
        header += f" {backend + '_ms':>12}"
    print(header)
    print("-" * len(header))
    results: list[dict[str, object]] = []
    for n in args.sizes:
        row = bench_at(n, args.window, args.n_bins, args.calls)
        results.append(row)
        line = f"{n:>6}{args.window:>6}{args.calls:>7}"
        for backend in AVAILABLE_BACKENDS:
            line += f" {float(row[f'{backend}_ms_per_call']):>12.5f}"
        print(line)
    if args.output:
        args.output.write_text(
            json.dumps({"results": results}, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
