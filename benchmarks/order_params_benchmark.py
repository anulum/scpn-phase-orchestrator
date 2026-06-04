# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Order parameters multi-backend benchmark

"""Per-backend wall-clock benchmark for ``upde/order_params.py``.

Measures ``compute_order_parameter`` across every loaded backend for
a range of N. Matches the AttnRes benchmark template.

Usage::

    python benchmarks/order_params_benchmark.py
    python benchmarks/order_params_benchmark.py \\
        --output benchmarks/order_params_results.json --sizes 100 1000 10000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde import order_params as op_mod
from scpn_phase_orchestrator.upde.order_params import (
    ACTIVE_BACKEND,
    AVAILABLE_BACKENDS,
    compute_layer_coherence,
    compute_order_parameter,
    compute_plv,
)

TWO_PI = 2.0 * np.pi
BACKEND_ORDER = ("rust", "mojo", "julia", "go", "python")
PARITY_TOLERANCES = {
    "rust": 1.0e-12,
    "mojo": 1.0e-9,
    "julia": 1.0e-12,
    "go": 1.0e-12,
    "python": 0.0,
}


def _bench_one(backend: str, phases: NDArray[np.floating], n_calls: int) -> float:
    phase_shift = np.ascontiguousarray((phases + 0.37) % TWO_PI, dtype=np.float64)
    indices = np.ascontiguousarray(np.arange(0, phases.size, 2), dtype=np.int64)
    elapsed, _order, _plv, _layer = _bench_with_outputs(
        backend,
        phases,
        phase_shift,
        indices,
        n_calls,
    )
    return elapsed


def _bench_with_outputs(
    backend: str,
    phases: NDArray[np.floating],
    phase_shift: NDArray[np.floating],
    layer_indices: NDArray[np.integer],
    n_calls: int,
) -> tuple[float, tuple[float, float], float, float]:
    saved = op_mod.ACTIVE_BACKEND
    try:
        op_mod.ACTIVE_BACKEND = backend
        compute_order_parameter(phases)
        compute_plv(phases, phase_shift)
        compute_layer_coherence(phases, layer_indices)
        t0 = time.perf_counter()
        order_out: tuple[float, float] | None = None
        plv_out: float | None = None
        layer_out: float | None = None
        for _ in range(n_calls):
            order_out = compute_order_parameter(phases)
            plv_out = compute_plv(phases, phase_shift)
            layer_out = compute_layer_coherence(phases, layer_indices)
        if order_out is None or plv_out is None or layer_out is None:
            raise RuntimeError("benchmark calls must be positive")
        return time.perf_counter() - t0, order_out, plv_out, layer_out
    finally:
        op_mod.ACTIVE_BACKEND = saved


def bench_at(n: int, n_calls: int) -> dict:
    rng = np.random.default_rng(42)
    phases = rng.uniform(0.0, TWO_PI, size=n)
    row: dict = {"n": n, "n_calls": n_calls, "available": AVAILABLE_BACKENDS}
    for backend in AVAILABLE_BACKENDS:
        t = _bench_one(backend, phases, n_calls)
        row[f"{backend}_ms_per_call"] = (t / n_calls) * 1000.0
    return row


def _backend_status(backend: str) -> tuple[bool, str]:
    if backend in AVAILABLE_BACKENDS:
        return True, ""
    return False, f"{backend} backend was not resolved by upde.order_params"


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


def _scalar_sha256(value: float) -> str:
    payload = json.dumps(float(value), separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _order_output_sha256(order: tuple[float, float]) -> str:
    payload = json.dumps(
        {"R": float(order[0]), "psi": float(order[1])},
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _phase_abs_error(a: float, b: float) -> float:
    return float(abs(((a - b + np.pi) % TWO_PI) - np.pi))


def benchmark_order_parameter_polyglot_parity_gate(
    *,
    n: int = 64,
    calls: int = 1,
    seed: int = 2026,
) -> dict[str, object]:
    """Record order-parameter parity across every declared backend slot.

    The gate covers the full public order-parameter surface rather than only
    one scalar: global Kuramoto ``R`` plus mean phase, PLV for a deterministic
    phase-shifted series, and layer coherence on a deterministic oscillator
    subset. Available optional backends must agree with the forced Python
    reference within backend-specific numerical tolerances.
    """

    n = _validate_int_control(n, name="n", minimum=2)
    calls = _validate_int_control(calls, name="calls", minimum=1)
    seed = _validate_int_control(seed, name="seed", minimum=0)

    rng = np.random.default_rng(seed)
    phases = np.ascontiguousarray(rng.uniform(0.0, TWO_PI, size=n), dtype=np.float64)
    phase_shift = np.ascontiguousarray(
        (phases + rng.normal(0.0, 0.2, size=n) + 0.37) % TWO_PI,
        dtype=np.float64,
    )
    layer_indices = np.ascontiguousarray(np.arange(0, n, 2), dtype=np.int64)

    t0 = time.perf_counter()
    _reference_elapsed, reference_order, reference_plv, reference_layer = (
        _bench_with_outputs(
            "python",
            phases,
            phase_shift,
            layer_indices,
            1,
        )
    )

    records: list[dict[str, object]] = []
    parity_checked_count = 0
    parity_pass_count = 0
    available_backend_count = 0

    for backend in BACKEND_ORDER:
        tolerance = PARITY_TOLERANCES[backend]
        available, reason = _backend_status(backend)
        if not available:
            records.append(
                {
                    "backend": backend,
                    "status": "unavailable",
                    "ms_per_call": None,
                    "order_parameter_sha256": None,
                    "plv_sha256": None,
                    "layer_coherence_sha256": None,
                    "r_abs_error": None,
                    "psi_abs_error": None,
                    "plv_abs_error": None,
                    "layer_abs_error": None,
                    "max_abs_error": None,
                    "tolerance": tolerance,
                    "parity_passed": False,
                    "unavailable_reason": reason,
                }
            )
            continue

        available_backend_count += 1
        elapsed, order, plv_value, layer_value = _bench_with_outputs(
            backend,
            phases,
            phase_shift,
            layer_indices,
            calls,
        )
        r_error = abs(float(order[0]) - float(reference_order[0]))
        psi_error = _phase_abs_error(float(order[1]), float(reference_order[1]))
        plv_error = abs(float(plv_value) - float(reference_plv))
        layer_error = abs(float(layer_value) - float(reference_layer))
        max_abs_error = max(r_error, psi_error, plv_error, layer_error)
        parity_passed = bool(max_abs_error <= tolerance)
        parity_checked_count += 1
        parity_pass_count += int(parity_passed)
        records.append(
            {
                "backend": backend,
                "status": "available",
                "ms_per_call": (elapsed / calls) * 1000.0,
                "order_parameter_sha256": _order_output_sha256(order),
                "plv_sha256": _scalar_sha256(plv_value),
                "layer_coherence_sha256": _scalar_sha256(layer_value),
                "r_abs_error": r_error,
                "psi_abs_error": psi_error,
                "plv_abs_error": plv_error,
                "layer_abs_error": layer_error,
                "max_abs_error": max_abs_error,
                "tolerance": tolerance,
                "parity_passed": parity_passed,
                "unavailable_reason": "",
            }
        )

    wall_time = time.perf_counter() - t0
    thresholds = {
        "backend_order": list(BACKEND_ORDER),
        "max_mojo_abs_error": PARITY_TOLERANCES["mojo"],
        "max_native_abs_error": PARITY_TOLERANCES["rust"],
        "require_all_available_parity": True,
        "require_all_declared_backend_records": True,
        "require_layer_coherence_contract": True,
        "require_plv_contract": True,
        "require_python_reference": True,
        "require_unit_interval_outputs": True,
    }
    acceptance_passed = (
        len(records) == len(BACKEND_ORDER)
        and any(
            record["backend"] == "python" and record["status"] == "available"
            for record in records
        )
        and parity_pass_count == parity_checked_count
        and 0.0 <= reference_order[0] <= 1.0
        and 0.0 <= reference_order[1] < TWO_PI
        and 0.0 <= reference_plv <= 1.0
        and 0.0 <= reference_layer <= 1.0
    )
    benchmark_payload = {
        "n": n,
        "calls": calls,
        "seed": seed,
        "records": records,
        "thresholds": thresholds,
        "reference_order_parameter_sha256": _order_output_sha256(reference_order),
        "reference_plv_sha256": _scalar_sha256(reference_plv),
        "reference_layer_coherence_sha256": _scalar_sha256(reference_layer),
    }
    benchmark_sha = hashlib.sha256(
        json.dumps(benchmark_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    return {
        "suite": "order_parameter_polyglot_parity_gate",
        "backend_count": len(records),
        "available_backend_count": available_backend_count,
        "unavailable_backend_count": len(records) - available_backend_count,
        "parity_checked_count": parity_checked_count,
        "parity_pass_count": parity_pass_count,
        "all_available_passed": int(parity_pass_count == parity_checked_count),
        "python_reference_present": 1,
        "n": n,
        "calls": calls,
        "seed": seed,
        "reference_r": float(reference_order[0]),
        "reference_psi": float(reference_order[1]),
        "reference_plv": float(reference_plv),
        "reference_layer_coherence": float(reference_layer),
        "reference_order_parameter_sha256": _order_output_sha256(reference_order),
        "reference_plv_sha256": _scalar_sha256(reference_plv),
        "reference_layer_coherence_sha256": _scalar_sha256(reference_layer),
        "benchmark_sha256": benchmark_sha,
        "wall_time_s": wall_time,
        "steps_per_second": parity_checked_count / wall_time if wall_time else 0.0,
        "acceptance_passed": int(acceptance_passed),
        "acceptance_thresholds_json": json.dumps(thresholds, sort_keys=True),
        "backend_records_json": json.dumps(records, sort_keys=True),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None, help="JSON results file.")
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 256, 4096, 65536])
    parser.add_argument("--calls", type=int, default=500)
    parser.add_argument(
        "--parity-gate",
        action="store_true",
        help="emit deterministic all-backend parity-gate JSON",
    )
    args = parser.parse_args()

    if args.parity_gate:
        result = benchmark_order_parameter_polyglot_parity_gate(
            n=args.sizes[0],
            calls=args.calls,
        )
        text = json.dumps(result, indent=2, sort_keys=True)
        print(text)
        if args.output:
            args.output.write_text(text + "\n", encoding="utf-8")
        return 0

    print(f"Active backend: {ACTIVE_BACKEND}")
    print(f"Available (fastest first): {AVAILABLE_BACKENDS}\n")
    header = f"{'N':>7}"
    for backend in AVAILABLE_BACKENDS:
        header += f" {backend + '_ms':>12}"
    print(header)
    print("-" * len(header))

    results: list[dict] = []
    for n in args.sizes:
        row = bench_at(n, args.calls)
        results.append(row)
        line = f"{n:>7}"
        for backend in AVAILABLE_BACKENDS:
            line += f" {row[f'{backend}_ms_per_call']:>12.4f}"
        print(line)

    if args.output:
        args.output.write_text(
            json.dumps({"results": results}, indent=2), encoding="utf-8"
        )
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
