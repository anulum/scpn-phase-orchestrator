# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Chimera multi-backend benchmark

"""Benchmark public local-order calls and original polyglot parity contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from benchmarks.chimera_local_order_reference import scalar_local_order
from scpn_phase_orchestrator.monitor.chimera import (
    ACTIVE_BACKEND,
    AVAILABLE_BACKENDS,
    local_order_parameter,
)

TWO_PI = 2.0 * np.pi
BACKEND_ORDER = ("rust", "mojo", "julia", "go", "python")
PARITY_TOLERANCES = {
    "rust": 1.0e-12,
    "mojo": 1.0e-9,
    "julia": 1.0e-12,
    "go": 1.0e-12,
    "python": 1.0e-12,
}


def _bench(
    backend: str, phases: NDArray[np.float64], knm: NDArray[np.float64], calls: int
) -> float:
    """Time actual named public calls after one untimed warm-up."""
    local_order_parameter(phases, knm, backend=backend)
    t0 = time.perf_counter()
    for _ in range(calls):
        local_order_parameter(phases, knm, backend=backend)
    return time.perf_counter() - t0


def bench_at(n: int, density: float, calls: int) -> dict[str, object]:
    """Compare named public owners on one deterministic admitted graph."""
    n = _validate_int_control(n, name="n", minimum=3)
    calls = _validate_int_control(calls, name="calls", minimum=1)
    density = _validate_density_control(density)
    rng = np.random.default_rng(42)
    phases = rng.uniform(0, TWO_PI, n)
    knm = rng.uniform(0.0, 1.0, (n, n))
    knm = (knm > (1.0 - density)).astype(np.float64) * knm
    np.fill_diagonal(knm, 0.0)
    row: dict[str, object] = {
        "n": n,
        "density": density,
        "calls": calls,
        "available": AVAILABLE_BACKENDS,
    }
    for backend in AVAILABLE_BACKENDS:
        t = _bench(backend, phases, knm, calls)
        row[f"{backend}_ms_per_call"] = (t / calls) * 1000.0
    return row


def _emit_json(text: str) -> None:
    """Emit complete native JSON despite Julia's nonblocking stdout side effect."""
    stream = sys.stdout
    if "julia" in AVAILABLE_BACKENDS and sys.stdout is sys.__stdout__:
        # Julia's libuv initialization changes the original POSIX pipe flags.
        # Captured Python streams remain owned by their caller.
        os.set_blocking(stream.fileno(), True)
    print(text)


def main() -> int:
    """Run real owner timing, parity or the complete comparison CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--density", type=float, default=0.3)
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--parity-gate",
        action="store_true",
        help="emit deterministic all-backend chimera parity-gate JSON",
    )
    modes.add_argument(
        "--comparison",
        action="store_true",
        help="emit current all-CPU and distinct JAX raw comparison JSON",
    )
    parser.add_argument(
        "--require-backends",
        nargs="*",
        default=[],
        help="refuse a missing exact owner before any measurement",
    )
    args = parser.parse_args()
    try:
        args.calls = _validate_int_control(args.calls, name="calls", minimum=1)
        args.seed = _validate_int_control(args.seed, name="seed", minimum=0)
        args.density = _validate_density_control(args.density)
        args.sizes = [
            _validate_int_control(n, name="size", minimum=3) for n in args.sizes
        ]
        from benchmarks.chimera_comparison import require_owners

        require_owners(args.require_backends)
        if args.comparison:
            from benchmarks.chimera_comparison import benchmark_comparison

            result = benchmark_comparison(
                args.sizes, calls=args.calls, density=args.density, seed=args.seed
            )
            text = json.dumps(result, indent=2, sort_keys=True)
            _emit_json(text)
            if args.output:
                args.output.write_text(text + "\n", encoding="utf-8")
            return 0
    except ValueError as exc:
        parser.error(str(exc))

    if args.parity_gate:
        result = benchmark_chimera_polyglot_parity_gate(
            n=args.sizes[0],
            density=args.density,
            calls=args.calls,
            seed=args.seed,
        )
        text = json.dumps(result, indent=2, sort_keys=True)
        _emit_json(text)
        if args.output:
            args.output.write_text(text + "\n", encoding="utf-8")
        return 0 if result["acceptance_passed"] == 1 else 1

    print(f"Active: {ACTIVE_BACKEND}  Available: {AVAILABLE_BACKENDS}\n")
    header = f"{'N':>5} {'dens':>6} {'calls':>6}"
    for b in AVAILABLE_BACKENDS:
        header += f" {b + '_ms':>12}"
    print(header)
    print("-" * len(header))
    results: list[dict[str, object]] = []
    for n in args.sizes:
        row = bench_at(n, args.density, args.calls)
        results.append(row)
        line = f"{n:>5} {args.density:>6.2f} {args.calls:>6}"
        for b in AVAILABLE_BACKENDS:
            line += f" {cast('float', row[f'{b}_ms_per_call']):>12.4f}"
        print(line)
    if args.output:
        args.output.write_text(
            json.dumps({"results": results}, indent=2), encoding="utf-8"
        )
    return 0


def _validate_int_control(value: object, *, name: str, minimum: int) -> int:
    """Admit a nonboolean integer control at its declared minimum."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _validate_density_control(value: object) -> float:
    """Admit a finite real adjacency density in the unit interval."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError("density must be a finite real scalar in [0, 1]")
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("density must be finite and lie in [0, 1]")
    return result


def _all_to_all(n: int) -> NDArray[np.float64]:
    """Construct positive unit coupling with explicitly excluded self edges."""
    knm = np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(knm, 0.0)
    return knm


def _problem(
    n: int,
    density: float,
    seed: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Build deterministic finite phases and directed positive non-self coupling."""
    rng = np.random.default_rng(seed)
    phases = np.ascontiguousarray(rng.uniform(0.0, TWO_PI, n), dtype=np.float64)
    knm = rng.uniform(0.0, 1.0, (n, n))
    knm = (knm > (1.0 - density)).astype(np.float64) * knm
    np.fill_diagonal(knm, 0.0)
    knm = np.ascontiguousarray(knm, dtype=np.float64)
    shifted = np.ascontiguousarray((phases + 17.0) % TWO_PI, dtype=np.float64)
    synchronised = np.full(n, 0.37, dtype=np.float64)
    uniform_circle = np.ascontiguousarray(
        np.linspace(0.0, TWO_PI, n, endpoint=False),
        dtype=np.float64,
    )
    disconnected = np.zeros((n, n), dtype=np.float64)
    all_to_all = _all_to_all(n)
    return (
        phases,
        knm,
        shifted,
        synchronised,
        uniform_circle,
        disconnected,
        all_to_all,
    )


def _vector_sha256(values: NDArray[np.float64]) -> str:
    """Hash canonical float64 result bytes for evidence custody."""
    payload = np.ascontiguousarray(values, dtype=np.float64)
    return hashlib.sha256(payload.tobytes()).hexdigest()


def _backend_status(backend: str) -> tuple[bool, str]:
    """Report actual optional runtime availability without changing dispatch."""
    if backend in AVAILABLE_BACKENDS:
        return True, ""
    return False, f"{backend} backend was not resolved by monitor.chimera"


def _direct_local_order(
    backend: str,
    phases: NDArray[np.float64],
    knm: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Exercise a strict named public owner; retained name preserves the harness."""
    return local_order_parameter(phases, knm, backend=backend)


def _direct_bundle(
    backend: str,
    phases: NDArray[np.float64],
    knm: NDArray[np.float64],
    shifted: NDArray[np.float64],
    synchronised: NDArray[np.float64],
    uniform_circle: NDArray[np.float64],
    disconnected: NDArray[np.float64],
    all_to_all: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    """Exercise the original local-order contracts through the named owner."""
    return {
        "local_order": _direct_local_order(backend, phases, knm),
        "shifted_local_order": _direct_local_order(backend, shifted, knm),
        "synchronised_local_order": _direct_local_order(
            backend,
            synchronised,
            all_to_all,
        ),
        "uniform_circle_local_order": _direct_local_order(
            backend,
            uniform_circle,
            all_to_all,
        ),
        "disconnected_local_order": _direct_local_order(
            backend,
            phases,
            disconnected,
        ),
    }


def _bench_with_output(
    backend: str,
    phases: NDArray[np.float64],
    knm: NDArray[np.float64],
    shifted: NDArray[np.float64],
    synchronised: NDArray[np.float64],
    uniform_circle: NDArray[np.float64],
    disconnected: NDArray[np.float64],
    all_to_all: NDArray[np.float64],
    *,
    calls: int,
) -> tuple[float, dict[str, NDArray[np.float64]], list[float]]:
    """Warm one complete contract bundle and retain each actual call duration."""
    output = _direct_bundle(
        backend,
        phases,
        knm,
        shifted,
        synchronised,
        uniform_circle,
        disconnected,
        all_to_all,
    )
    samples: list[float] = []
    for _ in range(calls):
        started = time.perf_counter_ns()
        output = _direct_bundle(
            backend,
            phases,
            knm,
            shifted,
            synchronised,
            uniform_circle,
            disconnected,
            all_to_all,
        )
        samples.append((time.perf_counter_ns() - started) / 1e9)
    return sum(samples), output, samples


def _bundle_errors(
    actual: Mapping[str, NDArray[np.float64]],
    expected: Mapping[str, NDArray[np.float64]],
) -> dict[str, float]:
    """Measure maximum absolute error against each independent reference."""
    errors: dict[str, float] = {}
    for key in expected:
        errors[key] = float(np.max(np.abs(actual[key] - expected[key])))
    return errors


def _unit_interval(values: NDArray[np.float64]) -> bool:
    """Check finite local-order bounds in every returned vector."""
    tolerance = 1.0e-12
    return bool(np.all(values >= -tolerance) and np.all(values <= 1.0 + tolerance))


def _reference_contracts_passed(
    bundle: Mapping[str, NDArray[np.float64]],
    *,
    n: int,
) -> bool:
    """Check topology, singleton, empty and synchronized reference behavior."""
    expected_uniform = 1.0 / (n - 1)
    return (
        _unit_interval(bundle["local_order"])
        and _unit_interval(bundle["shifted_local_order"])
        and bool(
            np.allclose(
                bundle["shifted_local_order"],
                bundle["local_order"],
                rtol=0.0,
                atol=1.0e-12,
            )
        )
        and bool(
            np.allclose(
                bundle["synchronised_local_order"],
                1.0,
                rtol=0.0,
                atol=1.0e-12,
            )
        )
        and bool(
            np.allclose(
                bundle["disconnected_local_order"],
                0.0,
                rtol=0.0,
                atol=1.0e-12,
            )
        )
        and bool(
            np.allclose(
                bundle["uniform_circle_local_order"],
                expected_uniform,
                rtol=0.0,
                atol=1.0e-12,
            )
        )
    )


def benchmark_chimera_polyglot_parity_gate(
    *,
    n: int = 32,
    density: float = 0.4,
    calls: int = 1,
    seed: int = 2026,
) -> dict[str, object]:
    """Record chimera local-order parity across declared backend slots.

    Available backends must preserve the unweighted positive-adjacency local-order
    vector against an independent scalar equation oracle, including phase-gauge
    invariance, synchronised unit local order, disconnected zero local order,
    and the exact uniform-circle all-to-all reference ``1 / (N - 1)``.
    """
    n = _validate_int_control(n, name="n", minimum=3)
    calls = _validate_int_control(calls, name="calls", minimum=1)
    seed = _validate_int_control(seed, name="seed", minimum=0)
    density = _validate_density_control(density)

    (
        phases,
        knm,
        shifted,
        synchronised,
        uniform_circle,
        disconnected,
        all_to_all,
    ) = _problem(n, density, seed)
    reference = {
        "local_order": scalar_local_order(phases, knm),
        "shifted_local_order": scalar_local_order(shifted, knm),
        "synchronised_local_order": scalar_local_order(synchronised, all_to_all),
        "uniform_circle_local_order": scalar_local_order(uniform_circle, all_to_all),
        "disconnected_local_order": scalar_local_order(phases, disconnected),
    }

    records: list[dict[str, object]] = []
    parity_checked_count = 0
    parity_pass_count = 0
    available_backend_count = 0
    t0 = time.perf_counter()

    for backend in BACKEND_ORDER:
        tolerance = PARITY_TOLERANCES[backend]
        available, reason = _backend_status(backend)
        if not available:
            records.append(
                {
                    "backend": backend,
                    "status": "unavailable",
                    "ms_per_call": None,
                    "local_order_sha256": None,
                    "shifted_local_order_sha256": None,
                    "synchronised_local_order_sha256": None,
                    "uniform_circle_local_order_sha256": None,
                    "disconnected_local_order_sha256": None,
                    "max_abs_error": None,
                    "local_order_abs_error": None,
                    "shifted_local_order_abs_error": None,
                    "synchronised_local_order_abs_error": None,
                    "uniform_circle_local_order_abs_error": None,
                    "disconnected_local_order_abs_error": None,
                    "reference_contracts_passed": False,
                    "tolerance": tolerance,
                    "parity_passed": False,
                    "unavailable_reason": reason,
                }
            )
            continue

        available_backend_count += 1
        elapsed, bundle, samples = _bench_with_output(
            backend,
            phases,
            knm,
            shifted,
            synchronised,
            uniform_circle,
            disconnected,
            all_to_all,
            calls=calls,
        )
        errors = _bundle_errors(bundle, reference)
        max_abs_error = max(errors.values())
        reference_contracts_passed = _reference_contracts_passed(bundle, n=n)
        parity_passed = bool(max_abs_error <= tolerance and reference_contracts_passed)
        parity_checked_count += 1
        parity_pass_count += int(parity_passed)
        records.append(
            {
                "backend": backend,
                "status": "available",
                "ms_per_call": (elapsed / calls) * 1000.0,
                "sample_seconds": samples,
                "sample_count": len(samples),
                "timed_operation": "five public local-order contract calls",
                "local_order_sha256": _vector_sha256(bundle["local_order"]),
                "shifted_local_order_sha256": _vector_sha256(
                    bundle["shifted_local_order"]
                ),
                "synchronised_local_order_sha256": _vector_sha256(
                    bundle["synchronised_local_order"]
                ),
                "uniform_circle_local_order_sha256": _vector_sha256(
                    bundle["uniform_circle_local_order"]
                ),
                "disconnected_local_order_sha256": _vector_sha256(
                    bundle["disconnected_local_order"]
                ),
                "max_abs_error": max_abs_error,
                "local_order_abs_error": errors["local_order"],
                "shifted_local_order_abs_error": errors["shifted_local_order"],
                "synchronised_local_order_abs_error": errors[
                    "synchronised_local_order"
                ],
                "uniform_circle_local_order_abs_error": errors[
                    "uniform_circle_local_order"
                ],
                "disconnected_local_order_abs_error": errors[
                    "disconnected_local_order"
                ],
                "reference_contracts_passed": reference_contracts_passed,
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
        "require_disconnected_zero_local_order": True,
        "require_global_phase_shift_invariance": True,
        "require_python_reference": True,
        "require_synchronised_unit_local_order": True,
        "require_uniform_circle_reference": True,
        "require_unit_interval_local_order": True,
    }
    acceptance_passed = (
        len(records) == len(BACKEND_ORDER)
        and any(
            record["backend"] == "python" and record["status"] == "available"
            for record in records
        )
        and parity_pass_count == parity_checked_count
        and _reference_contracts_passed(reference, n=n)
    )
    benchmark_payload = {
        "n": n,
        "density": density,
        "calls": calls,
        "seed": seed,
        "records": records,
        "thresholds": thresholds,
        "reference_local_order_sha256": _vector_sha256(reference["local_order"]),
        "reference_shifted_local_order_sha256": _vector_sha256(
            reference["shifted_local_order"]
        ),
        "reference_synchronised_local_order_sha256": _vector_sha256(
            reference["synchronised_local_order"]
        ),
        "reference_uniform_circle_local_order_sha256": _vector_sha256(
            reference["uniform_circle_local_order"]
        ),
        "reference_disconnected_local_order_sha256": _vector_sha256(
            reference["disconnected_local_order"]
        ),
    }
    benchmark_sha = hashlib.sha256(
        json.dumps(benchmark_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    return {
        "suite": "chimera_polyglot_parity_gate",
        "backend_count": len(records),
        "available_backend_count": available_backend_count,
        "unavailable_backend_count": len(records) - available_backend_count,
        "parity_checked_count": parity_checked_count,
        "parity_pass_count": parity_pass_count,
        "all_available_passed": int(parity_pass_count == parity_checked_count),
        "python_reference_present": 1,
        "n": n,
        "density": density,
        "calls": calls,
        "seed": seed,
        "reference_local_order_min": float(np.min(reference["local_order"])),
        "reference_local_order_max": float(np.max(reference["local_order"])),
        "reference_local_order_mean": float(np.mean(reference["local_order"])),
        "reference_uniform_circle_value": float(
            reference["uniform_circle_local_order"][0]
        ),
        "reference_local_order_sha256": benchmark_payload[
            "reference_local_order_sha256"
        ],
        "reference_shifted_local_order_sha256": benchmark_payload[
            "reference_shifted_local_order_sha256"
        ],
        "reference_synchronised_local_order_sha256": benchmark_payload[
            "reference_synchronised_local_order_sha256"
        ],
        "reference_uniform_circle_local_order_sha256": benchmark_payload[
            "reference_uniform_circle_local_order_sha256"
        ],
        "reference_disconnected_local_order_sha256": benchmark_payload[
            "reference_disconnected_local_order_sha256"
        ],
        "benchmark_sha256": benchmark_sha,
        "wall_time_s": wall_time,
        "steps_per_second": parity_checked_count / wall_time if wall_time else 0.0,
        "acceptance_passed": int(acceptance_passed),
        "acceptance_thresholds_json": json.dumps(thresholds, sort_keys=True),
        "backend_records_json": json.dumps(records, sort_keys=True),
    }


if __name__ == "__main__":
    raise SystemExit(main())
