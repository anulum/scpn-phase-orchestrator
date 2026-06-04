# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Chimera multi-backend benchmark

"""Per-backend wall-clock benchmark for
``monitor.chimera.local_order_parameter``."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor import chimera as ch_mod
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


def _bench(backend: str, phases, knm, calls: int) -> float:
    saved = ch_mod.ACTIVE_BACKEND
    try:
        ch_mod.ACTIVE_BACKEND = backend
        local_order_parameter(phases, knm)
        t0 = time.perf_counter()
        for _ in range(calls):
            local_order_parameter(phases, knm)
        return time.perf_counter() - t0
    finally:
        ch_mod.ACTIVE_BACKEND = saved


def bench_at(n: int, density: float, calls: int) -> dict:
    rng = np.random.default_rng(42)
    phases = rng.uniform(0, TWO_PI, n)
    knm = rng.uniform(0.0, 1.0, (n, n))
    knm = (knm > (1.0 - density)).astype(np.float64) * knm
    np.fill_diagonal(knm, 0.0)
    row: dict = {
        "n": n,
        "density": density,
        "calls": calls,
        "available": AVAILABLE_BACKENDS,
    }
    for backend in AVAILABLE_BACKENDS:
        t = _bench(backend, phases, knm, calls)
        row[f"{backend}_ms_per_call"] = (t / calls) * 1000.0
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--density", type=float, default=0.3)
    parser.add_argument("--calls", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--parity-gate",
        action="store_true",
        help="emit deterministic all-backend chimera parity-gate JSON",
    )
    args = parser.parse_args()

    if args.parity_gate:
        result = benchmark_chimera_polyglot_parity_gate(
            n=args.sizes[0],
            density=args.density,
            calls=args.calls,
            seed=args.seed,
        )
        text = json.dumps(result, indent=2, sort_keys=True)
        print(text)
        if args.output:
            args.output.write_text(text + "\n", encoding="utf-8")
        return 0

    print(f"Active: {ACTIVE_BACKEND}  Available: {AVAILABLE_BACKENDS}\n")
    header = f"{'N':>5} {'dens':>6} {'calls':>6}"
    for b in AVAILABLE_BACKENDS:
        header += f" {b + '_ms':>12}"
    print(header)
    print("-" * len(header))
    results: list[dict] = []
    for n in args.sizes:
        row = bench_at(n, args.density, args.calls)
        results.append(row)
        line = f"{n:>5} {args.density:>6.2f} {args.calls:>6}"
        for b in AVAILABLE_BACKENDS:
            line += f" {row[f'{b}_ms_per_call']:>12.4f}"
        print(line)
    if args.output:
        args.output.write_text(
            json.dumps({"results": results}, indent=2), encoding="utf-8"
        )
    return 0


def _validate_int_control(value: object, *, name: str, minimum: int) -> int:
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
    NDArray[np.float64],
]:
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
        np.zeros(n, dtype=np.float64),
    )


def _vector_sha256(values: NDArray[np.float64]) -> str:
    payload = np.ascontiguousarray(values, dtype=np.float64)
    return hashlib.sha256(payload.tobytes()).hexdigest()


def _bundle_sha256(bundle: Mapping[str, NDArray[np.float64]]) -> str:
    digest = hashlib.sha256()
    for key in sorted(bundle):
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.ascontiguousarray(bundle[key], dtype=np.float64).tobytes())
    return digest.hexdigest()


def _backend_status(backend: str) -> tuple[bool, str]:
    if backend in AVAILABLE_BACKENDS:
        return True, ""
    return False, f"{backend} backend was not resolved by monitor.chimera"


def _backend_function(backend: str) -> Callable[..., NDArray[np.float64]]:
    loaded = ch_mod._load_backend(backend)
    if not callable(loaded):
        raise ValueError(f"{backend} backend is not callable")
    return loaded


def _direct_local_order(
    backend: str,
    phases: NDArray[np.float64],
    knm: NDArray[np.float64],
) -> NDArray[np.float64]:
    phases, knm = ch_mod._validate_chimera_inputs(phases, knm)
    n = int(phases.size)
    if backend == "python":
        saved = ch_mod.ACTIVE_BACKEND
        try:
            ch_mod.ACTIVE_BACKEND = "python"
            return local_order_parameter(phases, knm)
        finally:
            ch_mod.ACTIVE_BACKEND = saved
    raw = _backend_function(backend)(
        phases,
        np.ascontiguousarray(knm.ravel(), dtype=np.float64),
        n,
    )
    return ch_mod._validate_local_order(raw, n_oscillators=n)


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
) -> tuple[float, dict[str, NDArray[np.float64]]]:
    _direct_bundle(
        backend,
        phases,
        knm,
        shifted,
        synchronised,
        uniform_circle,
        disconnected,
        all_to_all,
    )
    output: dict[str, NDArray[np.float64]] | None = None
    t0 = time.perf_counter()
    for _ in range(calls):
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
    if output is None:
        raise RuntimeError("benchmark calls must be positive")
    return time.perf_counter() - t0, output


def _bundle_errors(
    actual: Mapping[str, NDArray[np.float64]],
    expected: Mapping[str, NDArray[np.float64]],
) -> dict[str, float]:
    errors: dict[str, float] = {}
    for key in expected:
        errors[key] = (
            float(np.max(np.abs(actual[key] - expected[key])))
            if actual[key].size
            else 0.0
        )
    return errors


def _unit_interval(values: NDArray[np.float64]) -> bool:
    tolerance = 1.0e-12
    return bool(np.all(values >= -tolerance) and np.all(values <= 1.0 + tolerance))


def _reference_contracts_passed(
    bundle: Mapping[str, NDArray[np.float64]],
    *,
    n: int,
) -> bool:
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

    Available backends must preserve the Kuramoto-Battogtokh local-order
    vector against the Python reference, including global phase-gauge
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
        _empty,
    ) = _problem(n, density, seed)
    reference = _direct_bundle(
        "python",
        phases,
        knm,
        shifted,
        synchronised,
        uniform_circle,
        disconnected,
        all_to_all,
    )

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
        elapsed, bundle = _bench_with_output(
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
