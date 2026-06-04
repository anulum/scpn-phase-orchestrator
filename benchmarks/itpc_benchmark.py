# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — ITPC multi-backend benchmark

"""Per-backend wall-clock benchmark for :func:`compute_itpc`."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor import itpc as it_mod
from scpn_phase_orchestrator.monitor.itpc import (
    ACTIVE_BACKEND,
    AVAILABLE_BACKENDS,
    compute_itpc,
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
    backend: str,
    phases: NDArray[np.floating],
    calls: int,
) -> float:
    saved = it_mod.ACTIVE_BACKEND
    try:
        it_mod.ACTIVE_BACKEND = backend
        compute_itpc(phases)  # warm-up
        t0 = time.perf_counter()
        for _ in range(calls):
            compute_itpc(phases)
        return time.perf_counter() - t0
    finally:
        it_mod.ACTIVE_BACKEND = saved


def bench_at(n_trials: int, n_tp: int, calls: int) -> dict:
    rng = np.random.default_rng(42)
    phases = rng.uniform(0.0, TWO_PI, (n_trials, n_tp))
    row: dict = {
        "n_trials": n_trials,
        "n_tp": n_tp,
        "calls": calls,
        "available": AVAILABLE_BACKENDS,
        "boundary_contract": "exact_numpy_reference_validated",
    }
    for backend in AVAILABLE_BACKENDS:
        t = _bench(backend, phases, calls)
        row[f"{backend}_ms_per_call"] = (t / calls) * 1000.0
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--n-trials-list", type=int, nargs="+", default=[20, 100, 500])
    parser.add_argument("--n-tp-list", type=int, nargs="+", default=[100, 500])
    parser.add_argument("--calls", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--parity-gate",
        action="store_true",
        help="emit deterministic all-backend ITPC parity-gate JSON",
    )
    args = parser.parse_args()

    if args.parity_gate:
        result = benchmark_itpc_polyglot_parity_gate(
            n_trials=args.n_trials_list[0],
            n_tp=args.n_tp_list[0],
            calls=args.calls,
            seed=args.seed,
        )
        text = json.dumps(result, indent=2, sort_keys=True)
        print(text)
        if args.output:
            args.output.write_text(text + "\n", encoding="utf-8")
        return 0

    print(f"Active: {ACTIVE_BACKEND}  Available: {AVAILABLE_BACKENDS}")
    print("Boundary contract: exact NumPy reference validated\n")
    header = f"{'trials':>7} {'tp':>6} {'calls':>6}"
    for b in AVAILABLE_BACKENDS:
        header += f" {b + '_ms':>12}"
    print(header)
    print("-" * len(header))
    results: list[dict] = []
    for n_trials in args.n_trials_list:
        for n_tp in args.n_tp_list:
            row = bench_at(n_trials, n_tp, args.calls)
            results.append(row)
            line = f"{n_trials:>7} {n_tp:>6} {args.calls:>6}"
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


def _problem(
    n_trials: int,
    n_tp: int,
    seed: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.int64],
]:
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, 6.0 * np.pi, n_tp, dtype=np.float64)
    carrier = grid + 0.15 * np.sin(0.7 * grid)
    trial_offsets = rng.normal(0.0, 0.35, size=(n_trials, 1))
    trial_noise = rng.normal(0.0, 0.05, size=(n_trials, n_tp))
    phases = np.ascontiguousarray(
        (carrier[np.newaxis, :] + trial_offsets + trial_noise) % TWO_PI,
        dtype=np.float64,
    )
    aligned = np.ascontiguousarray(
        np.tile((carrier % TWO_PI)[np.newaxis, :], (n_trials, 1)),
        dtype=np.float64,
    )
    opposed = np.ascontiguousarray(
        np.vstack(
            (
                np.zeros(n_tp, dtype=np.float64),
                np.full(n_tp, np.pi, dtype=np.float64),
            )
        ),
        dtype=np.float64,
    )
    pause_start = max(0, n_tp - max(1, n_tp // 4))
    pause_indices = np.ascontiguousarray(
        np.arange(pause_start, n_tp, dtype=np.int64),
        dtype=np.int64,
    )
    out_of_bounds_indices = np.ascontiguousarray(
        np.array([n_tp, n_tp + max(1, n_tp // 3)], dtype=np.int64),
        dtype=np.int64,
    )
    return phases, aligned, opposed, pause_indices, out_of_bounds_indices


def _vector_sha256(values: NDArray[np.float64]) -> str:
    payload = np.ascontiguousarray(values, dtype=np.float64)
    return hashlib.sha256(payload.tobytes()).hexdigest()


def _scalar_sha256(value: float) -> str:
    payload = json.dumps(float(value), separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _bundle_sha256(bundle: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    for key in sorted(bundle):
        value = bundle[key]
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        if isinstance(value, np.ndarray):
            digest.update(np.ascontiguousarray(value, dtype=np.float64).tobytes())
        else:
            digest.update(
                json.dumps(float(value), separators=(",", ":")).encode("utf-8")
            )
    return digest.hexdigest()


def _backend_status(backend: str) -> tuple[bool, str]:
    if backend in AVAILABLE_BACKENDS:
        return True, ""
    return False, f"{backend} backend was not resolved by monitor.itpc"


def _backend_function(backend: str, name: str) -> Callable[..., object]:
    loaded = it_mod._load_backend(backend)
    fn = loaded.get(name)
    if not callable(fn):
        raise ValueError(f"{backend} backend does not expose {name}")
    return fn


def _expected_persistence(
    phases: NDArray[np.float64],
    pause_indices: NDArray[np.int64],
) -> float:
    itpc_full = it_mod._compute_itpc_reference(phases)
    valid = pause_indices[(pause_indices >= 0) & (pause_indices < itpc_full.size)]
    if valid.size == 0:
        return 0.0
    return float(np.mean(itpc_full[valid]))


def _direct_itpc(
    backend: str,
    phases: NDArray[np.float64],
) -> NDArray[np.float64]:
    n_trials, n_tp = int(phases.shape[0]), int(phases.shape[1])
    expected = it_mod._compute_itpc_reference(phases)
    tolerance = PARITY_TOLERANCES[backend]
    if backend == "python":
        return it_mod._validate_itpc_values(
            expected,
            n_timepoints=n_tp,
            expected=expected,
            atol=tolerance,
        )
    raw = _backend_function(backend, "itpc")(
        np.ascontiguousarray(phases.ravel(), dtype=np.float64),
        n_trials,
        n_tp,
    )
    return it_mod._validate_itpc_values(
        raw,
        n_timepoints=n_tp,
        expected=expected,
        atol=tolerance,
    )


def _direct_persistence(
    backend: str,
    phases: NDArray[np.float64],
    pause_indices: NDArray[np.int64],
) -> float:
    n_trials, n_tp = int(phases.shape[0]), int(phases.shape[1])
    expected = _expected_persistence(phases, pause_indices)
    tolerance = PARITY_TOLERANCES[backend]
    if backend == "python":
        return it_mod._validate_persistence_value(
            expected,
            expected=expected,
            atol=tolerance,
        )
    raw = _backend_function(backend, "persistence")(
        np.ascontiguousarray(phases.ravel(), dtype=np.float64),
        n_trials,
        n_tp,
        np.ascontiguousarray(pause_indices, dtype=np.int64),
    )
    return it_mod._validate_persistence_value(raw, expected=expected, atol=tolerance)


def _direct_bundle(
    backend: str,
    phases: NDArray[np.float64],
    aligned: NDArray[np.float64],
    opposed: NDArray[np.float64],
    pause_indices: NDArray[np.int64],
    out_of_bounds_indices: NDArray[np.int64],
) -> dict[str, object]:
    return {
        "itpc": _direct_itpc(backend, phases),
        "aligned_itpc": _direct_itpc(backend, aligned),
        "opposed_itpc": _direct_itpc(backend, opposed),
        "persistence": _direct_persistence(backend, phases, pause_indices),
        "aligned_persistence": _direct_persistence(backend, aligned, pause_indices),
        "out_of_bounds_persistence": _direct_persistence(
            backend,
            phases,
            out_of_bounds_indices,
        ),
    }


def _bench_with_output(
    backend: str,
    phases: NDArray[np.float64],
    aligned: NDArray[np.float64],
    opposed: NDArray[np.float64],
    pause_indices: NDArray[np.int64],
    out_of_bounds_indices: NDArray[np.int64],
    *,
    calls: int,
) -> tuple[float, dict[str, object]]:
    _direct_bundle(
        backend,
        phases,
        aligned,
        opposed,
        pause_indices,
        out_of_bounds_indices,
    )
    output: dict[str, object] | None = None
    t0 = time.perf_counter()
    for _ in range(calls):
        output = _direct_bundle(
            backend,
            phases,
            aligned,
            opposed,
            pause_indices,
            out_of_bounds_indices,
        )
    if output is None:
        raise RuntimeError("benchmark calls must be positive")
    return time.perf_counter() - t0, output


def _value_abs_error(actual: object, expected: object) -> float:
    if isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray):
        if actual.size == 0 and expected.size == 0:
            return 0.0
        return float(np.max(np.abs(actual - expected)))
    return abs(float(actual) - float(expected))


def _bundle_errors(
    actual: Mapping[str, object],
    expected: Mapping[str, object],
) -> dict[str, float]:
    return {key: _value_abs_error(actual[key], expected[key]) for key in expected}


def _unit_interval_vector(values: NDArray[np.float64]) -> bool:
    tolerance = 1.0e-12
    return bool(np.all(values >= -tolerance) and np.all(values <= 1.0 + tolerance))


def _reference_contracts_passed(bundle: Mapping[str, object]) -> bool:
    itpc = bundle["itpc"]
    aligned_itpc = bundle["aligned_itpc"]
    opposed_itpc = bundle["opposed_itpc"]
    if not isinstance(itpc, np.ndarray):
        return False
    if not isinstance(aligned_itpc, np.ndarray):
        return False
    if not isinstance(opposed_itpc, np.ndarray):
        return False
    return (
        _unit_interval_vector(itpc)
        and _unit_interval_vector(aligned_itpc)
        and _unit_interval_vector(opposed_itpc)
        and bool(np.allclose(aligned_itpc, 1.0, rtol=0.0, atol=1.0e-12))
        and bool(np.allclose(opposed_itpc, 0.0, rtol=0.0, atol=1.0e-12))
        and 0.0 <= float(bundle["persistence"]) <= 1.0
        and np.isclose(float(bundle["aligned_persistence"]), 1.0, atol=1.0e-12)
        and float(bundle["out_of_bounds_persistence"]) == 0.0
    )


def benchmark_itpc_polyglot_parity_gate(
    *,
    n_trials: int = 48,
    n_tp: int = 96,
    calls: int = 1,
    seed: int = 2026,
) -> dict[str, object]:
    """Record ITPC parity across every declared backend slot.

    The gate covers the Lachaux inter-trial phase coherence vector and the
    pause-persistence scalar. Available optional backends must agree with the
    forced Python reference while preserving the aligned-trial unit-coherence,
    opposite-phase zero-coherence, unit-interval, and out-of-bounds pause
    contracts.
    """

    n_trials = _validate_int_control(n_trials, name="n_trials", minimum=2)
    n_tp = _validate_int_control(n_tp, name="n_tp", minimum=2)
    calls = _validate_int_control(calls, name="calls", minimum=1)
    seed = _validate_int_control(seed, name="seed", minimum=0)

    phases, aligned, opposed, pause_indices, out_of_bounds_indices = _problem(
        n_trials,
        n_tp,
        seed,
    )
    reference = _direct_bundle(
        "python",
        phases,
        aligned,
        opposed,
        pause_indices,
        out_of_bounds_indices,
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
                    "itpc_sha256": None,
                    "aligned_itpc_sha256": None,
                    "opposed_itpc_sha256": None,
                    "persistence_sha256": None,
                    "aligned_persistence_sha256": None,
                    "out_of_bounds_persistence_sha256": None,
                    "max_abs_error": None,
                    "itpc_abs_error": None,
                    "aligned_itpc_abs_error": None,
                    "opposed_itpc_abs_error": None,
                    "persistence_abs_error": None,
                    "aligned_persistence_abs_error": None,
                    "out_of_bounds_persistence_abs_error": None,
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
            aligned,
            opposed,
            pause_indices,
            out_of_bounds_indices,
            calls=calls,
        )
        errors = _bundle_errors(bundle, reference)
        max_abs_error = max(errors.values())
        reference_contracts_passed = _reference_contracts_passed(bundle)
        parity_passed = bool(max_abs_error <= tolerance and reference_contracts_passed)
        parity_checked_count += 1
        parity_pass_count += int(parity_passed)
        records.append(
            {
                "backend": backend,
                "status": "available",
                "ms_per_call": (elapsed / calls) * 1000.0,
                "itpc_sha256": _vector_sha256(bundle["itpc"]),
                "aligned_itpc_sha256": _vector_sha256(bundle["aligned_itpc"]),
                "opposed_itpc_sha256": _vector_sha256(bundle["opposed_itpc"]),
                "persistence_sha256": _scalar_sha256(bundle["persistence"]),
                "aligned_persistence_sha256": _scalar_sha256(
                    bundle["aligned_persistence"]
                ),
                "out_of_bounds_persistence_sha256": _scalar_sha256(
                    bundle["out_of_bounds_persistence"]
                ),
                "max_abs_error": max_abs_error,
                "itpc_abs_error": errors["itpc"],
                "aligned_itpc_abs_error": errors["aligned_itpc"],
                "opposed_itpc_abs_error": errors["opposed_itpc"],
                "persistence_abs_error": errors["persistence"],
                "aligned_persistence_abs_error": errors["aligned_persistence"],
                "out_of_bounds_persistence_abs_error": errors[
                    "out_of_bounds_persistence"
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
        "require_aligned_trials_unit_coherence": True,
        "require_all_available_parity": True,
        "require_all_declared_backend_records": True,
        "require_opposed_trials_zero_coherence": True,
        "require_out_of_bounds_pause_zero": True,
        "require_pause_persistence_contract": True,
        "require_python_reference": True,
        "require_unit_interval_itpc": True,
    }
    acceptance_passed = (
        len(records) == len(BACKEND_ORDER)
        and any(
            record["backend"] == "python" and record["status"] == "available"
            for record in records
        )
        and parity_pass_count == parity_checked_count
        and _reference_contracts_passed(reference)
    )
    benchmark_payload = {
        "n_trials": n_trials,
        "n_tp": n_tp,
        "calls": calls,
        "seed": seed,
        "records": records,
        "thresholds": thresholds,
        "reference_itpc_sha256": _vector_sha256(reference["itpc"]),
        "reference_aligned_itpc_sha256": _vector_sha256(reference["aligned_itpc"]),
        "reference_opposed_itpc_sha256": _vector_sha256(reference["opposed_itpc"]),
        "reference_persistence_sha256": _scalar_sha256(reference["persistence"]),
        "reference_aligned_persistence_sha256": _scalar_sha256(
            reference["aligned_persistence"]
        ),
        "reference_out_of_bounds_persistence_sha256": _scalar_sha256(
            reference["out_of_bounds_persistence"]
        ),
    }
    benchmark_sha = hashlib.sha256(
        json.dumps(benchmark_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    reference_itpc = reference["itpc"]
    if not isinstance(reference_itpc, np.ndarray):
        raise RuntimeError("reference ITPC must be an array")

    return {
        "suite": "itpc_polyglot_parity_gate",
        "backend_count": len(records),
        "available_backend_count": available_backend_count,
        "unavailable_backend_count": len(records) - available_backend_count,
        "parity_checked_count": parity_checked_count,
        "parity_pass_count": parity_pass_count,
        "all_available_passed": int(parity_pass_count == parity_checked_count),
        "python_reference_present": 1,
        "n_trials": n_trials,
        "n_tp": n_tp,
        "calls": calls,
        "seed": seed,
        "reference_itpc_min": float(np.min(reference_itpc)),
        "reference_itpc_max": float(np.max(reference_itpc)),
        "reference_itpc_mean": float(np.mean(reference_itpc)),
        "reference_persistence": float(reference["persistence"]),
        "reference_aligned_persistence": float(reference["aligned_persistence"]),
        "reference_out_of_bounds_persistence": float(
            reference["out_of_bounds_persistence"]
        ),
        "reference_itpc_sha256": benchmark_payload["reference_itpc_sha256"],
        "reference_aligned_itpc_sha256": benchmark_payload[
            "reference_aligned_itpc_sha256"
        ],
        "reference_opposed_itpc_sha256": benchmark_payload[
            "reference_opposed_itpc_sha256"
        ],
        "reference_persistence_sha256": benchmark_payload[
            "reference_persistence_sha256"
        ],
        "reference_aligned_persistence_sha256": benchmark_payload[
            "reference_aligned_persistence_sha256"
        ],
        "reference_out_of_bounds_persistence_sha256": benchmark_payload[
            "reference_out_of_bounds_persistence_sha256"
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
