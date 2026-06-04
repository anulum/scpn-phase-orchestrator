# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Delay embedding benchmark gate

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor import embedding as embedding_module
from scpn_phase_orchestrator.monitor.embedding import (
    delay_embed,
    mutual_information,
    nearest_neighbor_distances,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

BACKEND_ORDER = ("rust", "mojo", "julia", "go", "python")
DELAY_TOLERANCE = 0.0
MI_TOLERANCE = 1.0e-8
NN_TOLERANCE = 1.0e-9
CONTRACT_TOLERANCE = 1.0e-12
BENCHMARK_EVIDENCE_KIND = "local_regression_non_isolated"


def _validate_int_control(value: int, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _problem(n: int, seed: int) -> FloatArray:
    n = _validate_int_control(n, name="n", minimum=64)
    seed = _validate_int_control(seed, name="seed", minimum=0)
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 8.0 * np.pi, n, dtype=np.float64)
    signal = np.sin(t) + 0.35 * np.sin(2.0 * t + 0.3) + 0.08 * rng.normal(0.0, 1.0, n)
    return np.ascontiguousarray(signal, dtype=np.float64)


def _force_backend(backend: str) -> str:
    previous = embedding_module.ACTIVE_BACKEND
    embedding_module.ACTIVE_BACKEND = backend
    return previous


def _restore_backend(previous: str) -> None:
    embedding_module.ACTIVE_BACKEND = previous


def _python_delay(signal: FloatArray, delay: int, dimension: int) -> FloatArray:
    previous = _force_backend("python")
    try:
        return delay_embed(signal, delay=delay, dimension=dimension)
    finally:
        _restore_backend(previous)


def _python_mi(signal: FloatArray, lag: int, n_bins: int) -> float:
    previous = _force_backend("python")
    try:
        return mutual_information(signal, lag=lag, n_bins=n_bins)
    finally:
        _restore_backend(previous)


def _python_nn(embedded: FloatArray) -> tuple[FloatArray, IntArray]:
    previous = _force_backend("python")
    try:
        return nearest_neighbor_distances(embedded)
    finally:
        _restore_backend(previous)


def _array_sha256(value: FloatArray | IntArray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _max_abs_error(actual: FloatArray, expected: FloatArray) -> float:
    if actual.shape != expected.shape:
        return float("inf")
    if actual.size == 0:
        return 0.0
    return float(np.max(np.abs(actual - expected)))


def _exact_delay_contract(
    signal: FloatArray,
    embedded: FloatArray,
    *,
    delay: int,
    dimension: int,
) -> float:
    t_effective = signal.size - (dimension - 1) * delay
    indices = np.arange(dimension, dtype=np.int64) * delay
    rows = np.arange(t_effective, dtype=np.int64)[:, np.newaxis] + indices
    expected = signal[rows]
    return _max_abs_error(embedded, expected)


def _reference_contracts(signal: FloatArray) -> dict[str, float | int]:
    delay = 3
    dimension = 4
    n_bins = 24
    embedded = _python_delay(signal, delay=delay, dimension=dimension)
    shifted = _python_delay(signal[5:], delay=delay, dimension=dimension)
    line = np.arange(8, dtype=np.float64).reshape(8, 1)
    line_distances, line_indices = _python_nn(line)
    constant = np.full(96, 0.25, dtype=np.float64)
    zero_lag_mi = _python_mi(signal, lag=0, n_bins=n_bins)
    distant_lag_mi = _python_mi(signal, lag=signal.size // 3, n_bins=n_bins)
    constant_mi = _python_mi(constant, lag=5, n_bins=12)

    line_expected_distances = np.ones(8, dtype=np.float64)
    first_row_shift_error = _max_abs_error(shifted[0], embedded[5])
    line_index_contract = int(line_indices[0] == 1 and line_indices[-1] == 6)
    finite_components = int(
        np.all(np.isfinite(embedded))
        and np.all(np.isfinite(line_distances))
        and np.all(np.isfinite(line_indices.astype(np.float64)))
        and np.isfinite(zero_lag_mi)
        and np.isfinite(distant_lag_mi)
        and np.isfinite(constant_mi)
    )
    return {
        "exact_indexing_max_abs_error": _exact_delay_contract(
            signal,
            embedded,
            delay=delay,
            dimension=dimension,
        ),
        "time_shift_row_max_abs_error": first_row_shift_error,
        "nearest_neighbor_line_distance_error": _max_abs_error(
            line_distances,
            line_expected_distances,
        ),
        "nearest_neighbor_line_index_contract": line_index_contract,
        "constant_signal_mi_abs_error": abs(constant_mi),
        "zero_lag_mi_exceeds_distant_lag": int(zero_lag_mi > distant_lag_mi),
        "finite_components": finite_components,
    }


def _contracts_passed(contracts: dict[str, float | int]) -> bool:
    if int(contracts["finite_components"]) != 1:
        return False
    if int(contracts["nearest_neighbor_line_index_contract"]) != 1:
        return False
    if int(contracts["zero_lag_mi_exceeds_distant_lag"]) != 1:
        return False
    return all(
        float(value) <= CONTRACT_TOLERANCE
        for key, value in contracts.items()
        if key
        not in {
            "finite_components",
            "nearest_neighbor_line_index_contract",
            "zero_lag_mi_exceeds_distant_lag",
        }
    )


def _load_backend(backend: str) -> dict[str, object] | None:
    if backend == "python":
        return None
    return embedding_module._load_backend(backend)  # noqa: SLF001


def _call_direct_delay(
    funcs: dict[str, object] | None,
    signal: FloatArray,
    delay: int,
    dimension: int,
) -> FloatArray:
    if funcs is None:
        return _python_delay(signal, delay=delay, dimension=dimension)
    fn = funcs["de"]
    if not callable(fn):
        raise RuntimeError("delay embedding primitive is not callable")
    raw = np.asarray(fn(signal, delay, dimension), dtype=np.float64)
    t_effective = signal.size - (dimension - 1) * delay
    if raw.shape == (t_effective * dimension,):
        raw = raw.reshape(t_effective, dimension)
    return np.ascontiguousarray(raw, dtype=np.float64)


def _call_direct_mi(
    funcs: dict[str, object] | None,
    signal: FloatArray,
    lag: int,
    n_bins: int,
) -> float | None:
    if funcs is None:
        return _python_mi(signal, lag=lag, n_bins=n_bins)
    fn = funcs.get("mi")
    if not callable(fn):
        return None
    return float(fn(signal, lag, n_bins))


def _call_direct_nn(
    funcs: dict[str, object] | None,
    embedded: FloatArray,
) -> tuple[FloatArray, IntArray] | None:
    if funcs is None:
        return _python_nn(embedded)
    fn = funcs.get("nn")
    if not callable(fn):
        return None
    distances, indices = fn(
        np.ascontiguousarray(embedded.ravel(), dtype=np.float64),
        embedded.shape[0],
        embedded.shape[1],
    )
    return (
        np.asarray(distances, dtype=np.float64),
        np.asarray(indices, dtype=np.int64),
    )


def _backend_record(
    backend: str,
    signal: FloatArray,
    *,
    delay: int,
    dimension: int,
    lag: int,
    n_bins: int,
    calls: int,
    reference_delay: FloatArray,
    reference_mi: float,
    reference_nn: tuple[FloatArray, IntArray],
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "backend": backend,
        "status": "unavailable",
        "delay_parity_passed": False,
        "mi_parity_passed": False,
        "nn_parity_passed": False,
        "mi_supported": False,
        "nn_supported": False,
        "public_dispatch_parity_passed": False,
        "contracts_passed": False,
        "delay_max_abs_error": None,
        "mi_abs_error": None,
        "nn_distance_max_abs_error": None,
        "nn_index_exact": False,
        "delay_sha256": None,
        "mi_sha256": None,
        "nn_distance_sha256": None,
        "nn_index_sha256": None,
        "ms_per_call": None,
        "unavailable_reason": "",
    }
    try:
        funcs = _load_backend(backend)
    except (ImportError, ModuleNotFoundError, OSError, RuntimeError) as exc:
        reason = str(exc).strip() or exc.__class__.__name__
        base["unavailable_reason"] = (
            f"{backend} backend unavailable for monitor.embedding: {reason}"
        )
        return base

    elapsed = 0.0
    direct_delay: FloatArray | None = None
    direct_mi: float | None = None
    direct_nn: tuple[FloatArray, IntArray] | None = None
    try:
        t0 = time.perf_counter()
        for _ in range(calls):
            direct_delay = _call_direct_delay(funcs, signal, delay, dimension)
            direct_mi = _call_direct_mi(funcs, signal, lag, n_bins)
            direct_nn = _call_direct_nn(funcs, reference_delay)
        elapsed = time.perf_counter() - t0
    except (TypeError, ValueError, RuntimeError) as exc:
        base["status"] = "invalid"
        base["unavailable_reason"] = (
            f"{backend} backend returned invalid embedding output: {exc}"
        )
        return base

    if direct_delay is None:
        raise RuntimeError("delay embedding benchmark did not run")
    delay_error = _max_abs_error(direct_delay, reference_delay)
    mi_error = None if direct_mi is None else abs(direct_mi - reference_mi)
    nn_distance_error = None
    nn_index_exact = False
    if direct_nn is not None:
        nn_distance_error = _max_abs_error(direct_nn[0], reference_nn[0])
        nn_index_exact = bool(np.array_equal(direct_nn[1], reference_nn[1]))

    previous = _force_backend(backend)
    try:
        public_delay = delay_embed(signal, delay=delay, dimension=dimension)
        public_mi = mutual_information(signal, lag=lag, n_bins=n_bins)
        public_nn = nearest_neighbor_distances(reference_delay)
    finally:
        _restore_backend(previous)
    public_dispatch_parity = (
        _max_abs_error(public_delay, reference_delay) <= DELAY_TOLERANCE
        and abs(public_mi - reference_mi) <= MI_TOLERANCE
        and _max_abs_error(public_nn[0], reference_nn[0]) <= NN_TOLERANCE
        and np.array_equal(public_nn[1], reference_nn[1])
    )
    contracts = _reference_contracts(signal)
    mi_supported = direct_mi is not None
    nn_supported = direct_nn is not None
    delay_parity = delay_error <= DELAY_TOLERANCE
    mi_parity = (not mi_supported) or (
        mi_error is not None and mi_error <= MI_TOLERANCE
    )
    nn_parity = (not nn_supported) or (
        nn_distance_error is not None
        and nn_distance_error <= NN_TOLERANCE
        and nn_index_exact
    )
    contracts_passed = _contracts_passed(contracts)
    base.update(
        {
            "status": "available",
            "delay_parity_passed": delay_parity,
            "mi_parity_passed": mi_parity,
            "nn_parity_passed": nn_parity,
            "mi_supported": mi_supported,
            "nn_supported": nn_supported,
            "public_dispatch_parity_passed": public_dispatch_parity,
            "contracts_passed": contracts_passed,
            "delay_max_abs_error": delay_error,
            "mi_abs_error": mi_error,
            "nn_distance_max_abs_error": nn_distance_error,
            "nn_index_exact": nn_index_exact,
            "delay_sha256": _array_sha256(direct_delay),
            "mi_sha256": None
            if direct_mi is None
            else hashlib.sha256(repr(float(direct_mi)).encode("ascii")).hexdigest(),
            "nn_distance_sha256": None
            if direct_nn is None
            else _array_sha256(direct_nn[0]),
            "nn_index_sha256": None
            if direct_nn is None
            else _array_sha256(direct_nn[1]),
            "ms_per_call": (elapsed / calls) * 1000.0,
        }
    )
    return base


def _deterministic_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _clear_backend_cache() -> None:
    embedding_module._BACKEND_CACHE.clear()  # noqa: SLF001


def benchmark_embedding_polyglot_parity_gate(
    *,
    n: int = 160,
    calls: int = 1,
    seed: int = 2026,
) -> dict[str, float | int | str]:
    """Run the Takens delay-embedding polyglot parity gate."""
    n = _validate_int_control(n, name="n", minimum=64)
    calls = _validate_int_control(calls, name="calls", minimum=1)
    seed = _validate_int_control(seed, name="seed", minimum=0)
    delay = 3
    dimension = 4
    lag = 5
    n_bins = 24
    signal = _problem(n, seed)
    reference_delay = _python_delay(signal, delay=delay, dimension=dimension)
    reference_mi = _python_mi(signal, lag=lag, n_bins=n_bins)
    reference_nn = _python_nn(reference_delay)
    reference_contracts = _reference_contracts(signal)

    _clear_backend_cache()
    t0 = time.perf_counter()
    records = [
        _backend_record(
            backend,
            signal,
            delay=delay,
            dimension=dimension,
            lag=lag,
            n_bins=n_bins,
            calls=calls,
            reference_delay=reference_delay,
            reference_mi=reference_mi,
            reference_nn=reference_nn,
        )
        for backend in BACKEND_ORDER
    ]
    elapsed = time.perf_counter() - t0

    available_records = [
        record for record in records if record["status"] == "available"
    ]
    unavailable_records = [
        record for record in records if record["status"] == "unavailable"
    ]
    invalid_records = [record for record in records if record["status"] == "invalid"]
    all_available_passed = all(
        record["delay_parity_passed"] is True
        and record["mi_parity_passed"] is True
        and record["nn_parity_passed"] is True
        and record["public_dispatch_parity_passed"] is True
        and record["contracts_passed"] is True
        for record in available_records
    )
    python_reference_present = any(
        record["backend"] == "python" and record["status"] == "available"
        for record in records
    )
    reference_contracts_passed = _contracts_passed(reference_contracts)
    acceptance_passed = (
        python_reference_present
        and reference_contracts_passed
        and all_available_passed
        and not invalid_records
    )
    deterministic_payload = {
        "suite": "embedding_polyglot_parity_gate",
        "backend_records": [
            {
                key: value
                for key, value in record.items()
                if key not in {"ms_per_call", "unavailable_reason"}
            }
            for record in records
        ],
        "reference_contracts": reference_contracts,
        "reference_delay_sha256": _array_sha256(reference_delay),
        "reference_nn_distance_sha256": _array_sha256(reference_nn[0]),
        "reference_nn_index_sha256": _array_sha256(reference_nn[1]),
        "reference_mi": reference_mi,
    }
    thresholds = {
        "backend_order": list(BACKEND_ORDER),
        "delay_tolerance": DELAY_TOLERANCE,
        "mi_tolerance": MI_TOLERANCE,
        "nn_tolerance": NN_TOLERANCE,
        "max_reference_contract_abs_error": CONTRACT_TOLERANCE,
        "require_python_reference_present": True,
        "require_all_available_backends_pass": True,
        "require_exact_delay_indexing": True,
        "require_time_shift_row_consistency": True,
        "require_non_negative_mutual_information": True,
        "require_constant_signal_zero_mutual_information": True,
        "require_zero_lag_mi_exceeds_distant_lag": True,
        "require_nearest_neighbor_self_exclusion": True,
        "require_public_dispatch_parity": True,
        "production_timing_claim": False,
    }
    return {
        "suite": "embedding_polyglot_parity_gate",
        "n": n,
        "calls": calls,
        "seed": seed,
        "delay": delay,
        "dimension": dimension,
        "lag": lag,
        "n_bins": n_bins,
        "backend_count": len(BACKEND_ORDER),
        "available_backend_count": len(available_records),
        "unavailable_backend_count": len(unavailable_records),
        "invalid_backend_count": len(invalid_records),
        "delay_parity_passed_count": sum(
            1 for record in available_records if record["delay_parity_passed"] is True
        ),
        "mi_parity_passed_count": sum(
            1 for record in available_records if record["mi_parity_passed"] is True
        ),
        "nn_parity_passed_count": sum(
            1 for record in available_records if record["nn_parity_passed"] is True
        ),
        "python_reference_present": int(python_reference_present),
        "all_available_passed": int(all_available_passed),
        "reference_contracts_passed": int(reference_contracts_passed),
        "acceptance_passed": int(acceptance_passed),
        "reference_delay_sha256": _array_sha256(reference_delay),
        "reference_mi": reference_mi,
        "reference_nn_distance_sha256": _array_sha256(reference_nn[0]),
        "reference_nn_index_sha256": _array_sha256(reference_nn[1]),
        "exact_indexing_max_abs_error": float(
            reference_contracts["exact_indexing_max_abs_error"]
        ),
        "time_shift_row_max_abs_error": float(
            reference_contracts["time_shift_row_max_abs_error"]
        ),
        "nearest_neighbor_line_distance_error": float(
            reference_contracts["nearest_neighbor_line_distance_error"]
        ),
        "nearest_neighbor_line_index_contract": int(
            reference_contracts["nearest_neighbor_line_index_contract"]
        ),
        "constant_signal_mi_abs_error": float(
            reference_contracts["constant_signal_mi_abs_error"]
        ),
        "zero_lag_mi_exceeds_distant_lag": int(
            reference_contracts["zero_lag_mi_exceeds_distant_lag"]
        ),
        "backend_records_json": json.dumps(records, sort_keys=True),
        "acceptance_thresholds_json": json.dumps(thresholds, sort_keys=True),
        "benchmark_sha256": _deterministic_hash(deterministic_payload),
        "benchmark_evidence_kind": BENCHMARK_EVIDENCE_KIND,
        "isolation_method": "none",
        "production_timing_claim": 0,
        "wall_time_s": elapsed,
        "steps_per_second": (n * calls * max(len(available_records), 1))
        / max(elapsed, 1.0e-12),
    }


def _bench(n: int, calls: int, *, seed: int = 2026) -> float:
    signal = _problem(n, seed)
    t0 = time.perf_counter()
    for _ in range(calls):
        delay_embed(signal, delay=3, dimension=4)
    return time.perf_counter() - t0


def bench_at(
    sizes: Sequence[int] = (128, 512, 2048),
    calls: int = 100,
    *,
    seed: int = 2026,
) -> list[dict[str, float | int | str]]:
    calls = _validate_int_control(calls, name="calls", minimum=1)
    seed = _validate_int_control(seed, name="seed", minimum=0)
    records: list[dict[str, float | int | str]] = []
    for size in sizes:
        n = _validate_int_control(int(size), name="size", minimum=64)
        elapsed = _bench(n, calls, seed=seed)
        records.append(
            {
                "suite": "embedding_wallclock_local",
                "n": n,
                "calls": calls,
                "seconds": elapsed,
                "ms_per_call": (elapsed / calls) * 1000.0,
                "steps_per_second": calls / max(elapsed, 1.0e-12),
                "benchmark_evidence_kind": BENCHMARK_EVIDENCE_KIND,
                "isolation_method": "none",
                "production_timing_claim": 0,
            }
        )
    return records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 512, 2048])
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--parity-gate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.parity_gate:
        payload = benchmark_embedding_polyglot_parity_gate(
            n=int(args.sizes[0]),
            calls=args.calls,
            seed=args.seed,
        )
    else:
        payload = {"benchmarks": bench_at(args.sizes, args.calls, seed=args.seed)}
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
