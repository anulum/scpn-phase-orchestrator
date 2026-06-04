# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Spectral eigendecomposition benchmark

"""Per-backend wall-clock benchmark for
``coupling.spectral.fiedler_value`` + ``fiedler_vector``."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.coupling import spectral as s_mod
from scpn_phase_orchestrator.coupling.spectral import (
    ACTIVE_BACKEND,
    AVAILABLE_BACKENDS,
    fiedler_value,
    fiedler_vector,
    graph_laplacian,
    spectral_gap,
)

BACKEND_ORDER = ("rust", "mojo", "julia", "go", "python")
PARITY_TOLERANCES = {
    "rust": 1.0e-8,
    "mojo": 1.0e-8,
    "julia": 1.0e-10,
    "go": 1.0e-10,
    "python": 0.0,
}


def _bench(backend: str, W, calls: int) -> float:
    elapsed, _, _, _ = _bench_with_outputs(backend, W, calls)
    return elapsed


def _clear_backend_caches() -> None:
    s_mod._PRIM_CACHE = None
    s_mod._RUST_CACHE = None


def _bench_with_outputs(
    backend: str,
    W: NDArray[np.floating],
    calls: int,
) -> tuple[float, float, NDArray[np.float64], float]:
    prev = s_mod.ACTIVE_BACKEND
    s_mod.ACTIVE_BACKEND = backend
    _clear_backend_caches()
    try:
        lam2 = fiedler_value(W)
        vector = fiedler_vector(W)
        gap = spectral_gap(W)
        t0 = time.perf_counter()
        for _ in range(calls):
            lam2 = fiedler_value(W)
            vector = fiedler_vector(W)
            gap = spectral_gap(W)
        return (
            time.perf_counter() - t0,
            float(lam2),
            np.ascontiguousarray(vector, dtype=np.float64),
            float(gap),
        )
    finally:
        s_mod.ACTIVE_BACKEND = prev
        _clear_backend_caches()


def bench_at(n: int, calls: int) -> dict:
    rng = np.random.default_rng(42)
    W = rng.uniform(0, 1, (n, n))
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0.0)
    row: dict = {"N": n, "calls": calls, "available": AVAILABLE_BACKENDS}
    for backend in AVAILABLE_BACKENDS:
        t = _bench(backend, W, calls)
        row[f"{backend}_ms_per_call"] = (t / calls) * 1000.0
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


def _problem(n: int, seed: int) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    weights = rng.uniform(0.35, 1.35, size=n - 1)
    W = np.zeros((n, n), dtype=np.float64)
    for idx, weight in enumerate(weights):
        W[idx, idx + 1] = weight
        W[idx + 1, idx] = weight
    return W


def _uniform_path_graph(n: int, weight: float) -> NDArray[np.float64]:
    W = np.zeros((n, n), dtype=np.float64)
    for idx in range(n - 1):
        W[idx, idx + 1] = weight
        W[idx + 1, idx] = weight
    return W


def _complete_graph(n: int, weight: float) -> NDArray[np.float64]:
    W = np.full((n, n), weight, dtype=np.float64)
    np.fill_diagonal(W, 0.0)
    return W


def _path_fiedler_value(n: int, weight: float) -> float:
    return float(2.0 * weight * (1.0 - np.cos(np.pi / n)))


def _array_sha256(array: NDArray[np.floating]) -> str:
    contiguous = np.ascontiguousarray(array, dtype=np.float64)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _scalar_sha256(value: float) -> str:
    payload = json.dumps(float(value), separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _backend_status(backend: str) -> tuple[bool, str]:
    if backend == "python" or backend in AVAILABLE_BACKENDS:
        return True, ""
    return False, f"{backend} backend was not resolved by coupling.spectral"


def _unit_vector(vector: NDArray[np.floating]) -> NDArray[np.float64]:
    values = np.ascontiguousarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if norm <= 1.0e-12:
        raise ValueError("Fiedler vector must be non-zero")
    return values / norm


def _align_fiedler_direction(
    vector: NDArray[np.floating],
    reference: NDArray[np.floating],
) -> NDArray[np.float64]:
    candidate = _unit_vector(vector)
    reference_unit = _unit_vector(reference)
    if float(np.dot(candidate, reference_unit)) < 0.0:
        candidate = -candidate
    return np.ascontiguousarray(candidate, dtype=np.float64)


def _laplacian_contracts(
    W: NDArray[np.floating],
    *,
    fiedler: NDArray[np.floating],
    lam2: float,
    gap: float,
) -> dict[str, object]:
    L = graph_laplacian(W)
    eigvals = np.linalg.eigvalsh(L)
    unit_fiedler = _unit_vector(fiedler)
    row_sum_error = float(np.max(np.abs(L.sum(axis=1))))
    symmetry_error = float(np.max(np.abs(L - L.T)))
    psd_floor = float(np.min(eigvals))
    zero_mode_abs = float(abs(eigvals[0]))
    orthogonality_abs = float(abs(np.sum(unit_fiedler)))
    return {
        "row_sum_error": row_sum_error,
        "symmetry_error": symmetry_error,
        "psd_floor": psd_floor,
        "zero_mode_abs": zero_mode_abs,
        "fiedler_orthogonality_abs": orthogonality_abs,
        "contracts_passed": bool(
            row_sum_error <= 1.0e-12
            and symmetry_error <= 1.0e-12
            and psd_floor >= -1.0e-10
            and zero_mode_abs <= 1.0e-10
            and orthogonality_abs <= 1.0e-8
            and lam2 > 0.0
            and gap >= -1.0e-10
        ),
    }


def benchmark_spectral_polyglot_parity_gate(
    *,
    n: int = 10,
    calls: int = 1,
    seed: int = 2026,
) -> dict[str, object]:
    """Record spectral graph parity across every declared backend slot.

    The gate uses a connected weighted path to avoid Fiedler-vector degeneracy.
    Parity is checked on algebraic connectivity, Fiedler-vector direction, and
    spectral gap. Independent reference checks assert the combinatorial
    Laplacian invariants and the exact uniform-path / complete-graph spectra.
    """

    n = _validate_int_control(n, name="n", minimum=3)
    calls = _validate_int_control(calls, name="calls", minimum=1)
    seed = _validate_int_control(seed, name="seed", minimum=0)

    W = _problem(n, seed)
    t0 = time.perf_counter()
    _, reference_lam2, reference_vector, reference_gap = _bench_with_outputs(
        "python",
        W,
        1,
    )
    reference_unit = _unit_vector(reference_vector)
    contracts = _laplacian_contracts(
        W,
        fiedler=reference_vector,
        lam2=reference_lam2,
        gap=reference_gap,
    )

    path_weight = 0.75
    path_n = max(n, 4)
    path_graph = _uniform_path_graph(path_n, path_weight)
    _, path_lam2, _, _ = _bench_with_outputs("python", path_graph, 1)
    analytic_path_lam2 = _path_fiedler_value(path_n, path_weight)
    path_abs_error = abs(path_lam2 - analytic_path_lam2)

    complete_weight = 0.5
    complete_n = max(n, 4)
    complete = _complete_graph(complete_n, complete_weight)
    _, complete_lam2, _, complete_gap = _bench_with_outputs("python", complete, 1)
    analytic_complete_lam2 = complete_n * complete_weight
    complete_abs_error = abs(complete_lam2 - analytic_complete_lam2)

    records: list[dict[str, object]] = []
    parity_pass_count = 0
    parity_checked_count = 0
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
                    "fiedler_value": None,
                    "spectral_gap": None,
                    "fiedler_vector_sha256": None,
                    "fiedler_value_sha256": None,
                    "spectral_gap_sha256": None,
                    "fiedler_value_abs_error": None,
                    "spectral_gap_abs_error": None,
                    "fiedler_direction_abs_error": None,
                    "max_abs_error": None,
                    "tolerance": tolerance,
                    "parity_passed": False,
                    "unavailable_reason": reason,
                }
            )
            continue

        available_backend_count += 1
        elapsed, lam2, vector, gap = _bench_with_outputs(backend, W, calls)
        aligned = _align_fiedler_direction(vector, reference_unit)
        fiedler_value_error = abs(lam2 - reference_lam2)
        spectral_gap_error = abs(gap - reference_gap)
        direction_error = float(np.max(np.abs(aligned - reference_unit)))
        max_abs_error = max(fiedler_value_error, spectral_gap_error, direction_error)
        parity_passed = max_abs_error <= tolerance
        parity_checked_count += 1
        parity_pass_count += int(parity_passed)
        records.append(
            {
                "backend": backend,
                "status": "available",
                "ms_per_call": (elapsed / calls) * 1000.0,
                "fiedler_value": lam2,
                "spectral_gap": gap,
                "fiedler_vector_sha256": _array_sha256(aligned),
                "fiedler_value_sha256": _scalar_sha256(lam2),
                "spectral_gap_sha256": _scalar_sha256(gap),
                "fiedler_value_abs_error": fiedler_value_error,
                "spectral_gap_abs_error": spectral_gap_error,
                "fiedler_direction_abs_error": direction_error,
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
        "require_complete_graph_exact_lambda2": True,
        "require_laplacian_psd_row_sum_contract": True,
        "require_python_reference": True,
        "require_uniform_path_exact_lambda2": True,
    }
    acceptance_passed = (
        len(records) == len(BACKEND_ORDER)
        and any(
            record["backend"] == "python" and record["status"] == "available"
            for record in records
        )
        and parity_pass_count == parity_checked_count
        and bool(contracts["contracts_passed"])
        and path_abs_error <= 1.0e-10
        and complete_abs_error <= 1.0e-10
        and abs(complete_gap) <= 1.0e-10
    )
    benchmark_payload = {
        "n": n,
        "calls": calls,
        "seed": seed,
        "records": records,
        "thresholds": thresholds,
        "reference_fiedler_value": reference_lam2,
        "reference_spectral_gap": reference_gap,
        "reference_fiedler_vector_sha256": _array_sha256(reference_unit),
        "laplacian_contracts": contracts,
        "path_lam2": path_lam2,
        "analytic_path_lam2": analytic_path_lam2,
        "complete_lam2": complete_lam2,
        "analytic_complete_lam2": analytic_complete_lam2,
        "complete_gap": complete_gap,
    }
    benchmark_sha = hashlib.sha256(
        json.dumps(benchmark_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    return {
        "suite": "spectral_polyglot_parity_gate",
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
        "reference_fiedler_value": reference_lam2,
        "reference_spectral_gap": reference_gap,
        "reference_fiedler_vector_sha256": _array_sha256(reference_unit),
        "laplacian_row_sum_error": contracts["row_sum_error"],
        "laplacian_symmetry_error": contracts["symmetry_error"],
        "laplacian_psd_floor": contracts["psd_floor"],
        "laplacian_zero_mode_abs": contracts["zero_mode_abs"],
        "fiedler_orthogonality_abs": contracts["fiedler_orthogonality_abs"],
        "laplacian_contracts_passed": int(bool(contracts["contracts_passed"])),
        "uniform_path_fiedler_value": path_lam2,
        "uniform_path_analytic_fiedler_value": analytic_path_lam2,
        "uniform_path_abs_error": path_abs_error,
        "complete_graph_fiedler_value": complete_lam2,
        "complete_graph_analytic_fiedler_value": analytic_complete_lam2,
        "complete_graph_abs_error": complete_abs_error,
        "complete_graph_spectral_gap": complete_gap,
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
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 64, 128])
    parser.add_argument("--calls", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--parity-gate", action="store_true")
    args = parser.parse_args()

    if args.parity_gate:
        result = benchmark_spectral_polyglot_parity_gate(
            n=args.sizes[0],
            calls=args.calls,
            seed=args.seed,
        )
        payload = json.dumps(result, indent=2, sort_keys=True)
        if args.output:
            args.output.write_text(payload + "\n", encoding="utf-8")
        print(payload)
        return 0

    print(f"Active: {ACTIVE_BACKEND}  Available: {AVAILABLE_BACKENDS}\n")
    header = f"{'N':>4} {'calls':>6}"
    for b in AVAILABLE_BACKENDS:
        header += f" {b + '_ms':>12}"
    print(header)
    print("-" * len(header))
    results: list[dict] = []
    for n in args.sizes:
        row = bench_at(n, args.calls)
        results.append(row)
        line = f"{n:>4} {args.calls:>6}"
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
