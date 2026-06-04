# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hodge decomposition benchmark gate

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.coupling import hodge as hodge_module
from scpn_phase_orchestrator.coupling.hodge import hodge_decomposition

FloatArray = NDArray[np.float64]
HodgeTuple = tuple[FloatArray, FloatArray, FloatArray]

BACKEND_ORDER = ("rust", "mojo", "julia", "go", "python")
PARITY_TOLERANCES = {
    "rust": 1.0e-10,
    "mojo": 1.0e-8,
    "julia": 1.0e-10,
    "go": 1.0e-10,
    "python": 0.0,
}
REFERENCE_TOLERANCE = 1.0e-10
BENCHMARK_EVIDENCE_KIND = "local_regression_non_isolated"


def _validate_int_control(value: int, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _problem(n: int, seed: int) -> tuple[FloatArray, FloatArray]:
    n = _validate_int_control(n, name="n", minimum=2)
    seed = _validate_int_control(seed, name="seed", minimum=0)
    rng = np.random.default_rng(seed)
    raw = rng.normal(loc=0.0, scale=0.55, size=(n, n))
    structured = 0.62 * ((raw + raw.T) / 2.0) + 0.38 * ((raw - raw.T) / 2.0)
    np.fill_diagonal(structured, 0.0)
    phases = rng.uniform(-np.pi, np.pi, size=n)
    return structured.astype(np.float64), phases.astype(np.float64)


def _total_flow(knm: FloatArray, phases: FloatArray) -> FloatArray:
    phase_diff = phases[np.newaxis, :] - phases[:, np.newaxis]
    return np.sum(knm * np.cos(phase_diff), axis=1, dtype=np.float64)


def _array_sha256(value: FloatArray) -> str:
    array = np.ascontiguousarray(value, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _tuple_sha256(value: HodgeTuple) -> str:
    digest = hashlib.sha256()
    for component in value:
        digest.update(_array_sha256(component).encode("ascii"))
    return digest.hexdigest()


def _max_abs_error(actual: FloatArray, expected: FloatArray) -> float:
    if actual.shape != expected.shape:
        return float("inf")
    if actual.size == 0:
        return 0.0
    return float(np.max(np.abs(actual - expected)))


def _tuple_component_errors(
    actual: HodgeTuple, expected: HodgeTuple
) -> dict[str, float]:
    return {
        "gradient_abs_error": _max_abs_error(actual[0], expected[0]),
        "curl_abs_error": _max_abs_error(actual[1], expected[1]),
        "harmonic_abs_error": _max_abs_error(actual[2], expected[2]),
    }


def _reference_contracts(
    knm: FloatArray, phases: FloatArray, reference: HodgeTuple
) -> dict[str, float | int]:
    total = _total_flow(knm, phases)
    reconstruction = reference[0] + reference[1] + reference[2]
    shifted = hodge_module._python_decomposition(knm, phases + 1.2345)  # noqa: SLF001
    scale = 2.5
    scaled = hodge_module._python_decomposition(scale * knm, phases)  # noqa: SLF001

    symmetric = (knm + knm.T) / 2.0
    antisymmetric = (knm - knm.T) / 2.0
    symmetric_result = hodge_module._python_decomposition(symmetric, phases)  # noqa: SLF001
    antisymmetric_result = hodge_module._python_decomposition(  # noqa: SLF001
        antisymmetric,
        phases,
    )

    two_node_weight = 0.7
    two_node_phases = np.array([0.3, 1.1], dtype=np.float64)
    two_node_knm = np.array(
        [[0.0, two_node_weight], [-two_node_weight, 0.0]],
        dtype=np.float64,
    )
    two_node = hodge_module._python_decomposition(two_node_knm, two_node_phases)  # noqa: SLF001
    two_node_expected_curl = np.array(
        [
            two_node_weight * np.cos(two_node_phases[1] - two_node_phases[0]),
            -two_node_weight * np.cos(two_node_phases[1] - two_node_phases[0]),
        ],
        dtype=np.float64,
    )

    finite_components = all(np.all(np.isfinite(component)) for component in reference)
    return {
        "reconstruction_max_abs_error": _max_abs_error(reconstruction, total),
        "harmonic_max_abs_error": float(np.max(np.abs(reference[2]))),
        "phase_shift_max_abs_error": max(
            _max_abs_error(actual, expected)
            for actual, expected in zip(shifted, reference, strict=True)
        ),
        "symmetric_curl_max_abs_error": float(np.max(np.abs(symmetric_result[1]))),
        "antisymmetric_gradient_max_abs_error": float(
            np.max(np.abs(antisymmetric_result[0]))
        ),
        "antisymmetric_curl_sum_abs_error": float(abs(np.sum(antisymmetric_result[1]))),
        "two_node_gradient_max_abs_error": float(np.max(np.abs(two_node[0]))),
        "two_node_curl_max_abs_error": _max_abs_error(
            two_node[1],
            two_node_expected_curl,
        ),
        "two_node_harmonic_max_abs_error": float(np.max(np.abs(two_node[2]))),
        "scale_covariance_max_abs_error": max(
            _max_abs_error(actual, scale * expected)
            for actual, expected in zip(scaled, reference, strict=True)
        ),
        "finite_components": int(finite_components),
    }


def _contracts_passed(contracts: dict[str, float | int]) -> bool:
    if int(contracts["finite_components"]) != 1:
        return False
    return all(
        float(value) <= REFERENCE_TOLERANCE
        for key, value in contracts.items()
        if key != "finite_components"
    )


def _unavailable_reason(backend: str, exc: BaseException) -> str:
    reason = str(exc).strip() or exc.__class__.__name__
    return f"{backend} backend unavailable for coupling.hodge: {reason}"


def _run_backend_once(backend: str, knm: FloatArray, phases: FloatArray) -> HodgeTuple:
    if backend == "python":
        return hodge_module._python_decomposition(knm, phases)  # noqa: SLF001
    backend_fn = hodge_module._load_backend(backend)  # noqa: SLF001
    raw = backend_fn(
        np.ascontiguousarray(knm.ravel(), dtype=np.float64), phases, phases.size
    )
    return hodge_module._normalise_backend_output(raw, expected_n=phases.size)  # noqa: SLF001


def _bench_backend(
    backend: str,
    knm: FloatArray,
    phases: FloatArray,
    calls: int,
) -> tuple[float, HodgeTuple]:
    t0 = time.perf_counter()
    result: HodgeTuple | None = None
    for _ in range(calls):
        result = _run_backend_once(backend, knm, phases)
    elapsed = time.perf_counter() - t0
    if result is None:
        raise RuntimeError("benchmark did not produce a Hodge result")
    return elapsed, result


def _backend_record(
    backend: str,
    knm: FloatArray,
    phases: FloatArray,
    reference: HodgeTuple,
    calls: int,
) -> dict[str, Any]:
    tolerance = PARITY_TOLERANCES[backend]
    base: dict[str, Any] = {
        "backend": backend,
        "status": "unavailable",
        "tolerance": tolerance,
        "parity_passed": False,
        "reference_contracts_passed": False,
        "gradient_abs_error": None,
        "curl_abs_error": None,
        "harmonic_abs_error": None,
        "max_abs_error": None,
        "gradient_sha256": None,
        "curl_sha256": None,
        "harmonic_sha256": None,
        "bundle_sha256": None,
        "ms_per_call": None,
        "unavailable_reason": "",
    }
    try:
        elapsed, result = _bench_backend(backend, knm, phases, calls)
    except (ImportError, ModuleNotFoundError, OSError, RuntimeError) as exc:
        base["unavailable_reason"] = _unavailable_reason(backend, exc)
        return base
    except (TypeError, ValueError) as exc:
        base["status"] = "invalid"
        base["unavailable_reason"] = (
            f"{backend} backend returned invalid Hodge output: {exc}"
        )
        return base

    errors = _tuple_component_errors(result, reference)
    max_error = max(errors.values())
    contracts = _reference_contracts(knm, phases, result)
    parity_passed = max_error <= tolerance and _contracts_passed(contracts)
    base.update(
        {
            "status": "available",
            "parity_passed": parity_passed,
            "reference_contracts_passed": _contracts_passed(contracts),
            "gradient_abs_error": errors["gradient_abs_error"],
            "curl_abs_error": errors["curl_abs_error"],
            "harmonic_abs_error": errors["harmonic_abs_error"],
            "max_abs_error": max_error,
            "gradient_sha256": _array_sha256(result[0]),
            "curl_sha256": _array_sha256(result[1]),
            "harmonic_sha256": _array_sha256(result[2]),
            "bundle_sha256": _tuple_sha256(result),
            "ms_per_call": (elapsed / calls) * 1000.0,
        }
    )
    return base


def _deterministic_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _clear_backend_cache() -> None:
    hodge_module._BACKEND_CACHE.clear()  # noqa: SLF001


def benchmark_hodge_polyglot_parity_gate(
    *,
    n: int = 10,
    calls: int = 1,
    seed: int = 2026,
) -> dict[str, float | int | str]:
    """Run the Hodge polyglot parity and mathematical-invariant gate."""
    n = _validate_int_control(n, name="n", minimum=2)
    calls = _validate_int_control(calls, name="calls", minimum=1)
    seed = _validate_int_control(seed, name="seed", minimum=0)
    knm, phases = _problem(n, seed)
    reference = hodge_module._python_decomposition(knm, phases)  # noqa: SLF001
    reference_contracts = _reference_contracts(knm, phases, reference)

    _clear_backend_cache()
    t0 = time.perf_counter()
    records = [
        _backend_record(backend, knm, phases, reference, calls)
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
        record["parity_passed"] is True for record in available_records
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
        "suite": "hodge_polyglot_parity_gate",
        "backend_records": [
            {
                key: value
                for key, value in record.items()
                if key not in {"ms_per_call", "unavailable_reason"}
            }
            for record in records
        ],
        "reference_contracts": reference_contracts,
        "reference_sha256": _tuple_sha256(reference),
    }
    thresholds = {
        "backend_order": list(BACKEND_ORDER),
        "parity_tolerances": PARITY_TOLERANCES,
        "max_reference_contract_abs_error": REFERENCE_TOLERANCE,
        "require_python_reference_present": True,
        "require_all_available_backends_pass": True,
        "require_reconstruction_contract": True,
        "require_global_phase_shift_invariance": True,
        "require_symmetric_zero_curl": True,
        "require_antisymmetric_zero_gradient": True,
        "require_two_node_antisymmetric_closed_form": True,
        "require_scale_covariance": True,
        "require_no_invalid_backend_outputs": True,
        "production_timing_claim": False,
    }
    return {
        "suite": "hodge_polyglot_parity_gate",
        "n": n,
        "calls": calls,
        "seed": seed,
        "backend_count": len(BACKEND_ORDER),
        "available_backend_count": len(available_records),
        "unavailable_backend_count": len(unavailable_records),
        "invalid_backend_count": len(invalid_records),
        "parity_passed_count": sum(
            1 for record in available_records if record["parity_passed"] is True
        ),
        "python_reference_present": int(python_reference_present),
        "all_available_passed": int(all_available_passed),
        "reference_contracts_passed": int(reference_contracts_passed),
        "acceptance_passed": int(acceptance_passed),
        "reference_gradient_sha256": _array_sha256(reference[0]),
        "reference_curl_sha256": _array_sha256(reference[1]),
        "reference_harmonic_sha256": _array_sha256(reference[2]),
        "reference_bundle_sha256": _tuple_sha256(reference),
        "reconstruction_max_abs_error": float(
            reference_contracts["reconstruction_max_abs_error"]
        ),
        "harmonic_max_abs_error": float(reference_contracts["harmonic_max_abs_error"]),
        "phase_shift_max_abs_error": float(
            reference_contracts["phase_shift_max_abs_error"]
        ),
        "symmetric_curl_max_abs_error": float(
            reference_contracts["symmetric_curl_max_abs_error"]
        ),
        "antisymmetric_gradient_max_abs_error": float(
            reference_contracts["antisymmetric_gradient_max_abs_error"]
        ),
        "antisymmetric_curl_sum_abs_error": float(
            reference_contracts["antisymmetric_curl_sum_abs_error"]
        ),
        "two_node_curl_max_abs_error": float(
            reference_contracts["two_node_curl_max_abs_error"]
        ),
        "scale_covariance_max_abs_error": float(
            reference_contracts["scale_covariance_max_abs_error"]
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
    knm, phases = _problem(n, seed)
    t0 = time.perf_counter()
    for _ in range(calls):
        hodge_decomposition(knm, phases)
    return time.perf_counter() - t0


def bench_at(
    sizes: Sequence[int] = (64, 256, 1024),
    calls: int = 100,
    *,
    seed: int = 2026,
) -> list[dict[str, float | int | str]]:
    calls = _validate_int_control(calls, name="calls", minimum=1)
    seed = _validate_int_control(seed, name="seed", minimum=0)
    records: list[dict[str, float | int | str]] = []
    for size in sizes:
        n = _validate_int_control(int(size), name="size", minimum=2)
        elapsed = _bench(n, calls, seed=seed)
        records.append(
            {
                "suite": "hodge_wallclock_local",
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
    parser.add_argument("--sizes", nargs="+", type=int, default=[64, 256, 1024])
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--parity-gate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.parity_gate:
        payload = benchmark_hodge_polyglot_parity_gate(
            n=int(args.sizes[0]),
            calls=args.calls,
            seed=args.seed,
        )
    else:
        payload = {"benchmarks": bench_at(args.sizes, args.calls, seed=args.seed)}
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
