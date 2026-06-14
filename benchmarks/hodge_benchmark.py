# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Combinatorial Hodge decomposition benchmark gate

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
IntArray = NDArray[np.int64]
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
# A non-trivial harmonic component must dominate this floor on the
# triangle-free cycle so the topological content is demonstrably present.
HARMONIC_PRESENCE_FLOOR = 1.0e-3
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


def _complex(knm: FloatArray, phases: FloatArray) -> tuple[IntArray, IntArray]:
    k_sym = 0.5 * (knm + knm.T)
    return hodge_module._simplicial_complex(  # noqa: SLF001
        k_sym, int(phases.size), None
    )


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


def _antisymmetry_error(component: FloatArray) -> float:
    return float(np.max(np.abs(component + component.T))) if component.size else 0.0


def _flow_inner(a: FloatArray, b: FloatArray) -> float:
    """L² inner product of two antisymmetric flow matrices (each edge once)."""
    if a.size == 0:
        return 0.0
    upper = np.triu_indices(a.shape[0], k=1)
    return float(np.sum(a[upper] * b[upper]))


def _cycle_contracts() -> dict[str, float | int]:
    """Triangle-free 4-cycle: β₁ = 1, curl vanishes, harmonic is non-trivial."""
    n = 4
    knm = np.zeros((n, n), dtype=np.float64)
    for i, j in ((0, 1), (1, 2), (2, 3), (0, 3)):
        knm[i, j] = 1.0
        knm[j, i] = 1.0
    phases = np.array([0.0, 0.7, 1.9, 2.8], dtype=np.float64)
    result = hodge_decomposition(knm, phases)
    return {
        "cycle_curl_max_abs_error": float(np.max(np.abs(result.curl))),
        "cycle_harmonic_max_abs": float(np.max(np.abs(result.harmonic))),
        "cycle_betti_one": int(result.betti_one),
    }


def _reference_contracts(
    knm: FloatArray,
    phases: FloatArray,
    reference: HodgeTuple,
    flow: FloatArray,
) -> dict[str, float | int]:
    gradient, curl, harmonic = reference
    reconstruction = gradient + curl + harmonic
    shifted = hodge_module._python_decomposition(knm, phases + 1.2345)  # noqa: SLF001
    scale = 2.5
    scaled = hodge_module._python_decomposition(scale * knm, phases)  # noqa: SLF001

    # Divergence of an antisymmetric flow at node i is the row sum.
    curl_divergence = float(np.max(np.abs(curl.sum(axis=1)))) if curl.size else 0.0
    harmonic_divergence = (
        float(np.max(np.abs(harmonic.sum(axis=1)))) if harmonic.size else 0.0
    )
    orthogonality = max(
        abs(_flow_inner(gradient, curl)),
        abs(_flow_inner(gradient, harmonic)),
        abs(_flow_inner(curl, harmonic)),
    )
    antisymmetry = max(
        _antisymmetry_error(gradient),
        _antisymmetry_error(curl),
        _antisymmetry_error(harmonic),
    )
    finite_components = all(np.all(np.isfinite(component)) for component in reference)
    contracts: dict[str, float | int] = {
        "reconstruction_max_abs_error": _max_abs_error(reconstruction, flow),
        "antisymmetry_max_abs_error": antisymmetry,
        "orthogonality_max_abs_error": orthogonality,
        "curl_divergence_max_abs_error": curl_divergence,
        "harmonic_divergence_max_abs_error": harmonic_divergence,
        "phase_shift_max_abs_error": max(
            _max_abs_error(actual, expected)
            for actual, expected in zip(shifted, reference, strict=True)
        ),
        "scale_covariance_max_abs_error": max(
            _max_abs_error(actual, scale * expected)
            for actual, expected in zip(scaled, reference, strict=True)
        ),
        "finite_components": int(finite_components),
    }
    contracts.update(_cycle_contracts())
    return contracts


def _contracts_passed(contracts: dict[str, float | int]) -> bool:
    if int(contracts["finite_components"]) != 1:
        return False
    if int(contracts["cycle_betti_one"]) != 1:
        return False
    if float(contracts["cycle_harmonic_max_abs"]) <= HARMONIC_PRESENCE_FLOOR:
        return False
    bounded_keys = (
        "reconstruction_max_abs_error",
        "antisymmetry_max_abs_error",
        "orthogonality_max_abs_error",
        "curl_divergence_max_abs_error",
        "harmonic_divergence_max_abs_error",
        "phase_shift_max_abs_error",
        "scale_covariance_max_abs_error",
        "cycle_curl_max_abs_error",
    )
    return all(float(contracts[key]) <= REFERENCE_TOLERANCE for key in bounded_keys)


def _unavailable_reason(backend: str, exc: BaseException) -> str:
    reason = str(exc).strip() or exc.__class__.__name__
    return f"{backend} backend unavailable for coupling.hodge: {reason}"


def _run_backend_once(
    backend: str,
    knm: FloatArray,
    phases: FloatArray,
    edges: IntArray,
    triangles: IntArray,
) -> HodgeTuple:
    if backend == "python":
        return hodge_module._python_decomposition(knm, phases)  # noqa: SLF001
    backend_fn = hodge_module._load_backend(backend)  # noqa: SLF001
    raw = backend_fn(
        np.ascontiguousarray(knm.ravel(), dtype=np.float64),
        phases,
        int(phases.size),
        np.ascontiguousarray(edges.ravel(), dtype=np.int64),
        int(edges.shape[0]),
        np.ascontiguousarray(triangles.ravel(), dtype=np.int64),
        int(triangles.shape[0]),
    )
    return hodge_module._normalise_backend_output(  # noqa: SLF001
        raw, expected_n=int(phases.size)
    )


def _bench_backend(
    backend: str,
    knm: FloatArray,
    phases: FloatArray,
    edges: IntArray,
    triangles: IntArray,
    calls: int,
) -> tuple[float, HodgeTuple]:
    t0 = time.perf_counter()
    result: HodgeTuple | None = None
    for _ in range(calls):
        result = _run_backend_once(backend, knm, phases, edges, triangles)
    elapsed = time.perf_counter() - t0
    if result is None:
        raise RuntimeError("benchmark did not produce a Hodge result")
    return elapsed, result


def _backend_record(
    backend: str,
    knm: FloatArray,
    phases: FloatArray,
    edges: IntArray,
    triangles: IntArray,
    reference: HodgeTuple,
    flow: FloatArray,
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
        elapsed, result = _bench_backend(
            backend, knm, phases, edges, triangles, calls
        )
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
    contracts = _reference_contracts(knm, phases, result, flow)
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
    """Run the Hodge polyglot parity and combinatorial-invariant gate."""
    n = _validate_int_control(n, name="n", minimum=2)
    calls = _validate_int_control(calls, name="calls", minimum=1)
    seed = _validate_int_control(seed, name="seed", minimum=0)
    knm, phases = _problem(n, seed)
    edges, triangles = _complex(knm, phases)
    reference = hodge_module._python_decomposition(knm, phases)  # noqa: SLF001
    flow = hodge_decomposition(knm, phases).flow
    reference_contracts = _reference_contracts(knm, phases, reference, flow)

    _clear_backend_cache()
    t0 = time.perf_counter()
    records = [
        _backend_record(
            backend, knm, phases, edges, triangles, reference, flow, calls
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
        "harmonic_presence_floor": HARMONIC_PRESENCE_FLOOR,
        "require_python_reference_present": True,
        "require_all_available_backends_pass": True,
        "require_reconstruction_contract": True,
        "require_antisymmetric_flow_components": True,
        "require_component_orthogonality": True,
        "require_divergence_free_curl_and_harmonic": True,
        "require_global_phase_shift_invariance": True,
        "require_scale_covariance": True,
        "require_cycle_betti_one": True,
        "require_nontrivial_cycle_harmonic": True,
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
        "antisymmetry_max_abs_error": float(
            reference_contracts["antisymmetry_max_abs_error"]
        ),
        "orthogonality_max_abs_error": float(
            reference_contracts["orthogonality_max_abs_error"]
        ),
        "curl_divergence_max_abs_error": float(
            reference_contracts["curl_divergence_max_abs_error"]
        ),
        "harmonic_divergence_max_abs_error": float(
            reference_contracts["harmonic_divergence_max_abs_error"]
        ),
        "phase_shift_max_abs_error": float(
            reference_contracts["phase_shift_max_abs_error"]
        ),
        "scale_covariance_max_abs_error": float(
            reference_contracts["scale_covariance_max_abs_error"]
        ),
        "cycle_curl_max_abs_error": float(
            reference_contracts["cycle_curl_max_abs_error"]
        ),
        "cycle_harmonic_max_abs": float(reference_contracts["cycle_harmonic_max_abs"]),
        "cycle_betti_one": int(reference_contracts["cycle_betti_one"]),
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
