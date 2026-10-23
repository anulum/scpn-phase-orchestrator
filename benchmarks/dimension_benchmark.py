# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Fractal-dimension multi-backend benchmark

"""Compare required public dimension runtimes against independent scalar oracles.

Timings include public validation and both CI/KY calls. They are shared-host
regression measurements, with no production speedup or native-kernel-only claim.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from benchmarks._dimension_reference import (
    FloatArray,
    array_sha256,
    output_errors,
    payload_sha256,
    reference_outputs,
)
from scpn_phase_orchestrator.monitor.dimension import (
    correlation_integral,
    kaplan_yorke_dimension,
)

BACKEND_ORDER = ("rust", "mojo", "julia", "go", "python")
PARITY_TOLERANCES = {
    "rust": 1.0e-12,
    "mojo": 1.0e-9,
    "julia": 1.0e-12,
    "go": 1.0e-12,
    "python": 0.0,
}
Report = dict[str, float | int | str | None]


def _validate_int_control(value: object, *, name: str, minimum: int) -> int:
    """Admit true Python/NumPy integers at or above the specified minimum."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _problem(
    t: int, d: int, n_k: int, seed: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Construct a seeded smooth trajectory and a binary-exact KY spectrum."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, 6.0 * np.pi, t, dtype=np.float64)
    basis = []
    for index in range(d):
        frequency = 0.45 + 0.23 * index
        phase = rng.uniform(0.0, 2.0 * np.pi)
        basis.append(
            np.sin(frequency * grid + phase) + 0.35 * np.cos(1.7 * frequency * grid)
        )
    return (
        np.ascontiguousarray(np.column_stack(basis), dtype=np.float64),
        np.logspace(-1.3, 0.45, n_k, dtype=np.float64),
        np.array([0.5, 0.03125, -0.25, -1.0, -2.0], dtype=np.float64),
    )


def _public_outputs(
    backend: str, problem: tuple[FloatArray, FloatArray, FloatArray]
) -> tuple[FloatArray, float]:
    """Call a required public owner over all pairs, preserving dispatch state."""
    trajectory, epsilons, spectrum = problem
    pairs = len(trajectory) * (len(trajectory) - 1) // 2
    return (
        correlation_integral(
            trajectory, epsilons, max_pairs=pairs, seed=0, backend=backend
        ),
        kaplan_yorke_dimension(spectrum, backend=backend),
    )


def _checked_outputs(
    backend: str,
    problem: tuple[FloatArray, FloatArray, FloatArray],
    reference: tuple[FloatArray, float],
) -> tuple[FloatArray, float]:
    """Require every public result to satisfy independent geometry and rational KY."""
    actual = _public_outputs(backend, problem)
    errors = output_errors(actual, reference)
    if (
        float(errors["max_abs_error"]) > PARITY_TOLERANCES[backend]
        or not errors["ci_monotonic"]
        or not errors["ci_unit_interval"]
        or not 0.0 <= actual[1] <= float(problem[2].size)
    ):
        raise ValueError(
            f"{backend} public dimension output disagrees with independent reference"
        )
    return actual


def _backend_record(
    backend: str,
    problem: tuple[FloatArray, FloatArray, FloatArray],
    reference: tuple[FloatArray, float],
    *,
    calls: int,
    warmups: int,
    repeats: int,
) -> dict[str, object]:
    """Record one actual owner; any failure discards timing and success hashes."""
    record: dict[str, object] = {
        "backend": backend,
        "status": "unavailable",
        "ms_per_call": None,
        "repeat_ms_per_call": [],
        "correlation_integral_sha256": None,
        "kaplan_yorke_sha256": None,
        "ci_max_abs_error": None,
        "ky_abs_error": None,
        "max_abs_error": None,
        "ci_monotonic": False,
        "ci_unit_interval": False,
        "ky_bounds_passed": False,
        "parity_passed": False,
        "tolerance": PARITY_TOLERANCES[backend],
        "unavailable_reason": "",
    }
    try:
        for _ in range(warmups):
            _checked_outputs(backend, problem, reference)
        samples = []
        actual: tuple[FloatArray, float] = reference
        for _ in range(repeats):
            started = time.perf_counter()
            for _ in range(calls):
                actual = _checked_outputs(backend, problem, reference)
            samples.append((time.perf_counter() - started) * 1000.0 / calls)
    except (ImportError, OSError, RuntimeError) as exc:
        record["unavailable_reason"] = f"{backend}: {type(exc).__name__}: {exc}"
        return record
    except (TypeError, ValueError, OverflowError) as exc:
        record["status"] = "invalid"
        record["unavailable_reason"] = f"{backend}: {type(exc).__name__}: {exc}"
        return record
    record.update(output_errors(actual, reference))
    record.update(
        {
            "status": "available",
            "ms_per_call": statistics.median(samples),
            "repeat_ms_per_call": samples,
            "correlation_integral_sha256": array_sha256(actual[0]),
            "kaplan_yorke_sha256": payload_sha256(actual[1]),
            "ky_bounds_passed": True,
            "parity_passed": True,
        }
    )
    return record


def benchmark_dimension_polyglot_parity_gate(
    *,
    t: int = 64,
    d: int = 3,
    n_k: int = 12,
    calls: int = 1,
    seed: int = 2026,
    warmups: int = 1,
    repeats: int = 3,
    require_backends: Sequence[str] = ("python",),
) -> Report:
    """Measure public CI/KY calls and refuse missing requested owners.

    Parameters
    ----------
    t, d, n_k : int
        Trajectory samples (at least2), coordinates (at least1), radii (at least2).
    calls, warmups, repeats : int
        Positive measured calls per repeat, unmeasured warmups and repeat count.
    seed : int
        Nonnegative seed for benchmark input generation, not native subsampling.
    require_backends : Sequence[str]
        Unique known owners that must pass. The portable reference-suite API
        requires Python by default; the CLI requires all five by default.

    Returns
    -------
    dict[str, float | int | str | None]
        Five honest owner slots, independent reference hashes, JSON records,
        explicit required scope and qualified wall-clock timings. Acceptance
        requires every requested owner and every available result to pass.
        Failure has no success benchmark hash. Timing is excluded from hashing.

    Raises
    ------
    ValueError
        If input controls or the nonempty required-owner selection are invalid.
    """
    t = _validate_int_control(t, name="t", minimum=2)
    d = _validate_int_control(d, name="d", minimum=1)
    n_k = _validate_int_control(n_k, name="n_k", minimum=2)
    calls = _validate_int_control(calls, name="calls", minimum=1)
    seed = _validate_int_control(seed, name="seed", minimum=0)
    warmups = _validate_int_control(warmups, name="warmups", minimum=1)
    repeats = _validate_int_control(repeats, name="repeats", minimum=1)
    required = tuple(require_backends)
    if not required or any(name not in BACKEND_ORDER for name in required):
        raise ValueError(
            "required dimension backends must be nonempty known runtime names"
        )
    if len(set(required)) != len(required):
        raise ValueError("required dimension backends must be unique")
    problem = _problem(t, d, n_k, seed)
    reference = reference_outputs(*problem)
    started = time.perf_counter()
    records = [
        _backend_record(
            backend, problem, reference, calls=calls, warmups=warmups, repeats=repeats
        )
        for backend in BACKEND_ORDER
    ]
    wall_time = time.perf_counter() - started
    available = [row for row in records if row["status"] == "available"]
    invalid = [row for row in records if row["status"] == "invalid"]
    unavailable = [row for row in records if row["status"] == "unavailable"]
    present = {str(row["backend"]) for row in available}
    required_present = set(required).issubset(present)
    acceptance = required_present and "python" in present and not invalid
    # Retain the reference-suite threshold schema; required scope is explicit
    # in required_backends_json and comparison_complete, never inferred from slots.
    thresholds = {
        "backend_order": list(BACKEND_ORDER),
        "max_mojo_abs_error": PARITY_TOLERANCES["mojo"],
        "max_native_abs_error": PARITY_TOLERANCES["rust"],
        "require_all_available_parity": True,
        "require_all_declared_backend_records": True,
        "require_correlation_integral_monotonic": True,
        "require_exact_full_pairs_reference": True,
        "require_kaplan_yorke_bounds": True,
        "require_kaplan_yorke_contract": True,
        "require_python_reference": True,
        "require_unit_interval_correlation_integral": True,
    }
    deterministic = {
        "controls": [t, d, n_k, calls, seed, warmups, repeats],
        "required_backends": required,
        "references": [array_sha256(reference[0]), payload_sha256(reference[1])],
        "records": [
            {
                key: value
                for key, value in row.items()
                if key
                not in {"ms_per_call", "repeat_ms_per_call", "unavailable_reason"}
            }
            for row in records
        ],
    }
    return {
        "suite": "dimension_polyglot_parity_gate",
        "backend_count": len(records),
        "available_backend_count": len(available),
        "unavailable_backend_count": len(unavailable),
        "invalid_backend_count": len(invalid),
        "parity_checked_count": len(available) + len(invalid),
        "parity_pass_count": len(available),
        "all_available_passed": int(bool(available) and not invalid),
        "python_reference_present": int("python" in present),
        "required_backends_present": int(required_present),
        "required_backends_json": json.dumps(required),
        "comparison_complete": int(len(available) == len(BACKEND_ORDER)),
        "T": t,
        "d": d,
        "n_k": n_k,
        "calls": calls,
        "seed": seed,
        "warmups": warmups,
        "repeats": repeats,
        "reference_ci_min": float(np.min(reference[0])),
        "reference_ci_max": float(np.max(reference[0])),
        "reference_kaplan_yorke": reference[1],
        "reference_correlation_integral_sha256": array_sha256(reference[0]),
        "reference_kaplan_yorke_sha256": payload_sha256(reference[1]),
        "benchmark_sha256": payload_sha256(deterministic) if acceptance else None,
        "wall_time_s": wall_time,
        "steps_per_second": len(available) * calls * repeats / wall_time
        if wall_time
        else 0.0,
        "acceptance_passed": int(acceptance),
        "acceptance_thresholds_json": json.dumps(thresholds, sort_keys=True),
        "backend_records_json": json.dumps(records, sort_keys=True, allow_nan=False),
        "reference_kind": "math.dist full-pair counts; exact Fraction KY prefixes",
        "timed_surface": "public CI+KY, production validators, oracle comparison",
        "benchmark_evidence_kind": "local_regression_non_isolated",
        "production_timing_claim": 0,
    }


def bench_at(
    t: int,
    d: int,
    n_k: int,
    calls: int,
    *,
    warmups: int = 1,
    repeats: int = 3,
    seed: int = 2026,
    require_backends: Sequence[str] = ("python",),
) -> Report:
    """Measure one full-pair public CI/KY problem with explicit required owners.

    Parameters
    ----------
    t, d, n_k, calls, warmups, repeats, seed, require_backends
        Controls with the same meaning and validation as the parity gate.

    Returns
    -------
    dict[str, float | int | str | None]
        Parity-gate evidence and legacy per-owner median timing keys. Each timing
        measures the documented combined public CI/KY surface.
    """
    result = benchmark_dimension_polyglot_parity_gate(
        t=t,
        d=d,
        n_k=n_k,
        calls=calls,
        seed=seed,
        warmups=warmups,
        repeats=repeats,
        require_backends=require_backends,
    )
    records = json.loads(str(result["backend_records_json"]))
    for row in records:
        value: object = row["ms_per_call"]
        result[f"{row['backend']}_ms_per_call"] = (
            float(value) if isinstance(value, (int, float)) else None
        )
    return result


def main() -> int:
    """Print and optionally save JSON; fail when required comparison is refused."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--T-list", type=int, nargs="+", default=[50, 150, 400])
    parser.add_argument("--d", type=int, default=3)
    parser.add_argument("--n-k", type=int, default=20)
    parser.add_argument("--calls", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--require-backends",
        nargs="+",
        choices=BACKEND_ORDER,
        default=list(BACKEND_ORDER),
    )
    parser.add_argument(
        "--parity-gate",
        action="store_true",
        help="emit one parity gate for the first T-list entry",
    )
    args = parser.parse_args()
    try:
        results = [
            bench_at(
                t,
                args.d,
                args.n_k,
                args.calls,
                seed=args.seed,
                warmups=args.warmups,
                repeats=args.repeats,
                require_backends=args.require_backends,
            )
            for t in (args.T_list[:1] if args.parity_gate else args.T_list)
        ]
    except ValueError as exc:
        parser.error(str(exc))
    payload: object = results[0] if args.parity_gate else {"results": results}
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if all(row["acceptance_passed"] == 1 for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
