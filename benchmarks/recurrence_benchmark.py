# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Recurrence matrix multi-backend benchmark

"""Per-backend wall-clock benchmark for
``monitor.recurrence.recurrence_matrix``."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor import recurrence as r_mod
from scpn_phase_orchestrator.monitor.recurrence import (
    ACTIVE_BACKEND,
    AVAILABLE_BACKENDS,
    recurrence_matrix,
)

TWO_PI = 2.0 * np.pi
BACKEND_ORDER = ("rust", "mojo", "julia", "go", "python")
PARITY_METRICS = ("euclidean", "angular")


def _bench(backend: str, traj, epsilon, calls: int) -> float:
    saved = r_mod.ACTIVE_BACKEND
    try:
        r_mod.ACTIVE_BACKEND = backend
        recurrence_matrix(traj, epsilon)
        t0 = time.perf_counter()
        for _ in range(calls):
            recurrence_matrix(traj, epsilon)
        return time.perf_counter() - t0
    finally:
        r_mod.ACTIVE_BACKEND = saved


def bench_at(t: int, d: int, epsilon: float, calls: int) -> dict:
    rng = np.random.default_rng(42)
    traj = rng.normal(0, 1, (t, d))
    row: dict = {
        "T": t,
        "d": d,
        "epsilon": epsilon,
        "calls": calls,
        "available": AVAILABLE_BACKENDS,
    }
    for backend in AVAILABLE_BACKENDS:
        elapsed = _bench(backend, traj, epsilon, calls)
        row[f"{backend}_ms_per_call"] = (elapsed / calls) * 1000.0
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--T-list", type=int, nargs="+", default=[30, 100, 300])
    parser.add_argument("--d", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=0.8)
    parser.add_argument("--calls", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--parity-gate",
        action="store_true",
        help="emit deterministic all-backend recurrence parity-gate JSON",
    )
    args = parser.parse_args()

    if args.parity_gate:
        result = benchmark_recurrence_polyglot_parity_gate(
            t=args.T_list[0],
            d=args.d,
            epsilon=args.epsilon,
            calls=args.calls,
            seed=args.seed,
        )
        text = json.dumps(result, indent=2, sort_keys=True)
        print(text)
        if args.output:
            args.output.write_text(text + "\n", encoding="utf-8")
        return 0

    print(f"Active: {ACTIVE_BACKEND}  Available: {AVAILABLE_BACKENDS}\n")
    header = f"{'T':>5} {'d':>3} {'eps':>5} {'calls':>6}"
    for b in AVAILABLE_BACKENDS:
        header += f" {b + '_ms':>12}"
    print(header)
    print("-" * len(header))
    results: list[dict] = []
    for t in args.T_list:
        row = bench_at(t, args.d, args.epsilon, args.calls)
        results.append(row)
        line = f"{t:>5} {args.d:>3} {args.epsilon:>5.2f} {args.calls:>6}"
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


def _validate_epsilon_control(value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError("epsilon must be a finite non-negative real")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError("epsilon must be finite and non-negative")
    return result


def _problem(
    t: int,
    d: int,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, 4.0 * np.pi, t, dtype=np.float64)
    frequencies = np.linspace(0.35, 1.15, d, dtype=np.float64)
    offsets = rng.uniform(0.0, TWO_PI, size=d)
    base = grid[:, np.newaxis] * frequencies[np.newaxis, :] + offsets
    modulation = 0.08 * np.sin(grid[:, np.newaxis] * (frequencies[np.newaxis, :] + 0.2))
    cross_shift = 0.17 + 0.05 * np.cos(
        grid[:, np.newaxis] * (frequencies[np.newaxis, :] + 0.5)
    )
    traj_a = np.ascontiguousarray((base + modulation) % TWO_PI, dtype=np.float64)
    traj_b = np.ascontiguousarray(
        (base + modulation + cross_shift) % TWO_PI,
        dtype=np.float64,
    )
    return traj_a, traj_b


def _matrix_sha256(matrix: NDArray[np.bool_]) -> str:
    payload = np.ascontiguousarray(matrix.astype(np.uint8, copy=False))
    return hashlib.sha256(payload.tobytes()).hexdigest()


def _bundle_sha256(bundle: Mapping[str, NDArray[np.bool_]]) -> str:
    digest = hashlib.sha256()
    for key in sorted(bundle):
        payload = np.ascontiguousarray(bundle[key].astype(np.uint8, copy=False))
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload.tobytes())
    return digest.hexdigest()


def _backend_status(backend: str) -> tuple[bool, str]:
    if backend in AVAILABLE_BACKENDS:
        return True, ""
    return False, f"{backend} backend was not resolved by monitor.recurrence"


def _backend_function(backend: str, name: str) -> Callable[..., object]:
    loaded = r_mod._load_backend(backend)
    fn = loaded.get(name)
    if not callable(fn):
        raise ValueError(f"{backend} backend does not expose {name}")
    return fn


def _direct_recurrence(
    backend: str,
    trajectory: NDArray[np.float64],
    *,
    epsilon: float,
    metric: str,
) -> NDArray[np.bool_]:
    angular = metric == "angular"
    t, d = int(trajectory.shape[0]), int(trajectory.shape[1])
    expected = r_mod._expected_recurrence_matrix(
        trajectory,
        trajectory,
        epsilon=epsilon,
        angular=angular,
    )
    if backend == "python":
        return expected.copy()
    flat = np.ascontiguousarray(trajectory.ravel(), dtype=np.float64)
    raw = _backend_function(backend, "rm")(flat, t, d, epsilon, angular)
    return r_mod._backend_recurrence_matrix(
        raw,
        t=t,
        name="recurrence_matrix",
        expected=expected,
    )


def _direct_cross_recurrence(
    backend: str,
    traj_a: NDArray[np.float64],
    traj_b: NDArray[np.float64],
    *,
    epsilon: float,
    metric: str,
) -> NDArray[np.bool_]:
    angular = metric == "angular"
    t, d = int(traj_a.shape[0]), int(traj_a.shape[1])
    expected = r_mod._expected_recurrence_matrix(
        traj_a,
        traj_b,
        epsilon=epsilon,
        angular=angular,
    )
    if backend == "python":
        return expected.copy()
    a_flat = np.ascontiguousarray(traj_a.ravel(), dtype=np.float64)
    b_flat = np.ascontiguousarray(traj_b.ravel(), dtype=np.float64)
    raw = _backend_function(backend, "cross_rm")(
        a_flat,
        b_flat,
        t,
        d,
        epsilon,
        angular,
    )
    return r_mod._backend_recurrence_matrix(
        raw,
        t=t,
        name="cross_recurrence_matrix",
        expected=expected,
    )


def _direct_bundle(
    backend: str,
    traj_a: NDArray[np.float64],
    traj_b: NDArray[np.float64],
    *,
    epsilon: float,
) -> dict[str, NDArray[np.bool_]]:
    bundle: dict[str, NDArray[np.bool_]] = {}
    for metric in PARITY_METRICS:
        bundle[f"{metric}_recurrence"] = _direct_recurrence(
            backend,
            traj_a,
            epsilon=epsilon,
            metric=metric,
        )
        bundle[f"{metric}_cross"] = _direct_cross_recurrence(
            backend,
            traj_a,
            traj_b,
            epsilon=epsilon,
            metric=metric,
        )
        bundle[f"{metric}_self_cross"] = _direct_cross_recurrence(
            backend,
            traj_a,
            traj_a,
            epsilon=epsilon,
            metric=metric,
        )
    return bundle


def _bench_with_output(
    backend: str,
    traj_a: NDArray[np.float64],
    traj_b: NDArray[np.float64],
    *,
    epsilon: float,
    calls: int,
) -> tuple[float, dict[str, NDArray[np.bool_]]]:
    _direct_bundle(backend, traj_a, traj_b, epsilon=epsilon)
    t0 = time.perf_counter()
    output: dict[str, NDArray[np.bool_]] | None = None
    for _ in range(calls):
        output = _direct_bundle(backend, traj_a, traj_b, epsilon=epsilon)
    if output is None:
        raise RuntimeError("benchmark calls must be positive")
    return time.perf_counter() - t0, output


def _bundle_mismatch_count(
    actual: Mapping[str, NDArray[np.bool_]],
    expected: Mapping[str, NDArray[np.bool_]],
) -> int:
    return sum(int(np.count_nonzero(actual[key] != expected[key])) for key in expected)


def _self_cross_mismatch_count(bundle: Mapping[str, NDArray[np.bool_]]) -> int:
    total = 0
    for metric in PARITY_METRICS:
        total += int(
            np.count_nonzero(
                bundle[f"{metric}_self_cross"] != bundle[f"{metric}_recurrence"]
            )
        )
    return total


def _recurrence_invariants_pass(bundle: Mapping[str, NDArray[np.bool_]]) -> bool:
    for metric in PARITY_METRICS:
        recurrence = bundle[f"{metric}_recurrence"]
        if not np.all(np.diag(recurrence)):
            return False
        if not np.array_equal(recurrence, recurrence.T):
            return False
    return True


def _rqa_bounds_passed(result: r_mod.RQAResult) -> bool:
    return (
        0.0 <= result.recurrence_rate <= 1.0
        and 0.0 <= result.determinism <= 1.0
        and result.avg_diagonal >= 0.0
        and result.max_diagonal >= 0
        and result.entropy_diagonal >= 0.0
        and 0.0 <= result.laminarity <= 1.0
        and result.trapping_time >= 0.0
        and result.max_vertical >= 0
    )


def _python_rqa(
    trajectory: NDArray[np.float64],
    *,
    epsilon: float,
) -> r_mod.RQAResult:
    saved = r_mod.ACTIVE_BACKEND
    try:
        r_mod.ACTIVE_BACKEND = "python"
        return r_mod.rqa(trajectory, epsilon, metric="angular")
    finally:
        r_mod.ACTIVE_BACKEND = saved


def benchmark_recurrence_polyglot_parity_gate(
    *,
    t: int = 64,
    d: int = 3,
    epsilon: float = 0.8,
    calls: int = 1,
    seed: int = 2026,
) -> dict[str, object]:
    """Record exact recurrence parity across every declared backend slot.

    The gate calls resolved backends directly instead of relying on the public
    fallback dispatcher. Available backends must preserve both Euclidean and
    angular threshold relations exactly for recurrence, cross-recurrence, and
    the self-cross equals recurrence identity.
    """

    t = _validate_int_control(t, name="t", minimum=2)
    d = _validate_int_control(d, name="d", minimum=1)
    calls = _validate_int_control(calls, name="calls", minimum=1)
    seed = _validate_int_control(seed, name="seed", minimum=0)
    epsilon = _validate_epsilon_control(epsilon)

    traj_a, traj_b = _problem(t, d, seed)
    reference = _direct_bundle("python", traj_a, traj_b, epsilon=epsilon)
    reference_rqa = _python_rqa(traj_a, epsilon=epsilon)

    records: list[dict[str, object]] = []
    parity_checked_count = 0
    parity_pass_count = 0
    available_backend_count = 0
    t0 = time.perf_counter()

    for backend in BACKEND_ORDER:
        available, reason = _backend_status(backend)
        if not available:
            records.append(
                {
                    "backend": backend,
                    "status": "unavailable",
                    "ms_per_call": None,
                    "recurrence_sha256": None,
                    "cross_recurrence_sha256": None,
                    "self_cross_sha256": None,
                    "mismatch_count": None,
                    "self_cross_mismatch_count": None,
                    "max_abs_error": None,
                    "recurrence_invariants_passed": False,
                    "self_cross_equals_recurrence": False,
                    "parity_passed": False,
                    "tolerance": 0,
                    "unavailable_reason": reason,
                }
            )
            continue

        available_backend_count += 1
        elapsed, bundle = _bench_with_output(
            backend,
            traj_a,
            traj_b,
            epsilon=epsilon,
            calls=calls,
        )
        mismatch_count = _bundle_mismatch_count(bundle, reference)
        self_cross_mismatch_count = _self_cross_mismatch_count(bundle)
        recurrence_invariants_passed = _recurrence_invariants_pass(bundle)
        self_cross_equals_recurrence = self_cross_mismatch_count == 0
        exact_match = mismatch_count == 0
        parity_passed = (
            exact_match
            and recurrence_invariants_passed
            and self_cross_equals_recurrence
        )
        parity_checked_count += 1
        parity_pass_count += int(parity_passed)
        recurrence_bundle = {
            key: value for key, value in bundle.items() if key.endswith("_recurrence")
        }
        cross_bundle = {
            key: value
            for key, value in bundle.items()
            if key.endswith("_cross") and not key.endswith("_self_cross")
        }
        self_cross_bundle = {
            key: value for key, value in bundle.items() if key.endswith("_self_cross")
        }
        records.append(
            {
                "backend": backend,
                "status": "available",
                "ms_per_call": (elapsed / calls) * 1000.0,
                "recurrence_sha256": _bundle_sha256(recurrence_bundle),
                "cross_recurrence_sha256": _bundle_sha256(cross_bundle),
                "self_cross_sha256": _bundle_sha256(self_cross_bundle),
                "mismatch_count": mismatch_count,
                "self_cross_mismatch_count": self_cross_mismatch_count,
                "max_abs_error": int(mismatch_count > 0),
                "recurrence_invariants_passed": recurrence_invariants_passed,
                "self_cross_equals_recurrence": self_cross_equals_recurrence,
                "parity_passed": parity_passed,
                "tolerance": 0,
                "unavailable_reason": "",
            }
        )

    wall_time = time.perf_counter() - t0
    thresholds = {
        "backend_order": list(BACKEND_ORDER),
        "metrics": list(PARITY_METRICS),
        "require_all_available_parity": True,
        "require_all_declared_backend_records": True,
        "require_cross_recurrence_contract": True,
        "require_exact_threshold_reference": True,
        "require_python_reference": True,
        "require_rqa_unit_interval_contract": True,
        "require_self_cross_equals_recurrence": True,
        "require_symmetric_true_diagonal_recurrence": True,
        "tolerance": 0,
    }
    acceptance_passed = (
        len(records) == len(BACKEND_ORDER)
        and any(
            record["backend"] == "python" and record["status"] == "available"
            for record in records
        )
        and parity_pass_count == parity_checked_count
        and _recurrence_invariants_pass(reference)
        and _self_cross_mismatch_count(reference) == 0
        and _rqa_bounds_passed(reference_rqa)
    )
    benchmark_payload = {
        "T": t,
        "d": d,
        "epsilon": epsilon,
        "calls": calls,
        "seed": seed,
        "records": records,
        "thresholds": thresholds,
        "reference_recurrence_sha256": _bundle_sha256(
            {
                key: value
                for key, value in reference.items()
                if key.endswith("_recurrence")
            }
        ),
        "reference_cross_recurrence_sha256": _bundle_sha256(
            {
                key: value
                for key, value in reference.items()
                if key.endswith("_cross") and not key.endswith("_self_cross")
            }
        ),
        "reference_self_cross_sha256": _bundle_sha256(
            {
                key: value
                for key, value in reference.items()
                if key.endswith("_self_cross")
            }
        ),
    }
    benchmark_sha = hashlib.sha256(
        json.dumps(benchmark_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    return {
        "suite": "recurrence_polyglot_parity_gate",
        "backend_count": len(records),
        "available_backend_count": available_backend_count,
        "unavailable_backend_count": len(records) - available_backend_count,
        "parity_checked_count": parity_checked_count,
        "parity_pass_count": parity_pass_count,
        "all_available_passed": int(parity_pass_count == parity_checked_count),
        "python_reference_present": 1,
        "T": t,
        "d": d,
        "epsilon": epsilon,
        "calls": calls,
        "seed": seed,
        "reference_recurrence_rate": reference_rqa.recurrence_rate,
        "reference_determinism": reference_rqa.determinism,
        "reference_laminarity": reference_rqa.laminarity,
        "reference_recurrence_sha256": benchmark_payload["reference_recurrence_sha256"],
        "reference_cross_recurrence_sha256": benchmark_payload[
            "reference_cross_recurrence_sha256"
        ],
        "reference_self_cross_sha256": benchmark_payload["reference_self_cross_sha256"],
        "benchmark_sha256": benchmark_sha,
        "wall_time_s": wall_time,
        "steps_per_second": parity_checked_count / wall_time if wall_time else 0.0,
        "acceptance_passed": int(acceptance_passed),
        "acceptance_thresholds_json": json.dumps(thresholds, sort_keys=True),
        "backend_records_json": json.dumps(records, sort_keys=True),
    }


if __name__ == "__main__":
    raise SystemExit(main())
