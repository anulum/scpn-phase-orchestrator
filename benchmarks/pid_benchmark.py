# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — PID decomposition multi-backend benchmark

"""Per-backend wall-clock benchmark and polyglot parity gate for
``monitor.pid`` (time-series Williams & Beer redundancy / synergy)."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor import pid as pid_mod
from scpn_phase_orchestrator.monitor.pid import (
    ACTIVE_BACKEND,
    AVAILABLE_BACKENDS,
    redundancy,
    synergy,
)

TWO_PI = 2.0 * np.pi
BACKEND_ORDER = ("rust", "mojo", "julia", "go", "python")
PARITY_TOLERANCES = {
    "rust": 1.0e-10,
    "mojo": 1.0e-6,
    "julia": 1.0e-10,
    "go": 1.0e-10,
    "python": 0.0,
}
# The co-varying source pair must expose a clearly positive synergy.
SYNERGY_FLOOR = 1.0e-2
ZERO_SYNERGY_TOLERANCE = 1.0e-9
BENCHMARK_EVIDENCE_KIND = "local_regression_non_isolated"
_GROUP_A = (0, 1, 2, 3)
_GROUP_B = (4, 5, 6, 7)


def _validate_int_control(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _covarying_history(t: int, seed: int) -> NDArray[np.float64]:
    """Group A sweeps one phase and group B another, so the global target
    depends on both sources (positive redundancy and synergy).

    A small deterministic jitter keeps the reduced phases off the exact bin
    edges so binning — and hence the estimate — is identical across backends;
    the smooth sweep alone would land on bin boundaries where a last-ULP
    ``atan2`` difference between languages flips a single bin.
    """
    idx = np.arange(t, dtype=np.float64)
    history = np.zeros((t, 8), dtype=np.float64)
    history[:, 0:4] = (TWO_PI * idx / t)[:, None]
    history[:, 4:8] = (TWO_PI * 2.0 * idx / t)[:, None]
    rng = np.random.default_rng(seed)
    history += rng.uniform(-1.0e-3, 1.0e-3, history.shape)
    return np.ascontiguousarray(history)


def _redundant_history(t: int) -> NDArray[np.float64]:
    """One sweeping phase shared by all oscillators → fully redundant."""
    idx = np.arange(t, dtype=np.float64)
    return np.ascontiguousarray(np.tile((TWO_PI * idx / t)[:, None], (1, 8)))


def _decompose(
    backend: str, history: NDArray[np.float64], n_bins: int
) -> tuple[float, float]:
    saved = pid_mod.ACTIVE_BACKEND
    try:
        pid_mod.ACTIVE_BACKEND = backend
        return (
            redundancy(history, list(_GROUP_A), list(_GROUP_B), n_bins),
            synergy(history, list(_GROUP_A), list(_GROUP_B), n_bins),
        )
    finally:
        pid_mod.ACTIVE_BACKEND = saved


def _reference_contracts(
    backend: str,
    covarying: NDArray[np.float64],
    redundant: NDArray[np.float64],
    n_bins: int,
) -> dict[str, float | int]:
    red_cov, syn_cov = _decompose(backend, covarying, n_bins)
    red_full, syn_full = _decompose(backend, redundant, n_bins)
    return {
        "covarying_redundancy": red_cov,
        "covarying_synergy": syn_cov,
        "redundant_redundancy": red_full,
        "redundant_synergy": syn_full,
    }


def _contracts_passed(contracts: dict[str, float | int]) -> bool:
    return (
        float(contracts["covarying_redundancy"]) >= 0.0
        and float(contracts["covarying_synergy"]) > SYNERGY_FLOOR
        and float(contracts["redundant_redundancy"]) > 0.0
        and float(contracts["redundant_synergy"]) <= ZERO_SYNERGY_TOLERANCE
    )


def _backend_status(backend: str) -> tuple[bool, str]:
    if backend in AVAILABLE_BACKENDS:
        return True, ""
    return False, f"{backend} backend was not resolved by monitor.pid"


def _bench_with_output(
    backend: str,
    covarying: NDArray[np.float64],
    redundant: NDArray[np.float64],
    n_bins: int,
    *,
    calls: int,
) -> tuple[float, dict[str, float | int]]:
    _reference_contracts(backend, covarying, redundant, n_bins)
    t0 = time.perf_counter()
    contracts: dict[str, float | int] = {}
    for _ in range(calls):
        contracts = _reference_contracts(backend, covarying, redundant, n_bins)
    return time.perf_counter() - t0, contracts


def benchmark_pid_polyglot_parity_gate(
    *,
    n_steps: int = 1500,
    n_bins: int = 12,
    calls: int = 1,
    seed: int = 2026,
) -> dict[str, object]:
    """Record PID redundancy/synergy parity across declared backend slots.

    Every available backend must reproduce the Python reference redundancy and
    synergy and satisfy the decomposition contracts: a co-varying source pair
    has positive redundancy and synergy, while a fully redundant configuration
    has positive redundancy and vanishing synergy.
    """
    n_steps = _validate_int_control(n_steps, name="n_steps", minimum=2)
    n_bins = _validate_int_control(n_bins, name="n_bins", minimum=2)
    calls = _validate_int_control(calls, name="calls", minimum=1)
    seed = _validate_int_control(seed, name="seed", minimum=0)

    covarying = _covarying_history(n_steps, seed)
    redundant = _redundant_history(n_steps)
    reference = _reference_contracts("python", covarying, redundant, n_bins)
    reference_contracts_passed = _contracts_passed(reference)

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
                    "redundancy_abs_error": None,
                    "synergy_abs_error": None,
                    "max_abs_error": None,
                    "covarying_redundancy": None,
                    "covarying_synergy": None,
                    "reference_contracts_passed": False,
                    "tolerance": tolerance,
                    "parity_passed": False,
                    "unavailable_reason": reason,
                }
            )
            continue

        available_backend_count += 1
        elapsed, contracts = _bench_with_output(
            backend, covarying, redundant, n_bins, calls=calls
        )
        red_err = abs(
            float(contracts["covarying_redundancy"])
            - float(reference["covarying_redundancy"])
        )
        syn_err = abs(
            float(contracts["covarying_synergy"])
            - float(reference["covarying_synergy"])
        )
        max_err = max(red_err, syn_err)
        contracts_passed = _contracts_passed(contracts)
        parity_passed = bool(max_err <= tolerance and contracts_passed)
        parity_checked_count += 1
        parity_pass_count += int(parity_passed)
        records.append(
            {
                "backend": backend,
                "status": "available",
                "ms_per_call": (elapsed / calls) * 1000.0,
                "redundancy_abs_error": red_err,
                "synergy_abs_error": syn_err,
                "max_abs_error": max_err,
                "covarying_redundancy": float(contracts["covarying_redundancy"]),
                "covarying_synergy": float(contracts["covarying_synergy"]),
                "reference_contracts_passed": contracts_passed,
                "tolerance": tolerance,
                "parity_passed": parity_passed,
                "unavailable_reason": "",
            }
        )

    wall_time = time.perf_counter() - t0
    thresholds = {
        "backend_order": list(BACKEND_ORDER),
        "parity_tolerances": PARITY_TOLERANCES,
        "synergy_floor": SYNERGY_FLOOR,
        "zero_synergy_tolerance": ZERO_SYNERGY_TOLERANCE,
        "require_all_available_parity": True,
        "require_all_declared_backend_records": True,
        "require_python_reference": True,
        "require_covarying_positive_synergy": True,
        "require_redundant_zero_synergy": True,
        "require_non_negative_components": True,
        "production_timing_claim": False,
    }
    acceptance_passed = (
        len(records) == len(BACKEND_ORDER)
        and any(
            r["backend"] == "python" and r["status"] == "available" for r in records
        )
        and parity_checked_count > 0
        and parity_pass_count == parity_checked_count
        and reference_contracts_passed
    )
    benchmark_payload = {
        "n_steps": n_steps,
        "n_bins": n_bins,
        "calls": calls,
        "seed": seed,
        "records": [
            {
                k: v
                for k, v in r.items()
                if k not in {"ms_per_call", "unavailable_reason"}
            }
            for r in records
        ],
        "reference_contracts": reference,
    }
    benchmark_sha = hashlib.sha256(
        json.dumps(benchmark_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    return {
        "suite": "pid_polyglot_parity_gate",
        "backend_count": len(records),
        "available_backend_count": available_backend_count,
        "unavailable_backend_count": len(records) - available_backend_count,
        "parity_checked_count": parity_checked_count,
        "parity_pass_count": parity_pass_count,
        "all_available_passed": int(parity_pass_count == parity_checked_count),
        "python_reference_present": 1,
        "reference_contracts_passed": int(reference_contracts_passed),
        "acceptance_passed": int(acceptance_passed),
        "n_steps": n_steps,
        "n_bins": n_bins,
        "calls": calls,
        "seed": seed,
        "reference_covarying_redundancy": float(reference["covarying_redundancy"]),
        "reference_covarying_synergy": float(reference["covarying_synergy"]),
        "reference_redundant_synergy": float(reference["redundant_synergy"]),
        "benchmark_sha256": benchmark_sha,
        "benchmark_evidence_kind": BENCHMARK_EVIDENCE_KIND,
        "isolation_method": "none",
        "production_timing_claim": 0,
        "wall_time_s": wall_time,
        "steps_per_second": parity_checked_count / wall_time if wall_time else 0.0,
        "acceptance_thresholds_json": json.dumps(thresholds, sort_keys=True),
        "backend_records_json": json.dumps(records, sort_keys=True),
    }


def bench_at(n_steps: int, n_bins: int, calls: int, *, seed: int = 2026) -> dict:
    covarying = _covarying_history(n_steps, seed)
    row: dict = {
        "n_steps": n_steps,
        "n_bins": n_bins,
        "calls": calls,
        "available": AVAILABLE_BACKENDS,
    }
    for backend in AVAILABLE_BACKENDS:
        saved = pid_mod.ACTIVE_BACKEND
        try:
            pid_mod.ACTIVE_BACKEND = backend
            redundancy(covarying, list(_GROUP_A), list(_GROUP_B), n_bins)
            t0 = time.perf_counter()
            for _ in range(calls):
                redundancy(covarying, list(_GROUP_A), list(_GROUP_B), n_bins)
                synergy(covarying, list(_GROUP_A), list(_GROUP_B), n_bins)
            elapsed = time.perf_counter() - t0
        finally:
            pid_mod.ACTIVE_BACKEND = saved
        row[f"{backend}_ms_per_call"] = (elapsed / calls) * 1000.0
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--n-steps", type=int, default=1500)
    parser.add_argument("--n-bins", type=int, default=12)
    parser.add_argument("--calls", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--parity-gate", action="store_true")
    args = parser.parse_args()

    if args.parity_gate:
        result = benchmark_pid_polyglot_parity_gate(
            n_steps=args.n_steps,
            n_bins=args.n_bins,
            calls=args.calls,
            seed=args.seed,
        )
        text = json.dumps(result, indent=2, sort_keys=True)
        print(text)
        if args.output:
            args.output.write_text(text + "\n", encoding="utf-8")
        return 0

    print(f"Active: {ACTIVE_BACKEND}  Available: {AVAILABLE_BACKENDS}\n")
    row = bench_at(args.n_steps, args.n_bins, args.calls, seed=args.seed)
    for backend in AVAILABLE_BACKENDS:
        print(f"{backend:>8}: {row[f'{backend}_ms_per_call']:.4f} ms/call")
    if args.output:
        args.output.write_text(json.dumps(row, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
