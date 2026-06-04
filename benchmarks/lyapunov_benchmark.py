# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Lyapunov spectrum multi-backend benchmark

"""Per-backend wall-clock benchmark for
``monitor.lyapunov.lyapunov_spectrum``.

Runs the Benettin 1980 / Shimada-Nagashima 1979 algorithm across the
Rust → Mojo → Julia → Go → Python fallback chain at increasing network
sizes so the cost profile of each backend is visible on the same axis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor import lyapunov as ly_mod
from scpn_phase_orchestrator.monitor.lyapunov import (
    ACTIVE_BACKEND,
    AVAILABLE_BACKENDS,
    lyapunov_spectrum,
)

TWO_PI = 2.0 * np.pi
BACKEND_ORDER = ("rust", "mojo", "julia", "go", "python")
PARITY_TOLERANCES = {
    "rust": 1e-12,
    "julia": 1e-12,
    "go": 1e-12,
    "mojo": 1e-6,
    "python": 0.0,
}


def _bench(
    backend: str,
    phases: NDArray[np.floating],
    omegas: NDArray[np.floating],
    knm: NDArray[np.floating],
    alpha: NDArray[np.floating],
    n_steps: int,
    qr_interval: int,
    zeta: float,
    psi: float,
    calls: int,
) -> float:
    elapsed, _spectrum = _bench_with_spectrum(
        backend,
        phases,
        omegas,
        knm,
        alpha,
        n_steps,
        qr_interval,
        zeta,
        psi,
        calls,
    )
    return elapsed


def _bench_with_spectrum(
    backend: str,
    phases: NDArray[np.floating],
    omegas: NDArray[np.floating],
    knm: NDArray[np.floating],
    alpha: NDArray[np.floating],
    n_steps: int,
    qr_interval: int,
    zeta: float,
    psi: float,
    calls: int,
) -> tuple[float, NDArray[np.float64]]:
    saved = ly_mod.ACTIVE_BACKEND
    try:
        ly_mod.ACTIVE_BACKEND = backend
        # Warm-up — first call covers JIT / library init / FFI cache.
        lyapunov_spectrum(
            phases,
            omegas,
            knm,
            alpha,
            n_steps=n_steps,
            qr_interval=qr_interval,
            zeta=zeta,
            psi=psi,
        )
        t0 = time.perf_counter()
        spectrum: NDArray[np.float64] | None = None
        for _ in range(calls):
            spectrum = lyapunov_spectrum(
                phases,
                omegas,
                knm,
                alpha,
                n_steps=n_steps,
                qr_interval=qr_interval,
                zeta=zeta,
                psi=psi,
            )
        if spectrum is None:
            raise RuntimeError("benchmark calls must be positive")
        return time.perf_counter() - t0, spectrum
    finally:
        ly_mod.ACTIVE_BACKEND = saved


def _problem(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, TWO_PI, size=n)
    omegas = rng.normal(0.0, 0.2, size=n)
    knm = rng.uniform(0.3, 1.2, size=(n, n))
    np.fill_diagonal(knm, 0.0)
    alpha = rng.uniform(-0.15, 0.15, size=(n, n))
    np.fill_diagonal(alpha, 0.0)
    return phases, omegas, knm, alpha


def bench_at(
    n: int,
    n_steps: int,
    qr_interval: int,
    calls: int,
    zeta: float,
    psi: float,
) -> dict:
    phases, omegas, knm, alpha = _problem(n)
    row: dict = {
        "n": n,
        "n_steps": n_steps,
        "qr_interval": qr_interval,
        "calls": calls,
        "zeta": zeta,
        "psi": psi,
        "available": AVAILABLE_BACKENDS,
    }
    for backend in AVAILABLE_BACKENDS:
        t = _bench(
            backend,
            phases,
            omegas,
            knm,
            alpha,
            n_steps,
            qr_interval,
            zeta,
            psi,
            calls,
        )
        row[f"{backend}_ms_per_call"] = (t / calls) * 1000.0
    return row


def _spectrum_sha256(spectrum: NDArray[np.float64]) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(spectrum, dtype=np.float64).tobytes()
    ).hexdigest()


def _backend_status(backend: str) -> tuple[bool, str]:
    if backend == "python":
        return True, ""
    if backend in AVAILABLE_BACKENDS:
        return True, ""
    return False, f"{backend} backend was not resolved by monitor.lyapunov"


def benchmark_lyapunov_polyglot_parity_gate(
    *,
    n: int = 4,
    n_steps: int = 120,
    qr_interval: int = 10,
    calls: int = 1,
    zeta: float = 0.4,
    psi: float = 0.2,
) -> dict[str, float | int | str]:
    """Benchmark all declared Lyapunov backends against the Python reference.

    Every declared language slot produces a record. Available backends are timed
    and compared against the same Benettin/Shimada-Nagashima reference problem;
    unavailable backends remain explicit records with a reason. This makes the
    benchmark suitable for release notes on hosts that do not have every
    auxiliary toolchain installed.
    """

    phases, omegas, knm, alpha = _problem(n, seed=2026)
    reference_elapsed, reference_spectrum = _bench_with_spectrum(
        "python",
        phases,
        omegas,
        knm,
        alpha,
        n_steps,
        qr_interval,
        zeta,
        psi,
        calls,
    )
    records: list[dict[str, object]] = []
    for backend in BACKEND_ORDER:
        tolerance = PARITY_TOLERANCES[backend]
        available, reason = _backend_status(backend)
        if not available:
            records.append(
                {
                    "backend": backend,
                    "status": "unavailable",
                    "tolerance": tolerance,
                    "ms_per_call": None,
                    "max_abs_error": None,
                    "parity_passed": False,
                    "spectrum_sha256": None,
                    "unavailable_reason": reason,
                }
            )
            continue
        if backend == "python":
            elapsed = reference_elapsed
            spectrum = reference_spectrum
        else:
            elapsed, spectrum = _bench_with_spectrum(
                backend,
                phases,
                omegas,
                knm,
                alpha,
                n_steps,
                qr_interval,
                zeta,
                psi,
                calls,
            )
        max_abs_error = float(np.max(np.abs(spectrum - reference_spectrum)))
        parity_passed = bool(max_abs_error <= tolerance)
        records.append(
            {
                "backend": backend,
                "status": "available",
                "tolerance": tolerance,
                "ms_per_call": (elapsed / calls) * 1000.0,
                "max_abs_error": max_abs_error,
                "parity_passed": parity_passed,
                "spectrum_sha256": _spectrum_sha256(spectrum),
                "unavailable_reason": "",
            }
        )

    available_records = [
        record for record in records if record["status"] == "available"
    ]
    unavailable_records = [
        record for record in records if record["status"] == "unavailable"
    ]
    parity_pass_count = sum(int(record["parity_passed"]) for record in records)
    all_available_passed = int(
        all(bool(record["parity_passed"]) for record in available_records)
    )
    deterministic_hash = hashlib.sha256(
        json.dumps(
            [
                {
                    "backend": record["backend"],
                    "status": record["status"],
                    "max_abs_error": record["max_abs_error"],
                    "parity_passed": record["parity_passed"],
                    "spectrum_sha256": record["spectrum_sha256"],
                }
                for record in records
            ],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    acceptance_passed = int(
        len(records) == len(BACKEND_ORDER)
        and any(record["backend"] == "python" for record in available_records)
        and all_available_passed == 1
        and parity_pass_count == len(available_records)
    )
    elapsed_total = sum(
        float(record["ms_per_call"]) / 1000.0
        for record in available_records
        if record["ms_per_call"] is not None
    )
    return {
        "suite": "lyapunov_polyglot_parity_gate",
        "backend_count": len(records),
        "available_backend_count": len(available_records),
        "unavailable_backend_count": len(unavailable_records),
        "parity_checked_count": len(available_records),
        "parity_pass_count": parity_pass_count,
        "all_available_passed": all_available_passed,
        "python_reference_present": int(
            any(record["backend"] == "python" for record in available_records)
        ),
        "n": n,
        "n_steps": n_steps,
        "qr_interval": qr_interval,
        "calls": calls,
        "zeta": zeta,
        "psi": psi,
        "reference_spectrum_sha256": _spectrum_sha256(reference_spectrum),
        "benchmark_sha256": deterministic_hash,
        "wall_time_s": elapsed_total,
        "steps_per_second": (len(available_records) * calls) / elapsed_total
        if elapsed_total > 0.0
        else 0.0,
        "acceptance_passed": acceptance_passed,
        "acceptance_thresholds_json": json.dumps(
            {
                "backend_order": list(BACKEND_ORDER),
                "max_mojo_abs_error": PARITY_TOLERANCES["mojo"],
                "max_native_abs_error": PARITY_TOLERANCES["rust"],
                "require_all_declared_backend_records": True,
                "require_all_available_parity": True,
                "require_python_reference": True,
            },
            sort_keys=True,
        ),
        "backend_records_json": json.dumps(records, sort_keys=True),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--qr-interval", type=int, default=10)
    parser.add_argument("--calls", type=int, default=3)
    parser.add_argument("--zeta", type=float, default=0.0)
    parser.add_argument("--psi", type=float, default=0.0)
    parser.add_argument(
        "--parity-gate",
        action="store_true",
        help="Emit the all-declared-backend parity benchmark gate.",
    )
    args = parser.parse_args()

    if args.parity_gate:
        result = benchmark_lyapunov_polyglot_parity_gate(
            n=args.sizes[0],
            n_steps=args.n_steps,
            qr_interval=args.qr_interval,
            calls=args.calls,
            zeta=args.zeta,
            psi=args.psi,
        )
        text = json.dumps(result, indent=2)
        print(text)
        if args.output:
            args.output.write_text(text, encoding="utf-8")
        return 0

    print(f"Active: {ACTIVE_BACKEND}  Available: {AVAILABLE_BACKENDS}\n")
    header = f"{'N':>4}{'steps':>7}{'calls':>7}"
    for b in AVAILABLE_BACKENDS:
        header += f" {b + '_ms':>12}"
    print(header)
    print("-" * len(header))
    results: list[dict] = []
    for n in args.sizes:
        row = bench_at(
            n,
            args.n_steps,
            args.qr_interval,
            args.calls,
            args.zeta,
            args.psi,
        )
        results.append(row)
        line = f"{n:>4}{args.n_steps:>7}{args.calls:>7}"
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
