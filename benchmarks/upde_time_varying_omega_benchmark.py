# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — UPDE time-varying omega benchmark gate

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from contextlib import suppress
from hashlib import sha256
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators.upde._engine_go import (
    upde_run_omega_schedule_go,
)
from scpn_phase_orchestrator.experimental.accelerators.upde._engine_julia import (
    upde_run_omega_schedule_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.upde._engine_mojo import (
    upde_run_omega_schedule_mojo,
)
from scpn_phase_orchestrator.upde._ref_kernel import upde_run_omega_schedule_python
from scpn_phase_orchestrator.upde.engine import UPDEEngine

FloatArray: TypeAlias = NDArray[np.float64]
BackendFn: TypeAlias = Callable[..., FloatArray]

BACKEND_ORDER = ("rust", "webgpu", "mojo", "julia", "go", "python")
TOLERANCES = {
    "rust": 1.0e-12,
    "webgpu": 1.0e-9,
    "mojo": 1.0e-9,
    "julia": 1.0e-10,
    "go": 1.0e-10,
    "python": 0.0,
}


def _hash_array(values: FloatArray) -> str:
    return sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()


def _problem(n: int, n_steps: int, seed: int) -> dict[str, FloatArray]:
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, 1.0, size=n).astype(np.float64)
    omega0 = rng.uniform(0.7, 1.3, size=n).astype(np.float64)
    slope = rng.uniform(-0.03, 0.03, size=n).astype(np.float64)
    dt = 1.0e-4
    times = np.arange(n_steps, dtype=np.float64)[:, None] * dt
    schedule = omega0[None, :] + times * slope[None, :]
    knm = rng.uniform(0.0, 0.08, size=(n, n)).astype(np.float64)
    knm = 0.5 * (knm + knm.T)
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n), dtype=np.float64)
    return {
        "phases": phases,
        "omega0": omega0,
        "slope": slope,
        "schedule": schedule,
        "knm": knm,
        "alpha": alpha,
        "dt": np.array([dt], dtype=np.float64),
    }


def _rust_schedule_backend() -> BackendFn:
    from spo_kernel import PyUPDEStepper

    if not hasattr(PyUPDEStepper, "run_omega_schedule"):
        raise ImportError("PyUPDEStepper.run_omega_schedule is unavailable")

    def run(
        phases: FloatArray,
        schedule: FloatArray,
        knm: FloatArray,
        alpha: FloatArray,
        zeta: float,
        psi: float,
        dt: float,
        method: str,
        n_substeps: int,
        atol: float,
        rtol: float,
    ) -> FloatArray:
        stepper = PyUPDEStepper(
            int(phases.size), dt, method, n_substeps=n_substeps, atol=atol, rtol=rtol
        )
        return np.asarray(
            stepper.run_omega_schedule(
                np.ascontiguousarray(phases.ravel(), dtype=np.float64),
                np.ascontiguousarray(schedule.ravel(), dtype=np.float64),
                np.ascontiguousarray(knm.ravel(), dtype=np.float64),
                float(zeta),
                float(psi),
                np.ascontiguousarray(alpha.ravel(), dtype=np.float64),
                int(schedule.shape[0]),
            ),
            dtype=np.float64,
        )

    return run


def _backend_map() -> dict[str, BackendFn]:
    backends: dict[str, BackendFn] = {
        "mojo": upde_run_omega_schedule_mojo,
        "julia": upde_run_omega_schedule_julia,
        "go": upde_run_omega_schedule_go,
        "python": upde_run_omega_schedule_python,
    }
    with suppress(ImportError):
        backends["rust"] = _rust_schedule_backend()
    return backends


def _stateful_reference(problem: dict[str, FloatArray], n_steps: int) -> FloatArray:
    dt = float(problem["dt"][0])
    omega0 = problem["omega0"]
    slope = problem["slope"]

    def omega(t: float) -> FloatArray:
        return np.asarray(omega0 + slope * t, dtype=np.float64)

    engine = UPDEEngine(int(omega0.size), dt=dt, method="rk4", omega=omega)
    return engine.run(
        problem["phases"],
        knm=problem["knm"],
        alpha=problem["alpha"],
        n_steps=n_steps,
    )


def benchmark_upde_time_varying_omega_polyglot_gate(
    *,
    n: int = 8,
    n_steps: int = 8,
    calls: int = 1,
    seed: int = 2026,
) -> dict[str, Any]:
    problem = _problem(n=n, n_steps=n_steps, seed=seed)
    phases = problem["phases"]
    schedule = problem["schedule"]
    knm = problem["knm"]
    alpha = problem["alpha"]
    dt = float(problem["dt"][0])
    reference = _stateful_reference(problem, n_steps)
    direct_python = upde_run_omega_schedule_python(
        phases,
        schedule,
        knm,
        alpha,
        0.0,
        0.0,
        dt,
        "rk4",
        1,
        1.0e-6,
        1.0e-3,
    )
    stateful_abs_error = float(np.max(np.abs(reference - direct_python)))

    records: list[dict[str, Any]] = []
    parity_pass_count = 0
    available_count = 0
    backends = _backend_map()

    start_all = time.perf_counter()
    for backend in BACKEND_ORDER:
        fn = backends.get(backend)
        if fn is None:
            records.append(
                {
                    "backend": backend,
                    "status": "unavailable",
                    "parity_passed": False,
                    "max_abs_error": None,
                    "ms_per_call": None,
                    "matrix_sha256": None,
                    "tolerance": TOLERANCES[backend],
                    "unavailable_reason": "schedule backend is not installed",
                }
            )
            continue
        try:
            elapsed_start = time.perf_counter()
            got = reference
            for _ in range(calls):
                got = fn(
                    phases,
                    schedule,
                    knm,
                    alpha,
                    0.0,
                    0.0,
                    dt,
                    "rk4",
                    1,
                    1.0e-6,
                    1.0e-3,
                )
            elapsed = time.perf_counter() - elapsed_start
            max_abs_error = float(np.max(np.abs(got - reference)))
            passed = max_abs_error <= TOLERANCES[backend]
            available_count += 1
            parity_pass_count += int(passed)
            records.append(
                {
                    "backend": backend,
                    "status": "available",
                    "parity_passed": passed,
                    "max_abs_error": max_abs_error,
                    "ms_per_call": (elapsed / calls) * 1000.0,
                    "matrix_sha256": _hash_array(got),
                    "tolerance": TOLERANCES[backend],
                    "unavailable_reason": "",
                }
            )
        except (AttributeError, ImportError, OSError, RuntimeError, ValueError) as exc:
            records.append(
                {
                    "backend": backend,
                    "status": "unavailable",
                    "parity_passed": False,
                    "max_abs_error": None,
                    "ms_per_call": None,
                    "matrix_sha256": None,
                    "tolerance": TOLERANCES[backend],
                    "unavailable_reason": str(exc),
                }
            )
    wall_time = time.perf_counter() - start_all
    all_available_passed = all(
        bool(record["parity_passed"])
        for record in records
        if record["status"] == "available"
    )
    acceptance_passed = bool(
        stateful_abs_error <= 1.0e-12
        and all_available_passed
        and any(
            record["backend"] == "python" and record["status"] == "available"
            for record in records
        )
    )
    payload = {
        "suite": "upde_time_varying_omega_polyglot_gate",
        "n": n,
        "n_steps": n_steps,
        "calls": calls,
        "seed": seed,
        "backend_count": len(BACKEND_ORDER),
        "available_backend_count": available_count,
        "unavailable_backend_count": len(BACKEND_ORDER) - available_count,
        "parity_pass_count": parity_pass_count,
        "stateful_schedule_abs_error": stateful_abs_error,
        "reference_matrix_sha256": _hash_array(reference),
        "schedule_sha256": _hash_array(schedule),
        "backend_records_json": json.dumps(records, sort_keys=True),
        "acceptance_passed": int(acceptance_passed),
        "all_available_passed": int(all_available_passed),
        "benchmark_evidence_kind": "local_regression_non_isolated",
        "isolation_method": "none",
        "production_timing_claim": 0,
        "wall_time_s": wall_time,
        "steps_per_second": (n_steps * calls) / wall_time if wall_time > 0.0 else 0.0,
        "acceptance_thresholds_json": json.dumps(
            {
                "backend_order": list(BACKEND_ORDER),
                "require_python_reference": True,
                "require_stateful_vs_schedule_parity": True,
                "require_all_available_parity": True,
                "production_timing_claim": False,
            },
            sort_keys=True,
        ),
    }
    payload["benchmark_sha256"] = sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parity-gate", action="store_true")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--calls", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    result = benchmark_upde_time_varying_omega_polyglot_gate(
        n=args.n,
        n_steps=args.steps,
        calls=args.calls,
        seed=args.seed,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if (not args.parity_gate or result["acceptance_passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
