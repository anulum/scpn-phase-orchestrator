# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Šotek. All rights reserved.
# © Code 2020-2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — moving-frame UPDE benchmark gate

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

from scpn_phase_orchestrator.coupling.spatial_modulator import SpatialCouplingModulator
from scpn_phase_orchestrator.experimental.accelerators.upde._moving_frame_go import (
    moving_frame_run_go,
)
from scpn_phase_orchestrator.experimental.accelerators.upde._moving_frame_julia import (
    moving_frame_run_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.upde._moving_frame_mojo import (
    moving_frame_run_mojo,
)
from scpn_phase_orchestrator.upde.moving_frame import (
    KINEMATIC_RESIDUAL_TOLERANCE_M,
    moving_frame_run_python,
)

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


def _expected_final_positions(
    positions: FloatArray,
    velocities: FloatArray,
    dt: float,
) -> FloatArray:
    return np.ascontiguousarray(
        positions + dt * np.sum(velocities, axis=0),
        dtype=np.float64,
    )


def _kinematic_residual_max(
    flat_state: FloatArray,
    *,
    n: int,
    expected_positions: FloatArray,
) -> float:
    return float(np.max(np.abs(flat_state[n:] - expected_positions)))


def _problem(n: int, n_steps: int, seed: int) -> dict[str, FloatArray]:
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, 1.0, size=n).astype(np.float64)
    positions = np.linspace(-0.015, 0.015, n, dtype=np.float64)
    omega = rng.uniform(-0.25, 0.25, size=(n_steps, n)).astype(np.float64)
    velocities = rng.uniform(-1250.0, 1250.0, size=(n_steps, n)).astype(np.float64)
    knm = rng.uniform(0.0, 0.5, size=(n, n)).astype(np.float64)
    knm = 0.5 * (knm + knm.T)
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n), dtype=np.float64)
    return {
        "phases": phases,
        "positions": positions,
        "omega": omega,
        "velocities": velocities,
        "knm": knm,
        "alpha": alpha,
        "dt": np.array([1.0e-6], dtype=np.float64),
        "spatial_k_base": np.array([0.8], dtype=np.float64),
        "spatial_decay_form": np.array([0], dtype=np.float64),
        "spatial_decay_exponent": np.array([1.0], dtype=np.float64),
        "spatial_decay_length_scale": np.array([1.0], dtype=np.float64),
        "spatial_epsilon": np.array([1.0e-12], dtype=np.float64),
        "strength": np.array([0.025], dtype=np.float64),
        "epsilon": np.array([1.0e-9], dtype=np.float64),
    }


def _rust_backend() -> BackendFn:
    from spo_kernel import PyUPDEStepper

    if not hasattr(PyUPDEStepper, "run_moving_frame_schedule"):
        raise ImportError("PyUPDEStepper.run_moving_frame_schedule is unavailable")

    def run(
        phases: FloatArray,
        positions: FloatArray,
        omega_schedule: FloatArray,
        knm: FloatArray,
        alpha: FloatArray,
        velocity_schedule: FloatArray,
        spatial_k_base: float,
        spatial_decay_form: int,
        spatial_decay_exponent: float,
        spatial_decay_length_scale: float,
        spatial_epsilon: float,
        doppler_strength: float,
        doppler_epsilon: float,
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
            stepper.run_moving_frame_schedule(
                np.ascontiguousarray(phases.ravel(), dtype=np.float64),
                np.ascontiguousarray(positions.ravel(), dtype=np.float64),
                np.ascontiguousarray(omega_schedule.ravel(), dtype=np.float64),
                np.ascontiguousarray(knm.ravel(), dtype=np.float64),
                float(zeta),
                float(psi),
                np.ascontiguousarray(alpha.ravel(), dtype=np.float64),
                np.ascontiguousarray(velocity_schedule.ravel(), dtype=np.float64),
                float(spatial_k_base),
                int(spatial_decay_form),
                float(spatial_decay_exponent),
                float(spatial_decay_length_scale),
                float(spatial_epsilon),
                float(doppler_strength),
                float(doppler_epsilon),
                int(omega_schedule.shape[0]),
            ),
            dtype=np.float64,
        )

    return run


def _backend_map() -> dict[str, BackendFn]:
    backends: dict[str, BackendFn] = {
        "mojo": moving_frame_run_mojo,
        "julia": moving_frame_run_julia,
        "go": moving_frame_run_go,
        "python": moving_frame_run_python,
    }
    with suppress(ImportError):
        backends["rust"] = _rust_backend()
    return backends


def benchmark_upde_moving_frame_polyglot_gate(
    *,
    n: int = 8,
    n_steps: int = 8,
    calls: int = 1,
    seed: int = 2026,
) -> dict[str, Any]:
    problem = _problem(n=n, n_steps=n_steps, seed=seed)
    phases = problem["phases"]
    positions = problem["positions"]
    omega = problem["omega"]
    velocities = problem["velocities"]
    knm = problem["knm"]
    alpha = problem["alpha"]
    dt = float(problem["dt"][0])
    k_base = float(problem["spatial_k_base"][0])
    decay_form = int(problem["spatial_decay_form"][0])
    decay_exponent = float(problem["spatial_decay_exponent"][0])
    decay_length_scale = float(problem["spatial_decay_length_scale"][0])
    spatial_epsilon = float(problem["spatial_epsilon"][0])
    strength = float(problem["strength"][0])
    epsilon = float(problem["epsilon"][0])
    modulator = SpatialCouplingModulator(K_base=k_base)
    reference = moving_frame_run_python(
        phases,
        positions,
        omega,
        knm,
        alpha,
        velocities,
        modulator.K_base,
        decay_form,
        modulator.decay_exponent,
        modulator.decay_length_scale,
        modulator.epsilon,
        strength,
        epsilon,
        0.0,
        0.0,
        dt,
        "rk4",
        1,
        1.0e-6,
        1.0e-3,
    )
    expected_positions = _expected_final_positions(positions, velocities, dt)
    reference_kinematic_residual = _kinematic_residual_max(
        reference,
        n=n,
        expected_positions=expected_positions,
    )

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
                    "kinematic_residual_max_m": None,
                    "kinematic_residual_passed": False,
                    "tolerance": TOLERANCES[backend],
                    "unavailable_reason": "moving-frame backend is not installed",
                }
            )
            continue
        try:
            elapsed_start = time.perf_counter()
            got = reference
            for _ in range(calls):
                got = fn(
                    phases.copy(),
                    positions.copy(),
                    omega,
                    knm,
                    alpha,
                    velocities,
                    k_base,
                    decay_form,
                    decay_exponent,
                    decay_length_scale,
                    spatial_epsilon,
                    strength,
                    epsilon,
                    0.0,
                    0.0,
                    dt,
                    "rk4",
                    1,
                    1.0e-6,
                    1.0e-3,
                )
            elapsed = time.perf_counter() - elapsed_start
            max_abs = float(np.max(np.abs(got - reference)))
            kinematic_residual = _kinematic_residual_max(
                got,
                n=n,
                expected_positions=expected_positions,
            )
            kinematic_passed = kinematic_residual <= KINEMATIC_RESIDUAL_TOLERANCE_M
            passed = max_abs <= TOLERANCES[backend] and kinematic_passed
            available_count += 1
            parity_pass_count += int(passed)
            records.append(
                {
                    "backend": backend,
                    "status": "available",
                    "parity_passed": passed,
                    "max_abs_error": max_abs,
                    "ms_per_call": 1000.0 * elapsed / max(1, calls),
                    "matrix_sha256": _hash_array(got),
                    "kinematic_residual_max_m": kinematic_residual,
                    "kinematic_residual_passed": kinematic_passed,
                    "tolerance": TOLERANCES[backend],
                    "unavailable_reason": "",
                }
            )
        except (ImportError, OSError, ValueError, AttributeError) as exc:
            records.append(
                {
                    "backend": backend,
                    "status": "unavailable",
                    "parity_passed": False,
                    "max_abs_error": None,
                    "ms_per_call": None,
                    "matrix_sha256": None,
                    "kinematic_residual_max_m": None,
                    "kinematic_residual_passed": False,
                    "tolerance": TOLERANCES[backend],
                    "unavailable_reason": str(exc),
                }
            )
    wall_time = time.perf_counter() - start_all
    all_available_passed = all(
        bool(row["parity_passed"]) for row in records if row["status"] == "available"
    )
    kinematic_residual_contract_passed = bool(
        reference_kinematic_residual <= KINEMATIC_RESIDUAL_TOLERANCE_M
        and all(
            bool(row["kinematic_residual_passed"])
            for row in records
            if row["status"] == "available"
        )
    )
    acceptance = bool(
        all_available_passed
        and available_count >= 1
        and kinematic_residual_contract_passed
    )
    return {
        "suite": "upde_moving_frame_polyglot_gate",
        "n": n,
        "n_steps": n_steps,
        "calls": calls,
        "seed": seed,
        "backend_count": len(BACKEND_ORDER),
        "available_backend_count": available_count,
        "unavailable_backend_count": len(BACKEND_ORDER) - available_count,
        "parity_pass_count": parity_pass_count,
        "all_available_passed": int(all_available_passed),
        "acceptance_passed": int(acceptance),
        "reference_matrix_sha256": _hash_array(reference),
        "omega_schedule_sha256": _hash_array(omega),
        "velocity_schedule_sha256": _hash_array(velocities),
        "position_initial_sha256": _hash_array(positions),
        "expected_final_position_sha256": _hash_array(expected_positions),
        "reference_kinematic_residual_max_m": reference_kinematic_residual,
        "kinematic_residual_contract_passed": int(
            kinematic_residual_contract_passed
        ),
        "max_abs_velocity_m_per_s": float(np.max(np.abs(velocities))),
        "path_length_max_m": float(np.max(np.sum(np.abs(velocities * dt), axis=0))),
        "benchmark_sha256": sha256(
            json.dumps(records, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "backend_records_json": json.dumps(records, sort_keys=True),
        "acceptance_thresholds_json": json.dumps(
            {
                "backend_order": list(BACKEND_ORDER),
                "require_python_reference": True,
                "require_all_available_parity": True,
                "require_kinematic_residual_contract": True,
                "max_kinematic_residual_m": KINEMATIC_RESIDUAL_TOLERANCE_M,
                "production_timing_claim": False,
            },
            sort_keys=True,
        ),
        "benchmark_evidence_kind": "local_regression_non_isolated",
        "isolation_method": "none",
        "production_timing_claim": 0,
        "wall_time_s": wall_time,
        "steps_per_second": (n_steps * calls) / wall_time if wall_time > 0.0 else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parity-gate", action="store_true")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--calls", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = benchmark_upde_moving_frame_polyglot_gate(
        n=args.n,
        n_steps=args.steps,
        calls=args.calls,
        seed=args.seed,
    )
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.parity_gate and result["acceptance_passed"] != 1:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
