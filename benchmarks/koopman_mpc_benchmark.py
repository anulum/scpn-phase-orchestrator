# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Koopman-MPC QP-layer benchmark + parity gate

"""Timing and parity gate for the Koopman-MPC quadratic-programme layer.

A representative condensed model-predictive-control programme is solved by every
available QP backend; the gate records each backend's wall-clock time and the
maximum deviation of its solution from the deterministic ADMM floor, and fails if
any present backend exceeds the parity tolerance.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.actuation._qp import available_qp_backends, solve_qp
from scpn_phase_orchestrator.actuation.koopman_mpc import (
    _build_constraints,
    _condensed_prediction,
)
from scpn_phase_orchestrator.monitor.koopman_edmd import (
    KoopmanDictionary,
    fit_koopman_predictor,
)

FloatArray = NDArray[np.float64]

_PARITY_TOLERANCE = 1.0e-5


def _representative_qp() -> tuple[
    FloatArray, FloatArray, FloatArray, FloatArray, FloatArray
]:
    """Build the condensed QP of a representative oscillation-damping programme."""
    rng = np.random.default_rng(2026)
    plant_a = np.array([[0.98, 0.30], [-0.30, 0.98]])
    plant_b = np.array([[0.0], [1.0]])
    states = rng.standard_normal((600, 2))
    inputs = rng.standard_normal((600, 1))
    next_states = states @ plant_a.T + inputs @ plant_b.T
    predictor = fit_koopman_predictor(
        states,
        next_states,
        inputs,
        dictionary=KoopmanDictionary(
            kind="identity", state_dim=2, include_constant=False
        ),
        regularisation=1.0e-10,
    )
    horizon, n_state, n_input = 25, predictor.state_dim, predictor.input_dim
    psi, theta = _condensed_prediction(predictor, horizon)
    free_output = psi @ predictor.lift(np.array([2.0, 2.0]))
    stage_q = np.ones(horizon * n_state)
    stage_r = np.full(horizon * n_input, 0.01)
    weighted_theta = theta * stage_q[:, None]
    hessian = 2.0 * (theta.T @ weighted_theta + np.diag(stage_r))
    hessian = 0.5 * (hessian + hessian.T)
    linear = 2.0 * theta.T @ (stage_q * free_output)
    constraint, lower, upper = _build_constraints(
        horizon,
        n_input,
        np.full(n_input, -0.5),
        np.full(n_input, 0.5),
        None,
        None,
    )
    return hessian, linear, constraint, lower, upper


def benchmark_koopman_mpc_qp_parity_gate(*, calls: int = 5) -> dict[str, object]:
    """Record Koopman-MPC QP solve timing and parity across QP backends."""
    objective_matrix, objective_vector, constraint, lower, upper = _representative_qp()
    reference = solve_qp(
        objective_matrix, objective_vector, constraint, lower, upper, backend="admm"
    )

    records: list[dict[str, object]] = []
    for backend in available_qp_backends():
        start = time.perf_counter()
        solution = reference
        for _ in range(calls):
            solution = solve_qp(
                objective_matrix,
                objective_vector,
                constraint,
                lower,
                upper,
                backend=backend,
            )
        elapsed = (time.perf_counter() - start) / calls
        deviation = float(np.max(np.abs(solution.x - reference.x)))
        records.append(
            {
                "backend": backend,
                "seconds_per_call": elapsed,
                "iterations": solution.iterations,
                "max_abs_deviation": deviation,
                "within_tolerance": deviation <= _PARITY_TOLERANCE,
            }
        )

    parity_ok = all(record["within_tolerance"] for record in records)
    return {
        "suite": "koopman_mpc_qp_parity_gate",
        "boundary_contract": "deterministic_admm_floor",
        "parity_tolerance": _PARITY_TOLERANCE,
        "parity_ok": parity_ok,
        "backend_records": records,
        "backend_records_json": json.dumps(records, sort_keys=True),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parity-gate", action="store_true")
    parser.add_argument("--calls", type=int, default=5)
    args = parser.parse_args()
    result = benchmark_koopman_mpc_qp_parity_gate(calls=args.calls)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["parity_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
