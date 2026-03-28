#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SCPN Phase Orchestrator — Example: Inverse Problem (Learn Coupling)
#
# Given observed phase trajectories, infer the coupling matrix K_nm.
# Uses finite-difference gradient descent on the cost 1-R.
# For GPU-accelerated JAX version, see examples/inverse_kuramoto.py.
#
# Usage: python examples/inverse_coupling_demo.py
# Requires: pip install scpn-phase-orchestrator

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.upde.adjoint import cost_R, gradient_knm_fd
from scpn_phase_orchestrator.upde.engine import UPDEEngine

TWO_PI = 2.0 * np.pi


def main() -> None:
    n = 4
    rng = np.random.default_rng(42)

    # Ground truth coupling (unknown to the learner)
    knm_true = np.array(
        [
            [0.0, 2.0, 0.5, 0.0],
            [2.0, 0.0, 0.0, 1.5],
            [0.5, 0.0, 0.0, 2.0],
            [0.0, 1.5, 2.0, 0.0],
        ]
    )

    # Generate "observed" data from ground truth
    omegas = np.array([1.0, 1.5, 0.8, 1.2])
    phases_init = rng.uniform(0, TWO_PI, n)
    eng = UPDEEngine(n, dt=0.01)
    alpha = np.zeros((n, n))

    phases_observed = phases_init.copy()
    for _ in range(200):
        phases_observed = eng.step(phases_observed, omegas, knm_true, 0.0, 0.0, alpha)

    print("Inverse Problem: Learn Coupling from Data")
    print("=" * 50)
    print("Ground truth K_nm (hidden):")
    for row in knm_true:
        print(f"  [{', '.join(f'{v:.1f}' for v in row)}]")

    # Start with uniform guess
    knm_guess = np.ones((n, n)) * 1.0
    np.fill_diagonal(knm_guess, 0.0)
    lr = 0.5

    print("\nLearning coupling via gradient descent...")
    print(f"{'Iter':>5s}  {'Cost (1-R)':>10s}  {'K error':>8s}")
    print("-" * 30)

    for iteration in range(20):
        # Compute gradient of cost w.r.t. K_nm
        grad = gradient_knm_fd(eng, phases_init, omegas, knm_guess, alpha, n_steps=50)

        # Gradient descent
        knm_guess -= lr * grad
        knm_guess = np.maximum(knm_guess, 0.0)
        np.fill_diagonal(knm_guess, 0.0)

        # Evaluate
        phases_test = phases_init.copy()
        for _ in range(50):
            phases_test = eng.step(phases_test, omegas, knm_guess, 0.0, 0.0, alpha)
        cost = cost_R(phases_test)
        k_err = np.mean(np.abs(knm_guess - knm_true))

        if iteration % 5 == 0 or iteration == 19:
            print(f"{iteration:>5d}  {cost:>10.4f}  {k_err:>8.3f}")

    print("\nLearned K_nm:")
    for row in knm_guess:
        print(f"  [{', '.join(f'{v:.1f}' for v in row)}]")

    print("\nFor JAX GPU acceleration: see examples/inverse_kuramoto.py")


if __name__ == "__main__":
    main()
