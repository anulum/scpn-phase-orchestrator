# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Augmented Components Benchmarks

import time

import numpy as np
from spo_kernel import PyActiveInferenceAgent

from scpn_phase_orchestrator.ssgf.pgbo import PGBO
from scpn_phase_orchestrator.upde.sparse_engine import SparseUPDEEngine


def dense_to_csr(knm, alpha):
    n = knm.shape[0]
    row_ptr = [0]
    col_indices = []
    knm_values = []
    alpha_values = []
    for i in range(n):
        for j in range(n):
            if knm[i, j] != 0:
                col_indices.append(j)
                knm_values.append(knm[i, j])
                alpha_values.append(alpha[i, j])
        row_ptr.append(len(col_indices))
    return (
        np.array(row_ptr, dtype=np.uint64),
        np.array(col_indices, dtype=np.uint64),
        np.array(knm_values, dtype=np.float64),
        np.array(alpha_values, dtype=np.float64),
    )


def bench_sparse_engine(n=1024, density=0.01):
    engine = SparseUPDEEngine(n, dt=0.01, method="euler")
    phases = np.random.uniform(0, 2 * np.pi, n)
    omegas = np.ones(n)

    # Create sparse coupling
    knm = np.random.choice([0, 0.1], size=(n, n), p=[1 - density, density])
    alpha = np.zeros((n, n))
    row_ptr, col_indices, knm_values, alpha_values = dense_to_csr(knm, alpha)

    # Warmup
    for _ in range(10):
        engine.step(
            phases, omegas, row_ptr, col_indices, knm_values, 0.0, 0.0, alpha_values
        )

    start = time.perf_counter()
    iters = 100
    for _ in range(iters):
        engine.step(
            phases, omegas, row_ptr, col_indices, knm_values, 0.0, 0.0, alpha_values
        )
    end = time.perf_counter()

    avg_time_us = (end - start) / iters * 1e6
    print(
        f"SparseUPDEEngine (N={n}, edges={len(knm_values)}): {avg_time_us:.2f} us/step"
    )


def bench_pgbo(n=256):
    pgbo = PGBO()
    phases = np.random.uniform(0, 2 * np.pi, n)
    W = np.random.uniform(0, 1, (n, n))

    # Warmup
    for _ in range(10):
        pgbo.observe(phases, W)

    start = time.perf_counter()
    iters = 100
    for _ in range(iters):
        pgbo.observe(phases, W)
    end = time.perf_counter()

    avg_time_us = (end - start) / iters * 1e6
    print(f"Gauged PGBO (N={n}): {avg_time_us:.2f} us/observation")


def bench_active_inference():
    agent = PyActiveInferenceAgent(n_hidden=8, target_r=0.5, lr=1.0)

    start = time.perf_counter()
    iters = 1000
    for _ in range(iters):
        agent.control(r_obs=0.6, psi_obs=0.1, dt=0.01)
    end = time.perf_counter()

    avg_time_us = (end - start) / iters * 1e6
    print(f"ActiveInferenceAgent (Rust): {avg_time_us:.2f} us/control_step")


if __name__ == "__main__":
    print("--- Augmented Components Benchmarks ---")
    bench_sparse_engine(n=1024, density=0.01)
    bench_sparse_engine(n=4096, density=0.001)
    bench_pgbo(n=256)
    bench_active_inference()
