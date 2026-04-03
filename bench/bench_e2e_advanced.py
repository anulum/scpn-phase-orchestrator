# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator -- End-to-End Advanced Benchmarks

import time
import numpy as np
from scpn_phase_orchestrator.upde.sparse_engine import SparseUPDEEngine
from scpn_phase_orchestrator.ssgf.pgbo import PGBO
from spo_kernel import PyUPDEStepper, PyActiveInferenceAgent

def run_bench():
    print(f"{'Experiment':<40} | {'Scale':<15} | {'Latency':<15}")
    print("-" * 76)

    # 1. Sparse Scalability (N=10,000)
    n_sparse = 10000
    engine_sparse = SparseUPDEEngine(n_sparse, dt=0.01, method='euler')
    phases = np.random.uniform(0, 2*np.pi, n_sparse).astype(np.float64)
    omegas = np.ones(n_sparse).astype(np.float64)
    # Correct CSR for 10 edges per node
    row_ptr = np.linspace(0, n_sparse * 10, n_sparse + 1).astype(np.uint64)
    col_indices = np.random.randint(0, n_sparse, n_sparse * 10).astype(np.uint64)
    knm_values = np.random.uniform(0, 0.1, n_sparse * 10).astype(np.float64)
    alpha_values = np.zeros(n_sparse * 10).astype(np.float64)
    
    t0 = time.perf_counter()
    for _ in range(100):
        phases = engine_sparse.step(phases, omegas, row_ptr, col_indices, knm_values, 0.0, 0.0, alpha_values)
    t1 = time.perf_counter()
    print(f"{'Sparse UPDE Integration (Rust)':<40} | N={n_sparse:<13} | {(t1-t0)/100*1e6:>8.2f} us/step")

    # 2. Plasticity Overhead (Dense)
    n_dense = 256
    engine_dense = PyUPDEStepper(n_dense, dt=0.01, method='euler')
    phases = np.random.uniform(0, 2*np.pi, n_dense).astype(np.float64)
    omegas = np.ones(n_dense).astype(np.float64)
    knm = np.random.uniform(0, 0.1, (n_dense, n_dense)).astype(np.float64).ravel()
    alpha = np.zeros((n_dense, n_dense)).astype(np.float64).ravel()
    
    # Baseline
    t0 = time.perf_counter()
    for _ in range(100):
        phases = engine_dense.step(phases, omegas, knm, 0.0, 0.0, alpha)
    t1 = time.perf_counter()
    base_latency = (t1-t0)/100*1e6
    print(f"{'Dense Integration Baseline (Rust)':<40} | N={n_dense:<13} | {base_latency:>8.2f} us/step")

    # With Plasticity
    engine_dense.set_plasticity(lr=0.01, decay=0.001, modulator=1.0)
    t0 = time.perf_counter()
    for _ in range(100):
        phases = engine_dense.step(phases, omegas, knm, 0.0, 0.0, alpha)
    t1 = time.perf_counter()
    plast_latency = (t1-t0)/100*1e6
    print(f"{'Dense Integration + Plasticity (Rust)':<40} | N={n_dense:<13} | {plast_latency:>8.2f} us/step")
    print(f"{'  Plasticity Overhead':<40} | {'':<15} | {plast_latency - base_latency:>8.2f} us/step")

    # 3. Active Inference Control
    agent = PyActiveInferenceAgent(n_hidden=16, target_r=0.5, lr=1.0)
    t0 = time.perf_counter()
    for _ in range(1000):
        agent.control(r_obs=0.6, psi_obs=0.1, dt=0.01)
    t1 = time.perf_counter()
    print(f"{'Active Inference Control Step (Rust)':<40} | Hidden=16 {'':<5} | {(t1-t0)/1000*1e6:>8.2f} us/step")

    # 4. Gauged PGBO (Metric Tensor Computation)
    pgbo = PGBO()
    n_pgbo = 256
    W = np.random.uniform(0, 1, (n_pgbo, n_pgbo)).astype(np.float64)
    phases = np.random.uniform(0, 2*np.pi, n_pgbo).astype(np.float64)
    t0 = time.perf_counter()
    for _ in range(10):
        pgbo.observe(phases, W)
    t1 = time.perf_counter()
    print(f"{'Gauged PGBO Curvature (Python/NumPy)':<40} | N={n_pgbo:<13} | {(t1-t0)/10*1e3:>8.2f} ms/obs")

if __name__ == "__main__":
    run_bench()
