import time

import numpy as np
from scipy.sparse import csr_matrix

from scpn_phase_orchestrator.upde.order_params import compute_order_parameter
from scpn_phase_orchestrator.upde.sparse_engine import SparseUPDEEngine


def run_sparse_bench(n=1000, density=0.01, n_steps=100):
    rng = np.random.default_rng(42)
    phases = rng.uniform(0, 2 * np.pi, n)
    omegas = np.ones(n)

    # Create random sparse matrix
    num_entries = int(n * n * density)
    rows = rng.integers(0, n, num_entries)
    cols = rng.integers(0, n, num_entries)
    data = rng.uniform(0, 0.5 / n, num_entries)
    knm_sparse = csr_matrix((data, (rows, cols)), shape=(n, n))
    knm_sparse.eliminate_zeros()

    alpha_values = np.zeros(knm_sparse.nnz)

    engine = SparseUPDEEngine(n, dt=0.01)

    row_ptr = knm_sparse.indptr
    col_indices = knm_sparse.indices
    knm_values = knm_sparse.data

    t0 = time.perf_counter()
    for _ in range(n_steps):
        phases = engine.step(
            phases, omegas, row_ptr, col_indices, knm_values, 0.0, 0.0, alpha_values
        )
    elapsed = time.perf_counter() - t0

    R, _ = compute_order_parameter(phases)
    rate = n_steps / elapsed
    print(f"Sparse Engine (N={n}, density={density}): {rate:.1f} steps/s, R={R:.4f}")


if __name__ == "__main__":
    run_sparse_bench(n=1000, density=0.01)
    run_sparse_bench(n=10000, density=0.001)
