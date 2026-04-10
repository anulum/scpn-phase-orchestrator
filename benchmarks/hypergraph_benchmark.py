import time

import numpy as np

from scpn_phase_orchestrator.upde.hypergraph import HypergraphEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter


def run_hypergraph_bench(n=1000, n_steps=100):
    rng = np.random.default_rng(42)
    phases = rng.uniform(0, 2 * np.pi, n)
    omegas = np.ones(n)

    engine = HypergraphEngine(n, dt=0.01)
    # Create random hyperedges
    for _ in range(n):
        # 3-body
        nodes = tuple(rng.integers(0, n, 3))
        engine.add_edge(nodes, 0.5)
        # 4-body
        nodes = tuple(rng.integers(0, n, 4))
        engine.add_edge(nodes, 0.3)

    t0 = time.perf_counter()
    for _ in range(n_steps):
        phases = engine.step(phases, omegas)
    elapsed = time.perf_counter() - t0

    R, _ = compute_order_parameter(phases)
    print(f"Hypergraph Engine (N={n}): {n_steps / elapsed:.1f} steps/s, R={R:.4f}")


if __name__ == "__main__":
    run_hypergraph_bench(n=1000)
