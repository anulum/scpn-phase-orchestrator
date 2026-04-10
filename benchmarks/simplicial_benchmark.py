import time

import numpy as np

from scpn_phase_orchestrator.upde.order_params import compute_order_parameter
from scpn_phase_orchestrator.upde.simplicial import SimplicialEngine


def run_simplicial_bench(n=1000, n_steps=100):
    rng = np.random.default_rng(42)
    phases = rng.uniform(0, 2 * np.pi, n)
    omegas = np.ones(n)
    knm = rng.uniform(0, 0.5 / n, (n, n))
    alpha = np.zeros((n, n))

    engine = SimplicialEngine(n, dt=0.01, sigma2=0.5)

    t0 = time.perf_counter()
    for _ in range(n_steps):
        phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)
    elapsed = time.perf_counter() - t0

    R, _ = compute_order_parameter(phases)
    print(f"Simplicial Engine (N={n}): {n_steps / elapsed:.1f} steps/s, R={R:.4f}")


if __name__ == "__main__":
    run_simplicial_bench(n=1000)
