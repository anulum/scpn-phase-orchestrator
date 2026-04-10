import time

import numpy as np

from scpn_phase_orchestrator.upde.delay import DelayedEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter


def run_delayed_bench(n=1000, n_steps=100, delay_steps=10):
    rng = np.random.default_rng(42)
    phases = rng.uniform(0, 2 * np.pi, n)
    omegas = np.ones(n)
    knm = rng.uniform(0, 0.5 / n, (n, n))
    alpha = np.zeros((n, n))

    engine = DelayedEngine(n, dt=0.01, delay_steps=delay_steps)

    t0 = time.perf_counter()
    phases = engine.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=n_steps)
    elapsed = time.perf_counter() - t0

    R, _ = compute_order_parameter(phases)
    rate = n_steps / elapsed
    print(f"Delayed Engine (N={n}, delay={delay_steps}): {rate:.1f} steps/s, R={R:.4f}")


if __name__ == "__main__":
    run_delayed_bench(n=1000)
