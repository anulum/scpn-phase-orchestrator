import time

import numpy as np

from scpn_phase_orchestrator.upde.inertial import InertialKuramotoEngine


def run_inertial_bench(n=1000, n_steps=100):
    rng = np.random.default_rng(42)
    theta = rng.uniform(0, 2 * np.pi, n)
    omega_dot = np.zeros(n)
    power = rng.uniform(-0.1, 0.1, n)
    knm = rng.uniform(0, 0.5 / n, (n, n))
    inertia = np.ones(n)
    damping = np.ones(n) * 0.1

    engine = InertialKuramotoEngine(n, dt=0.01)

    t0 = time.perf_counter()
    for _ in range(n_steps):
        theta, omega_dot = engine.step(theta, omega_dot, power, knm, inertia, damping)
    elapsed = time.perf_counter() - t0

    print(f"Inertial Engine (N={n}): {n_steps / elapsed:.1f} steps/s")


if __name__ == "__main__":
    run_inertial_bench(n=1000)
