import time
import numpy as np
from scpn_phase_orchestrator.upde.swarmalator import SwarmalatorEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

def run_swarmalator_bench(n=500, n_steps=50):
    # n=500 because Swarmalator is O(N^2) and heavier than Kuramoto
    rng = np.random.default_rng(42)
    pos = rng.uniform(-1, 1, (n, 2))
    phases = rng.uniform(0, 2 * np.pi, n)
    omegas = rng.uniform(-0.1, 0.1, n)
    
    engine = SwarmalatorEngine(n, dim=2, dt=0.01)
    
    t0 = time.perf_counter()
    for _ in range(n_steps):
        pos, phases = engine.step(pos, phases, omegas)
    elapsed = time.perf_counter() - t0
    
    R, _ = compute_order_parameter(phases)
    print(f"Swarmalator Engine (N={n}): {n_steps/elapsed:.1f} steps/s, R={R:.4f}")

if __name__ == "__main__":
    run_swarmalator_bench(n=500)
