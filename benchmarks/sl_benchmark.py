import time
import numpy as np
from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

def run_sl_bench(n=1000, n_steps=100):
    rng = np.random.default_rng(42)
    phases = rng.uniform(0, 2 * np.pi, n)
    amplitudes = np.ones(n)
    state = np.concatenate([phases, amplitudes])
    omegas = np.ones(n)
    mu = np.ones(n)
    knm = rng.uniform(0, 0.5/n, (n, n))
    knm_r = rng.uniform(0, 0.1/n, (n, n))
    alpha = np.zeros((n, n))
    
    engine = StuartLandauEngine(n, dt=0.01)
    
    t0 = time.perf_counter()
    for _ in range(n_steps):
        state = engine.step(state, omegas, mu, knm, knm_r, 0.0, 0.0, alpha, 0.1)
    elapsed = time.perf_counter() - t0
    
    R, _ = compute_order_parameter(state[:n])
    print(f"Stuart-Landau Engine (N={n}): {n_steps/elapsed:.1f} steps/s, R={R:.4f}")

if __name__ == "__main__":
    run_sl_bench(n=1000)
