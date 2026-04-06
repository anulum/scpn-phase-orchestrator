import time
import numpy as np
from scpn_phase_orchestrator.monitor.recurrence import recurrence_matrix

def run_recurrence_bench(t=2000, d=10):
    rng = np.random.default_rng(42)
    trajectory = rng.uniform(0, 2 * np.pi, (t, d))
    
    t0 = time.perf_counter()
    r = recurrence_matrix(trajectory, epsilon=0.5, metric='angular')
    elapsed = time.perf_counter() - t0
    
    print(f"Recurrence Matrix (T={t}, d={d}): {elapsed:.3f}s")

if __name__ == "__main__":
    run_recurrence_bench(t=2000, d=10)
    run_recurrence_bench(t=5000, d=10)
