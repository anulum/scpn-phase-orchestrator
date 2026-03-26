# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — GPU benchmark suite (JarvisLabs)

"""Self-contained GPU benchmark suite for SPO nn/ module.

Designed for cloud GPU instances with built-in recovery:
- Validates environment before running anything
- Saves results after EACH benchmark (not at the end)
- Can resume from where it left off (checks existing results)
- Prints download commands at the end
- Never destroys instances — that's manual

Usage on GPU instance:
    cd /home/user/scpn-phase-orchestrator
    pip install -e ".[nn]"
    python tools/gpu_benchmark.py

Results saved to: benchmarks/results/gpu_benchmark_YYYY-MM-DD.json
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ──────────────────────────────────────────────────
# Phase 0: Environment validation (before ANY import)
# ──────────────────────────────────────────────────

RESULTS_DIR = Path("benchmarks/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / f"gpu_benchmark_{datetime.now(tz=timezone.utc).strftime('%Y-%m-%d')}.json"


def _load_existing() -> dict:
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {"benchmarks": {}, "metadata": {}}


def _save(data: dict) -> None:
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  [SAVED] {RESULTS_FILE}")


def validate_environment() -> dict:
    """Check GPU, JAX, imports. Returns metadata dict or exits."""
    errors = []
    metadata: dict = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "python": sys.version,
    }

    # Check JAX
    try:
        import jax

        metadata["jax_version"] = jax.__version__
        devices = jax.devices()
        metadata["jax_devices"] = [str(d) for d in devices]
        gpu_devices = [d for d in devices if d.platform == "gpu"]
        if not gpu_devices:
            errors.append("JAX found no GPU devices. Install jax[cuda12].")
        else:
            metadata["gpu_count"] = len(gpu_devices)
            metadata["gpu_name"] = str(gpu_devices[0])
    except ImportError:
        errors.append("JAX not installed. Run: pip install jax[cuda12]")

    # Check equinox
    try:
        import equinox

        metadata["equinox_version"] = equinox.__version__
    except ImportError:
        errors.append("equinox not installed. Run: pip install equinox")

    # Check SPO
    try:
        import scpn_phase_orchestrator

        metadata["spo_version"] = scpn_phase_orchestrator.__version__
    except ImportError:
        errors.append("scpn-phase-orchestrator not installed. Run: pip install -e '.[nn]'")

    # Check disk space
    import shutil

    usage = shutil.disk_usage("/")
    free_gb = usage.free / (1024**3)
    metadata["disk_free_gb"] = round(free_gb, 1)
    if free_gb < 2:
        errors.append(f"Low disk space: {free_gb:.1f} GB free")

    if errors:
        print("=" * 60)
        print("ENVIRONMENT VALIDATION FAILED")
        print("=" * 60)
        for e in errors:
            print(f"  ERROR: {e}")
        print()
        print("Fix these issues and re-run.")
        sys.exit(1)

    print("=" * 60)
    print("ENVIRONMENT OK")
    print("=" * 60)
    for k, v in metadata.items():
        print(f"  {k}: {v}")
    print()
    return metadata


# ──────────────────────────────────────────────────
# Benchmark functions (each self-contained)
# ──────────────────────────────────────────────────


def bench_kuramoto_layer_forward(sizes: list[int] | None = None) -> dict:
    """Benchmark KuramotoLayer forward pass at various N."""
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    from scpn_phase_orchestrator.nn.kuramoto_layer import KuramotoLayer

    if sizes is None:
        sizes = [8, 16, 32, 64, 128, 256, 512]

    results = {}
    for N in sizes:
        try:
            key = jr.PRNGKey(42)
            layer = KuramotoLayer(N, dt=0.01, key=key)
            phases = jr.uniform(key, (N,), minval=0, maxval=2 * jnp.pi)
            omegas = jr.normal(key, (N,))

            # Warmup (JIT compilation)
            _ = layer(phases, omegas)

            # Timed runs
            n_runs = 100
            t0 = time.perf_counter()
            for _ in range(n_runs):
                phases = layer(phases, omegas)
            jax.block_until_ready(phases)
            elapsed = time.perf_counter() - t0

            results[str(N)] = {
                "n_oscillators": N,
                "mean_step_us": round(elapsed / n_runs * 1e6, 2),
                "steps_per_sec": round(n_runs / elapsed, 1),
                "status": "ok",
            }
            print(f"  KuramotoLayer N={N}: {elapsed/n_runs*1e6:.1f} us/step")
        except Exception as e:
            results[str(N)] = {"n_oscillators": N, "status": "error", "error": str(e)}
            print(f"  KuramotoLayer N={N}: ERROR {e}")

    return results


def bench_stuart_landau_layer_forward(sizes: list[int] | None = None) -> dict:
    """Benchmark StuartLandauLayer forward pass."""
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    from scpn_phase_orchestrator.nn.stuart_landau_layer import StuartLandauLayer

    if sizes is None:
        sizes = [8, 16, 32, 64, 128, 256]

    results = {}
    for N in sizes:
        try:
            key = jr.PRNGKey(0)
            layer = StuartLandauLayer(N, dt=0.01, key=key)
            phases = jr.uniform(key, (N,), minval=0, maxval=2 * jnp.pi)
            amplitudes = jnp.ones(N)
            omegas = jr.normal(key, (N,))

            _ = layer(phases, amplitudes, omegas)

            n_runs = 100
            t0 = time.perf_counter()
            for _ in range(n_runs):
                phases, amplitudes = layer(phases, amplitudes, omegas)
            jax.block_until_ready(phases)
            elapsed = time.perf_counter() - t0

            results[str(N)] = {
                "n_oscillators": N,
                "mean_step_us": round(elapsed / n_runs * 1e6, 2),
                "steps_per_sec": round(n_runs / elapsed, 1),
                "status": "ok",
            }
            print(f"  StuartLandauLayer N={N}: {elapsed/n_runs*1e6:.1f} us/step")
        except Exception as e:
            results[str(N)] = {"n_oscillators": N, "status": "error", "error": str(e)}
            print(f"  StuartLandauLayer N={N}: ERROR {e}")

    return results


def bench_inverse_coupling(sizes: list[int] | None = None) -> dict:
    """Benchmark coupling inference accuracy."""
    import jax.numpy as jnp
    import jax.random as jr
    import numpy as np

    from scpn_phase_orchestrator.nn.functional import kuramoto_forward, order_parameter
    from scpn_phase_orchestrator.nn.inverse import infer_coupling

    if sizes is None:
        sizes = [4, 8, 16, 32]

    results = {}
    for N in sizes:
        try:
            key = jr.PRNGKey(42)
            k1, k2, k3 = jr.split(key, 3)

            # Ground truth coupling
            K_true = jr.normal(k1, (N, N)) * 0.3
            K_true = (K_true + K_true.T) / 2
            K_true = K_true.at[jnp.diag_indices(N)].set(0.0)

            # Generate synthetic phase data
            phases_init = jr.uniform(k2, (N,), minval=0, maxval=2 * jnp.pi)
            omegas = jr.normal(k3, (N,)) * 0.1
            alpha = jnp.zeros((N, N))
            dt = 0.01

            trajectory = kuramoto_forward(phases_init, omegas, K_true, alpha, dt, 500)

            # Infer coupling
            K_est = infer_coupling(trajectory, omegas, dt, n_steps=200, lr=0.01)

            # Correlation between true and estimated
            corr = float(np.corrcoef(
                np.array(K_true).ravel(),
                np.array(K_est).ravel(),
            )[0, 1])

            # RMSE
            rmse = float(jnp.sqrt(jnp.mean((K_true - K_est) ** 2)))

            R_true = float(order_parameter(trajectory[-1]))

            results[str(N)] = {
                "n_oscillators": N,
                "correlation": round(corr, 4),
                "rmse": round(rmse, 4),
                "R_final": round(R_true, 4),
                "status": "ok",
            }
            print(f"  Inverse N={N}: corr={corr:.3f}, RMSE={rmse:.4f}")
        except Exception as e:
            results[str(N)] = {"n_oscillators": N, "status": "error", "error": str(e)}
            print(f"  Inverse N={N}: ERROR {e}")

    return results


def bench_oim_coloring() -> dict:
    """Benchmark OIM graph coloring quality."""
    import jax.numpy as jnp
    import jax.random as jr

    from scpn_phase_orchestrator.nn.oim import (
        coloring_violations,
        extract_coloring,
        oim_forward,
    )

    results = {}
    for N, n_colors in [(8, 3), (16, 4), (32, 4)]:
        try:
            key = jr.PRNGKey(42)
            # Random graph (Erdos-Renyi p=0.3)
            k1, k2 = jr.split(key)
            adj = (jr.uniform(k1, (N, N)) < 0.3).astype(jnp.float32)
            adj = (adj + adj.T) / 2
            adj = adj.at[jnp.diag_indices(N)].set(0.0)

            phases_init = jr.uniform(k2, (N,), minval=0, maxval=2 * jnp.pi)
            omegas = jnp.zeros(N)
            K = -adj  # repulsive coupling for coloring

            trajectory = oim_forward(phases_init, omegas, K, jnp.zeros((N, N)), 0.01, 1000)
            coloring = extract_coloring(trajectory[-1], n_colors)
            violations = int(coloring_violations(coloring, adj))
            n_edges = int(jnp.sum(adj > 0)) // 2

            results[f"{N}_{n_colors}"] = {
                "n_nodes": N,
                "n_colors": n_colors,
                "n_edges": n_edges,
                "violations": violations,
                "violation_rate": round(violations / max(n_edges, 1), 4),
                "status": "ok",
            }
            print(f"  OIM N={N}, k={n_colors}: {violations}/{n_edges} violations")
        except Exception as e:
            results[f"{N}_{n_colors}"] = {"status": "error", "error": str(e)}
            print(f"  OIM N={N}: ERROR {e}")

    return results


def bench_jax_engine_scaling() -> dict:
    """Benchmark JAX engine vs NumPy at various N."""
    import time

    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import numpy as np

    results = {}
    for N in [16, 32, 64, 128, 256, 512]:
        try:
            key = jr.PRNGKey(42)
            phases_jax = jr.uniform(key, (N,), minval=0, maxval=2 * jnp.pi)
            omegas_jax = jr.normal(key, (N,))
            knm_jax = jnp.ones((N, N)) * 0.5 / N
            knm_jax = knm_jax.at[jnp.diag_indices(N)].set(0.0)
            alpha_jax = jnp.zeros((N, N))

            from scpn_phase_orchestrator.nn.functional import kuramoto_forward

            # JAX warmup
            _ = kuramoto_forward(phases_jax, omegas_jax, knm_jax, alpha_jax, 0.01, 10)

            n_steps = 500
            t0 = time.perf_counter()
            traj = kuramoto_forward(phases_jax, omegas_jax, knm_jax, alpha_jax, 0.01, n_steps)
            jax.block_until_ready(traj)
            jax_time = time.perf_counter() - t0

            # NumPy comparison
            phases_np = np.array(phases_jax)
            omegas_np = np.array(omegas_jax)
            knm_np = np.array(knm_jax)
            alpha_np = np.array(alpha_jax)

            t0 = time.perf_counter()
            p = phases_np.copy()
            for _ in range(n_steps):
                diff = p[np.newaxis, :] - p[:, np.newaxis] - alpha_np
                coupling = np.sum(knm_np * np.sin(diff), axis=1)
                p = p + 0.01 * (omegas_np + coupling)
            numpy_time = time.perf_counter() - t0

            speedup = numpy_time / jax_time if jax_time > 0 else 0

            results[str(N)] = {
                "n_oscillators": N,
                "n_steps": n_steps,
                "jax_time_ms": round(jax_time * 1000, 2),
                "numpy_time_ms": round(numpy_time * 1000, 2),
                "speedup": round(speedup, 2),
                "status": "ok",
            }
            print(f"  Engine N={N}: JAX={jax_time*1000:.1f}ms NumPy={numpy_time*1000:.1f}ms → {speedup:.1f}x")
        except Exception as e:
            results[str(N)] = {"n_oscillators": N, "status": "error", "error": str(e)}
            print(f"  Engine N={N}: ERROR {e}")

    return results


# ──────────────────────────────────────────────────
# Main runner with checkpoint/resume
# ──────────────────────────────────────────────────

BENCHMARKS = [
    ("kuramoto_forward", bench_kuramoto_layer_forward),
    ("stuart_landau_forward", bench_stuart_landau_layer_forward),
    ("inverse_coupling", bench_inverse_coupling),
    ("oim_coloring", bench_oim_coloring),
    ("jax_vs_numpy", bench_jax_engine_scaling),
]


def main() -> int:
    print("=" * 60)
    print("SPO GPU BENCHMARK SUITE")
    print("=" * 60)
    print()

    metadata = validate_environment()

    data = _load_existing()
    data["metadata"] = metadata
    _save(data)

    for name, func in BENCHMARKS:
        if name in data["benchmarks"] and data["benchmarks"][name].get("_complete"):
            print(f"[SKIP] {name} (already complete, delete from JSON to re-run)")
            continue

        # Disk check before each benchmark
        import shutil

        free_gb = shutil.disk_usage("/").free / (1024**3)
        if free_gb < 2:
            print(f"ABORT: Disk nearly full ({free_gb:.1f} GB free). Download results and free space.")
            _save(data)
            return 1

        print(f"\n{'─' * 40}")
        print(f"Running: {name}")
        print(f"{'─' * 40}")

        t0 = time.perf_counter()
        try:
            result = func()
            elapsed = time.perf_counter() - t0
            data["benchmarks"][name] = {
                **result,
                "_complete": True,
                "_elapsed_s": round(elapsed, 2),
            }
        except Exception as e:
            elapsed = time.perf_counter() - t0
            data["benchmarks"][name] = {
                "_complete": False,
                "_error": str(e),
                "_elapsed_s": round(elapsed, 2),
            }
            print(f"  BENCHMARK FAILED: {e}")

        # Save after EACH benchmark (crash recovery)
        _save(data)

    print()
    print("=" * 60)
    print("ALL BENCHMARKS COMPLETE")
    print("=" * 60)
    print(f"Results: {RESULTS_FILE}")
    print()
    print("Download command (run from LOCAL machine):")
    print(f"  scp user@<INSTANCE_IP>:~/scpn-phase-orchestrator/{RESULTS_FILE} .")
    print()
    print("DO NOT destroy the instance until results are downloaded and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
