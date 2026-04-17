# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
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
_date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
RESULTS_FILE = RESULTS_DIR / f"gpu_benchmark_{_date_str}.json"


def _load_existing() -> dict:
    if RESULTS_FILE.exists():
        with RESULTS_FILE.open() as f:
            return json.load(f)
    return {"benchmarks": {}, "metadata": {}}


def _save(data: dict) -> None:
    with RESULTS_FILE.open("w") as f:
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
        errors.append(
            "scpn-phase-orchestrator not installed. Run: pip install -e '.[nn]'"
        )

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

            # JIT compilation timing
            t_jit0 = time.perf_counter()
            out = layer(phases)
            jax.block_until_ready(out)
            jit_time = time.perf_counter() - t_jit0

            # 4 more warmup calls
            for _ in range(4):
                out = layer(phases)
            jax.block_until_ready(out)

            # Timed runs
            n_runs = 100
            t0 = time.perf_counter()
            for _ in range(n_runs):
                phases = layer(phases)
            jax.block_until_ready(phases)
            elapsed = time.perf_counter() - t0

            results[str(N)] = {
                "n_oscillators": N,
                "jit_compile_ms": round(jit_time * 1000, 2),
                "mean_step_us": round(elapsed / n_runs * 1e6, 2),
                "steps_per_sec": round(n_runs / elapsed, 1),
                "status": "ok",
            }
            us = elapsed / n_runs * 1e6
            print(f"  KuramotoLayer N={N}: JIT={jit_time * 1000:.0f}ms, {us:.1f}us")
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

            _ = layer(phases, amplitudes)

            n_runs = 100
            t0 = time.perf_counter()
            for _ in range(n_runs):
                phases, amplitudes = layer(phases, amplitudes)
            jax.block_until_ready(phases)
            elapsed = time.perf_counter() - t0

            results[str(N)] = {
                "n_oscillators": N,
                "mean_step_us": round(elapsed / n_runs * 1e6, 2),
                "steps_per_sec": round(n_runs / elapsed, 1),
                "status": "ok",
            }
            print(f"  StuartLandauLayer N={N}: {elapsed / n_runs * 1e6:.1f} us/step")
        except Exception as e:
            results[str(N)] = {"n_oscillators": N, "status": "error", "error": str(e)}
            print(f"  StuartLandauLayer N={N}: ERROR {e}")

    return results


def bench_inverse_coupling(sizes: list[int] | None = None) -> dict:
    """Benchmark coupling inference accuracy.

    Uses shorter trajectories (100 steps) for better gradient signal through
    the ODE solver, more optimisation epochs (500), and no L1 penalty to
    avoid fighting the reconstruction objective.
    """
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
            k1, k2 = jr.split(key, 2)

            # Ground truth: moderate coupling, zero natural frequencies
            K_true = jr.normal(k1, (N, N)) * 0.3
            K_true = (K_true + K_true.T) / 2
            K_true = K_true.at[jnp.diag_indices(N)].set(0.0)

            phases_init = jr.uniform(k2, (N,), minval=0, maxval=2 * jnp.pi)
            dt = 0.01
            omegas = jnp.zeros(N)

            # Short trajectory (100 steps) — gradients through 500 steps vanish
            final_phases, trajectory = kuramoto_forward(
                phases_init, omegas, K_true, dt, 100
            )

            # More epochs, lower lr, no L1 penalty for reconstruction
            K_est, omegas_est, losses = infer_coupling(
                trajectory, dt, n_epochs=500, lr=0.005, l1_weight=0.0
            )

            corr = float(
                np.corrcoef(
                    np.array(K_true).ravel(),
                    np.array(K_est).ravel(),
                )[0, 1]
            )

            rmse = float(jnp.sqrt(jnp.mean((K_true - K_est) ** 2)))
            R_true = float(order_parameter(final_phases))

            results[str(N)] = {
                "n_oscillators": N,
                "correlation": round(corr, 4),
                "rmse": round(rmse, 4),
                "R_final": round(R_true, 4),
                "final_loss": round(float(losses[-1]), 6) if losses else None,
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
            k1, k2 = jr.split(key)
            adj = (jr.uniform(k1, (N, N)) < 0.3).astype(jnp.float32)
            adj = (adj + adj.T) / 2
            adj = adj.at[jnp.diag_indices(N)].set(0.0)

            phases_init = jr.uniform(k2, (N,), minval=0, maxval=2 * jnp.pi)

            # oim_forward(phases, adjacency, n_colors, dt, n_steps)
            final_phases, trajectory = oim_forward(
                phases_init, adj, n_colors, 0.01, 1000
            )
            coloring = extract_coloring(final_phases, n_colors)
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

    from scpn_phase_orchestrator.nn.functional import kuramoto_forward

    results = {}
    for N in [16, 32, 64, 128, 256, 512, 1024, 2048]:
        try:
            key = jr.PRNGKey(42)
            phases_jax = jr.uniform(key, (N,), minval=0, maxval=2 * jnp.pi)
            omegas_jax = jr.normal(key, (N,))
            knm_jax = jnp.ones((N, N)) * 0.5 / N
            knm_jax = knm_jax.at[jnp.diag_indices(N)].set(0.0)

            # JAX warmup — kuramoto_forward(phases, omegas, K, dt, n_steps)
            _ = kuramoto_forward(phases_jax, omegas_jax, knm_jax, 0.01, 10)

            n_steps = 500
            t0 = time.perf_counter()
            final, traj = kuramoto_forward(
                phases_jax,
                omegas_jax,
                knm_jax,
                0.01,
                n_steps,
            )
            jax.block_until_ready(final)
            jax_time = time.perf_counter() - t0

            # NumPy comparison
            phases_np = np.array(phases_jax)
            omegas_np = np.array(omegas_jax)
            knm_np = np.array(knm_jax)

            t0 = time.perf_counter()
            p = phases_np.copy()
            for _ in range(n_steps):
                diff = p[np.newaxis, :] - p[:, np.newaxis]
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
            jms = jax_time * 1000
            nms = numpy_time * 1000
            print(
                f"  Engine N={N}: JAX={jms:.1f}ms NumPy={nms:.1f}ms -> {speedup:.1f}x"
            )
        except Exception as e:
            results[str(N)] = {"n_oscillators": N, "status": "error", "error": str(e)}
            print(f"  Engine N={N}: ERROR {e}")

    return results


def bench_batched_kuramoto() -> dict:
    """Benchmark vmap-batched Kuramoto forward — where GPU wins.

    Runs B independent initial conditions in parallel via jax.vmap.
    This amortises kernel launch overhead across the batch.
    """
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    from scpn_phase_orchestrator.nn.functional import kuramoto_forward

    results = {}
    N = 64
    n_steps = 200

    for B in [1, 4, 16, 64, 256]:
        try:
            key = jr.PRNGKey(42)
            keys = jr.split(key, B)

            phases_batch = jax.vmap(
                lambda k: jr.uniform(k, (N,), minval=0, maxval=2 * jnp.pi)
            )(keys)
            omegas = jr.normal(key, (N,))
            K = jnp.ones((N, N)) * 0.5 / N
            K = K.at[jnp.diag_indices(N)].set(0.0)

            def run_one(phases, o=omegas, k=K, ns=n_steps):
                final, _ = kuramoto_forward(phases, o, k, 0.01, ns)
                return final

            batched_run = jax.vmap(run_one)

            # Warmup
            _ = batched_run(phases_batch)

            n_runs = 20
            t0 = time.perf_counter()
            for _ in range(n_runs):
                out = batched_run(phases_batch)
            jax.block_until_ready(out)
            elapsed = time.perf_counter() - t0

            per_run_ms = elapsed / n_runs * 1000
            per_instance_us = elapsed / n_runs / B * 1e6

            results[str(B)] = {
                "batch_size": B,
                "n_oscillators": N,
                "n_steps": n_steps,
                "total_ms": round(per_run_ms, 2),
                "per_instance_us": round(per_instance_us, 2),
                "status": "ok",
            }
            print(
                f"  Batched B={B}: {per_run_ms:.1f}ms total, "
                f"{per_instance_us:.1f}us/instance"
            )
        except Exception as e:
            results[str(B)] = {"batch_size": B, "status": "error", "error": str(e)}
            print(f"  Batched B={B}: ERROR {e}")

    return results


def bench_analytical_inverse(sizes: list[int] | None = None) -> dict:
    """Benchmark analytical vs gradient inverse coupling."""
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    from scpn_phase_orchestrator.nn.functional import kuramoto_forward
    from scpn_phase_orchestrator.nn.inverse import (
        analytical_inverse,
        coupling_correlation,
        infer_coupling,
    )

    if sizes is None:
        sizes = [4, 8, 16, 32]

    results = {}
    for N in sizes:
        try:
            key = jr.PRNGKey(42)
            k1, k2 = jr.split(key, 2)
            K_true = jr.normal(k1, (N, N)) * 0.3
            K_true = (K_true + K_true.T) / 2
            K_true = K_true.at[jnp.diag_indices(N)].set(0.0)
            omegas_true = jnp.zeros(N)
            p0 = jr.uniform(k2, (N,), maxval=2 * jnp.pi)
            _, traj = kuramoto_forward(p0, omegas_true, K_true, 0.01, 200)
            observed = jnp.concatenate([p0[jnp.newaxis, :], traj], axis=0)

            # Analytical
            t0 = time.perf_counter()
            K_a, _ = analytical_inverse(observed, 0.01)
            jax.block_until_ready(K_a)
            t_analytical = time.perf_counter() - t0
            corr_a = float(coupling_correlation(K_true, K_a))

            # Gradient (shorter for speed)
            t0 = time.perf_counter()
            K_g, _, _ = infer_coupling(
                traj,
                0.01,
                n_epochs=200,
                lr=0.005,
                l1_weight=0.0,
            )
            jax.block_until_ready(K_g)
            t_gradient = time.perf_counter() - t0
            corr_g = float(coupling_correlation(K_true, K_g))

            results[str(N)] = {
                "n_oscillators": N,
                "analytical_corr": round(corr_a, 4),
                "analytical_time_s": round(t_analytical, 3),
                "gradient_corr": round(corr_g, 4),
                "gradient_time_s": round(t_gradient, 3),
                "speedup": round(t_gradient / max(t_analytical, 0.001), 1),
                "status": "ok",
            }
            print(
                f"  Inverse N={N}: analytical={corr_a:.3f}/{t_analytical:.2f}s"
                f" gradient={corr_g:.3f}/{t_gradient:.2f}s"
            )
        except Exception as e:
            results[str(N)] = {"n_oscillators": N, "status": "error", "error": str(e)}
            print(f"  Inverse N={N}: ERROR {e}")

    return results


def bench_oim_solve() -> dict:
    """Benchmark OIM solver on standard graphs."""
    import jax.numpy as jnp
    import jax.random as jr

    from scpn_phase_orchestrator.nn.oim import coloring_violations, oim_solve

    results = {}

    # K3 triangle, 3-color
    A3 = jnp.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=jnp.float32)

    # K33 bipartite, 2-color
    A6 = jnp.zeros((6, 6))
    for i in range(3):
        for j in range(3, 6):
            A6 = A6.at[i, j].set(1.0)
            A6 = A6.at[j, i].set(1.0)

    for label, adj, nc in [("K3_3color", A3, 3), ("K33_2color", A6, 2)]:
        try:
            key = jr.PRNGKey(42)
            t0 = time.perf_counter()
            colors, _, energy = oim_solve(adj, nc, key=key)
            elapsed = time.perf_counter() - t0
            v = int(coloring_violations(colors, adj))

            results[label] = {
                "n_nodes": int(adj.shape[0]),
                "n_colors": nc,
                "violations": v,
                "energy": round(energy, 4),
                "time_s": round(elapsed, 3),
                "status": "ok",
            }
            print(f"  OIM {label}: {v} violations in {elapsed:.2f}s")
        except Exception as e:
            results[label] = {"status": "error", "error": str(e)}
            print(f"  OIM {label}: ERROR {e}")

    return results


def bench_simplicial_layer(sizes: list[int] | None = None) -> dict:
    """Benchmark SimplicialKuramotoLayer forward pass."""
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    from scpn_phase_orchestrator.nn.simplicial_layer import SimplicialKuramotoLayer

    if sizes is None:
        sizes = [8, 16, 32, 64, 128, 256]

    results = {}
    for N in sizes:
        try:
            key = jr.PRNGKey(42)
            layer = SimplicialKuramotoLayer(N, dt=0.01, sigma2_init=1.0, key=key)
            phases = jr.uniform(key, (N,), minval=0, maxval=2 * jnp.pi)

            # JIT compile
            t_jit0 = time.perf_counter()
            out = layer(phases)
            jax.block_until_ready(out)
            jit_time = time.perf_counter() - t_jit0

            for _ in range(4):
                out = layer(phases)
            jax.block_until_ready(out)

            n_runs = 100
            t0 = time.perf_counter()
            for _ in range(n_runs):
                phases = layer(phases)
            jax.block_until_ready(phases)
            elapsed = time.perf_counter() - t0

            results[str(N)] = {
                "n_oscillators": N,
                "jit_compile_ms": round(jit_time * 1000, 2),
                "mean_step_us": round(elapsed / n_runs * 1e6, 2),
                "steps_per_sec": round(n_runs / elapsed, 1),
                "status": "ok",
            }
            us = elapsed / n_runs * 1e6
            print(f"  SimplicialLayer N={N}: JIT={jit_time * 1000:.0f}ms, {us:.1f}us")
        except Exception as e:
            results[str(N)] = {"n_oscillators": N, "status": "error", "error": str(e)}
            print(f"  SimplicialLayer N={N}: ERROR {e}")

    return results


# ──────────────────────────────────────────────────
# Main runner with checkpoint/resume
# ──────────────────────────────────────────────────

BENCHMARKS = [
    ("kuramoto_forward", bench_kuramoto_layer_forward),
    ("stuart_landau_forward", bench_stuart_landau_layer_forward),
    ("simplicial_forward", bench_simplicial_layer),
    ("inverse_coupling", bench_inverse_coupling),
    ("analytical_inverse", bench_analytical_inverse),
    ("oim_coloring", bench_oim_coloring),
    ("oim_solve", bench_oim_solve),
    ("jax_vs_numpy", bench_jax_engine_scaling),
    ("batched_kuramoto", bench_batched_kuramoto),
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
            print(f"[SKIP] {name} (already complete, delete to re-run)")
            continue

        # Disk check before each benchmark
        import shutil

        free_gb = shutil.disk_usage("/").free / (1024**3)
        if free_gb < 2:
            print(
                f"ABORT: Disk nearly full ({free_gb:.1f} GB free). "
                "Download results and free space."
            )
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
