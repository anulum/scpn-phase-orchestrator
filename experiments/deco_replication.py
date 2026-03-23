# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Deco Replication Experiment
#
# Replicates: Deco et al. whole-brain model (Stuart-Landau on HCP connectome)
# Target: 72% correlation between simulated and empirical functional connectivity
# Reference: Deco et al. 2018, Scientific Reports 8:3460

"""Deco replication: Stuart-Landau on real HCP connectome.

Runs Stuart-Landau dynamics on the HCP structural connectivity matrix,
computes simulated functional connectivity, and compares against
empirical fMRI functional connectivity.

Usage:
    python experiments/deco_replication.py
    python experiments/deco_replication.py --K-sweep 0.1 0.5 1.0 2.0 5.0
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy.stats import pearsonr

from neurolib.utils.loadData import Dataset


def run_stuart_landau(
    sc: np.ndarray,
    K: float,
    dt: float = 0.01,
    duration: float = 60.0,
    mu: float = -0.05,
    noise_sigma: float = 0.02,
    seed: int = 42,
) -> np.ndarray:
    """Run Stuart-Landau near Hopf bifurcation on structural connectivity.

    Deco's model: each region is a Stuart-Landau oscillator
    dz_j/dt = (a_j + iω_j - |z_j|²)z_j + K Σ_k C_jk (z_k - z_j) + noise

    Near bifurcation (a ≈ 0), the system operates at maximum metastability.

    Returns (n_regions, n_timepoints) complex time series.
    """
    n = sc.shape[0]
    n_steps = int(duration / dt)
    rng = np.random.default_rng(seed)

    # Natural frequencies: Deco uses ~0.05 Hz (infraslow BOLD timescale)
    # NOT gamma band — the SL model operates at the BOLD envelope timescale
    omega = rng.uniform(0.04, 0.07, n) * 2 * np.pi

    # Initialize near fixed point with small perturbation
    z = 0.01 * rng.standard_normal((n,)) + 0.01j * rng.standard_normal((n,))

    # Store BOLD-like signal (real part, downsampled)
    downsample = max(1, int(0.72 / dt))  # ~0.72s TR
    bold = []

    for step in range(n_steps):
        # Stuart-Landau dynamics
        dzdt = (mu + 1j * omega - np.abs(z) ** 2) * z

        # Global coupling: K * Σ_k C_jk (z_k - z_j) / degree_j
        # Normalization by degree prevents high-degree nodes from exploding
        degrees = sc.sum(axis=1)
        degrees[degrees < 1e-10] = 1.0
        coupling = K * (sc @ z - degrees * z) / degrees

        # Noise
        noise = noise_sigma * (
            rng.standard_normal(n) + 1j * rng.standard_normal(n)
        )

        z = z + dt * (dzdt + coupling) + np.sqrt(dt) * noise

        if step % downsample == 0:
            bold.append(z.real.copy())

    return np.array(bold).T  # (n_regions, n_timepoints)


def compute_fc(bold: np.ndarray) -> np.ndarray:
    """Compute functional connectivity as pairwise Pearson correlation."""
    n = bold.shape[0]
    fc = np.corrcoef(bold)
    np.fill_diagonal(fc, 0.0)
    return fc


def compare_fc(sim_fc: np.ndarray, emp_fc: np.ndarray) -> dict:
    """Compare simulated vs empirical FC."""
    n = sim_fc.shape[0]
    triu = np.triu_indices(n, k=1)
    sim_flat = sim_fc[triu]
    emp_flat = emp_fc[triu]

    r, p = pearsonr(sim_flat, emp_flat)
    return {
        "correlation": round(float(r), 4),
        "p_value": float(p),
        "n_pairs": len(sim_flat),
        "sim_fc_mean": round(float(sim_flat.mean()), 4),
        "emp_fc_mean": round(float(emp_flat.mean()), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Deco replication experiment")
    parser.add_argument(
        "--K-sweep",
        type=float,
        nargs="+",
        default=None,
        help="Coupling strengths to sweep",
    )
    parser.add_argument("--K", type=float, default=1.0, help="Global coupling")
    parser.add_argument("--duration", type=float, default=60.0, help="Sim seconds")
    parser.add_argument("--mu", type=float, default=-0.01, help="Bifurcation param")
    parser.add_argument("--output", type=str, default=None, help="JSON output path")
    args = parser.parse_args()

    print("Loading HCP data...")
    ds = Dataset("hcp")
    sc = ds.Cmat
    emp_fc = np.mean(ds.FCs, axis=0)  # average across subjects
    np.fill_diagonal(emp_fc, 0.0)
    print(f"  SC: {sc.shape}, FC: {emp_fc.shape}, subjects: {len(ds.FCs)}")

    K_values = args.K_sweep or [args.K]
    results = []

    for K in K_values:
        print(f"\nRunning K={K:.2f}, duration={args.duration}s, mu={args.mu}...")
        t0 = time.perf_counter()
        bold = run_stuart_landau(sc, K=K, duration=args.duration, mu=args.mu)
        elapsed = time.perf_counter() - t0
        print(f"  Simulation: {elapsed:.1f}s, BOLD shape: {bold.shape}")

        sim_fc = compute_fc(bold)
        result = compare_fc(sim_fc, emp_fc)
        result["K"] = K
        result["duration"] = args.duration
        result["mu"] = args.mu
        result["wall_time_s"] = round(elapsed, 2)
        results.append(result)

        print(f"  FC correlation: r={result['correlation']:.4f} (p={result['p_value']:.2e})")

    # Find best K
    best = max(results, key=lambda r: r["correlation"])
    print(f"\n{'='*60}")
    print(f"Best: K={best['K']:.2f}, r={best['correlation']:.4f}")
    print(f"Target: r=0.72 (Deco et al. 2018)")
    print(f"{'='*60}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
