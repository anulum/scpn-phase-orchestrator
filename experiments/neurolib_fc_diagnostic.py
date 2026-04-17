# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — neurolib FC diagnostic
#
# Previous FC sweep gave r=-0.40 to -0.45 at ALL K values.
# This is suspicious: uniform negative correlation regardless of K
# suggests a systematic issue, not wrong parameters.
#
# Diagnostics:
# 1. Check if neurolib's OWN FC analysis gives different results
# 2. Try neurolib's built-in BOLD signal (model.BOLD)
# 3. Try rates-based FC (no BOLD, just firing rate correlation)
# 4. Check if the empirical FC matrix is loaded correctly
# 5. 2D sweep: Ke_gl x sigma_ou

"""Diagnose why FC correlation is negative for all K values.

Usage:
    python experiments/neurolib_fc_diagnostic.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from neurolib.models.aln import ALNModel
from neurolib.models.bold import BOLDModel
from neurolib.utils.functions import fc
from neurolib.utils.loadData import Dataset
from scipy.stats import pearsonr


def fc_correlation(sim_fc, emp_fc):
    n = sim_fc.shape[0]
    triu = np.triu_indices(n, k=1)
    r, p = pearsonr(sim_fc[triu], emp_fc[triu])
    return float(r), float(p)


def main():
    print("Loading HCP data...")
    ds = Dataset("hcp")
    emp_fc = np.mean(ds.FCs, axis=0)
    np.fill_diagonal(emp_fc, 0.0)

    # Diagnostic 1: Check empirical FC properties
    triu = np.triu_indices(80, k=1)
    emp_vals = emp_fc[triu]
    print("\nEmpirical FC stats:")
    print(f"  range: [{emp_vals.min():.3f}, {emp_vals.max():.3f}]")
    print(f"  mean: {emp_vals.mean():.3f}, std: {emp_vals.std():.3f}")
    print(f"  fraction > 0: {(emp_vals > 0).mean():.3f}")

    K = 2.0
    duration = 60000  # 60s for speed

    print(f"\nRunning ALN: K={K}, duration={duration}ms...")
    model = ALNModel(Cmat=ds.Cmat, Dmat=ds.Dmat)
    model.params["duration"] = duration
    model.params["Ke_gl"] = K

    t0 = time.perf_counter()
    model.run()
    elapsed = time.perf_counter() - t0
    print(f"  Simulation: {elapsed:.1f}s")

    rates = model.rates_exc
    print(f"  rates shape: {rates.shape}")
    print(f"  rates range: [{rates.min():.2f}, {rates.max():.2f}] Hz")

    # Method A: Raw rates FC (no BOLD)
    rates_fc = np.corrcoef(rates)
    np.fill_diagonal(rates_fc, 0.0)
    r_rates, _ = fc_correlation(rates_fc, emp_fc)
    print(f"\n[A] Rates FC (no BOLD): r={r_rates:.4f}")

    # Method B: neurolib's own fc() function on rates
    try:
        neurolib_fc_result = fc(rates)
        np.fill_diagonal(neurolib_fc_result, 0.0)
        r_neurolib, _ = fc_correlation(neurolib_fc_result, emp_fc)
        print(f"[B] neurolib fc() on rates: r={r_neurolib:.4f}")
    except Exception as e:
        print(f"[B] neurolib fc() failed: {e}")
        r_neurolib = None

    # Method C: Balloon-Windkessel BOLD
    bold_model = BOLDModel(N=80, dt=model.params["dt"])
    bold_model.run(rates)
    bold = bold_model.BOLD
    if bold.shape[1] > 10:
        bold_fc = np.corrcoef(bold)
        np.fill_diagonal(bold_fc, 0.0)
        r_bold, _ = fc_correlation(bold_fc, emp_fc)
        print(f"[C] BOLD FC: r={r_bold:.4f} ({bold.shape[1]} timepoints)")
    else:
        r_bold = None
        print(f"[C] BOLD FC: too few timepoints ({bold.shape[1]})")

    # Method D: Downsampled rates (1Hz, like original experiment)
    ds_factor = max(1, int(1000 / model.params["dt"]))
    n_t = rates.shape[1]
    n_samples = n_t // ds_factor
    rates_ds = (
        rates[:, : n_samples * ds_factor].reshape(80, n_samples, ds_factor).mean(axis=2)
    )
    rates_ds_fc = np.corrcoef(rates_ds)
    np.fill_diagonal(rates_ds_fc, 0.0)
    r_ds, _ = fc_correlation(rates_ds_fc, emp_fc)
    print(f"[D] Downsampled rates FC: r={r_ds:.4f} ({n_samples} samples)")

    # Method E: Check SC-FC correlation (structure-function)
    sc = ds.Cmat.copy()
    np.fill_diagonal(sc, 0.0)
    r_sc_fc, _ = fc_correlation(sc, emp_fc)
    print(f"\n[E] SC-FC correlation: r={r_sc_fc:.4f}")

    # Method F: Quick 2D sweep
    print("\n2D sweep: Ke_gl x sigma_ou (rates FC, 10s each)")
    sweep_results = []
    for K_val in [0.5, 1.0, 2.0, 4.0]:
        for sigma in [0.0, 0.05, 0.1, 0.5]:
            m = ALNModel(Cmat=ds.Cmat, Dmat=ds.Dmat)
            m.params["duration"] = 10000
            m.params["Ke_gl"] = K_val
            m.params["sigma_ou"] = sigma
            try:
                m.run()
                r_exc = m.rates_exc
                if r_exc is not None and r_exc.shape[1] > 100:
                    sim_fc = np.corrcoef(r_exc)
                    np.fill_diagonal(sim_fc, 0.0)
                    r_val, _ = fc_correlation(sim_fc, emp_fc)
                else:
                    r_val = float("nan")
            except Exception:
                r_val = float("nan")
            sweep_results.append({"K": K_val, "sigma": sigma, "r": round(r_val, 4)})
            print(f"  K={K_val}, sigma={sigma}: r={r_val:.4f}")

    results = {
        "emp_fc_mean": round(float(emp_vals.mean()), 4),
        "emp_fc_std": round(float(emp_vals.std()), 4),
        "r_rates_fc": round(r_rates, 4),
        "r_neurolib_fc": (round(r_neurolib, 4) if r_neurolib is not None else None),
        "r_bold_fc": round(r_bold, 4) if r_bold is not None else None,
        "r_downsampled_fc": round(r_ds, 4),
        "r_sc_fc": round(r_sc_fc, 4),
        "sweep_2d": sweep_results,
    }

    out = Path("experiments/neurolib_fc_diagnostic_results.json")
    with out.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
