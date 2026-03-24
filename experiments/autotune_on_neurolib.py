# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Auto-tune pipeline on neurolib ALN output
#
# Tests identify_binding_spec() on real neural dynamics:
# Can SPO recover coupling parameters from neurolib's firing rates?

"""Auto-tune experiment on neurolib output.

Usage:
    python experiments/autotune_on_neurolib.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from neurolib.models.aln import ALNModel
from neurolib.utils.loadData import Dataset
from scipy.stats import pearsonr

from scpn_phase_orchestrator.autotune.pipeline import identify_binding_spec


def main():
    print("Loading HCP + running neurolib ALN...")
    ds = Dataset("hcp")
    model = ALNModel(Cmat=ds.Cmat, Dmat=ds.Dmat)
    model.params["duration"] = 10000  # 10s (enough for coupling estimation)
    model.params["Ke_gl"] = 2.0

    t0 = time.perf_counter()
    model.run()
    sim_time = time.perf_counter() - t0
    print(f"  neurolib simulation: {sim_time:.1f}s")

    rates = model.rates_exc  # (80, T)
    print(f"  rates shape: {rates.shape}")

    # Use a subset of regions (auto-tune on 80 regions is expensive)
    n_subset = 20
    rates_subset = rates[:n_subset]
    sc_subset = ds.Cmat[:n_subset, :n_subset]
    print(f"  Using {n_subset} regions for auto-tune")

    # Compute sampling frequency from model dt
    dt_ms = model.params["dt"]
    native_fs = 1000.0 / dt_ms  # Hz
    print(f"  Native sampling: {native_fs:.1f} Hz")

    # Downsample to 1kHz (auto-tune doesn't need 10kHz neural dynamics)
    target_fs = 1000.0
    ds_factor = max(1, int(native_fs / target_fs))
    rates_subset = rates_subset[:, ::ds_factor]
    fs = native_fs / ds_factor
    print(f"  Downsampled to {fs:.1f} Hz ({rates_subset.shape[1]} samples)")

    # Run auto-tune
    print("\nRunning identify_binding_spec()...")
    t0 = time.perf_counter()
    result = identify_binding_spec(rates_subset, fs)
    autotune_time = time.perf_counter() - t0
    print(f"  Auto-tune completed in {autotune_time:.1f}s")

    # Compare recovered coupling with structural connectivity
    recovered_knm = result.knm
    triu = np.triu_indices(n_subset, k=1)

    # Normalize both for correlation
    sc_upper = sc_subset[triu]
    knm_upper = recovered_knm[triu]

    # SC→recovered coupling correlation
    if np.std(sc_upper) > 0 and np.std(knm_upper) > 0:
        r_sc, p_sc = pearsonr(sc_upper, knm_upper)
    else:
        r_sc, p_sc = 0.0, 1.0

    results = {
        "n_regions_total": int(rates.shape[0]),
        "n_regions_used": n_subset,
        "n_timepoints": int(rates.shape[1]),
        "fs_hz": round(fs, 1),
        "autotune_time_s": round(autotune_time, 2),
        "dominant_freqs_hz": [round(f, 2) for f in result.dominant_freqs],
        "omegas_rad_s": [round(o, 2) for o in result.omegas],
        "K_c_estimate": round(result.K_c_estimate, 4),
        "recovered_knm_mean": round(float(np.mean(recovered_knm)), 4),
        "recovered_knm_max": round(float(np.max(recovered_knm)), 4),
        "sc_vs_recovered_r": round(float(r_sc), 4),
        "sc_vs_recovered_p": float(p_sc),
        "interpretation": (
            "SC→recovered coupling correlation measures whether auto-tune "
            "recovers the structural connectivity from functional dynamics. "
            "r>0.3 would be meaningful, r>0.5 would be strong."
        ),
    }

    print(f"\n{'=' * 60}")
    print("Auto-Tune Results on neurolib ALN")
    print(f"{'=' * 60}")
    print(f"Regions: {n_subset} (of {rates.shape[0]})")
    print(f"K_c estimate: {result.K_c_estimate:.4f}")
    print(
        f"Recovered K_nm: mean={np.mean(recovered_knm):.4f}, "
        f"max={np.max(recovered_knm):.4f}"
    )
    print(f"SC vs recovered correlation: r={r_sc:.4f} (p={p_sc:.2e})")
    freq_range = (min(result.dominant_freqs), max(result.dominant_freqs))
    print(f"Dominant frequencies: {freq_range[0]:.1f} - {freq_range[1]:.1f} Hz")

    with Path("experiments/autotune_on_neurolib_results.json").open("w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to experiments/autotune_on_neurolib_results.json")


if __name__ == "__main__":
    main()
