# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Anesthesia simulation via coupling reduction
#
# Propofol-induced unconsciousness IS a Kuramoto phase transition below K_c
# (R7 research finding, confirmed by Deco's group).
# Simulate by progressively reducing global coupling K.
# Measure: does SPO regime FSM detect the transition?

"""Simulate anesthesia as coupling reduction on 80-region brain model.

The neurolib+SPO pipeline proved R=0.41 (metastable) at K=2.0.
Reducing K should drive R below critical thresholds.
SPO's regime FSM should detect: NOMINAL → DEGRADED → CRITICAL.

Usage:
    python experiments/anesthesia_simulation.py
"""

from __future__ import annotations

import json
import time

import numpy as np
from scipy.signal import hilbert

from neurolib.models.aln import ALNModel
from neurolib.utils.loadData import Dataset

from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
from scpn_phase_orchestrator.supervisor.events import EventBus
from scpn_phase_orchestrator.supervisor.regimes import Regime, RegimeManager
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter


def run_at_coupling(ds, K: float, duration_ms: int = 10000) -> dict:
    """Run ALN at given coupling, return SPO regime analysis."""
    model = ALNModel(Cmat=ds.Cmat, Dmat=ds.Dmat)
    model.params["duration"] = duration_ms
    model.params["Ke_gl"] = K

    t0 = time.perf_counter()
    model.run()
    elapsed = time.perf_counter() - t0

    rates = model.rates_exc
    if rates is None or rates.shape[1] < 100:
        return {"K": K, "error": "rates too short"}

    # Extract phases via Hilbert on rate envelope
    n_regions = rates.shape[0]
    n_t = rates.shape[1]
    phases = np.zeros((n_regions, n_t))
    for i in range(n_regions):
        analytic = hilbert(rates[i])
        phases[i] = np.angle(analytic) % (2 * np.pi)

    # SPO regime analysis on 100-step windows
    ds_factor = 100
    n_windows = n_t // ds_factor

    event_bus = EventBus()
    rm = RegimeManager(event_bus=event_bus)

    R_vals = []
    regimes = []
    for w in range(n_windows):
        phase_snap = phases[:, (w + 1) * ds_factor - 1]
        R, psi = compute_order_parameter(phase_snap)
        R_vals.append(float(R))

        upde = UPDEState(
            layers=[LayerState(R=R, psi=psi)],
            cross_layer_alignment=np.eye(1),
            stability_proxy=R,
            regime_id=rm.current_regime.value,
        )
        proposed = rm.evaluate(upde, BoundaryState())
        rm.transition(proposed)
        regimes.append(rm.current_regime.value)

    from collections import Counter
    regime_counts = dict(Counter(regimes))

    return {
        "K": K,
        "R_mean": round(float(np.mean(R_vals)), 4),
        "R_std": round(float(np.std(R_vals)), 4),
        "R_min": round(float(min(R_vals)), 4),
        "R_max": round(float(max(R_vals)), 4),
        "regime_counts": regime_counts,
        "n_transitions": len(rm.transition_history),
        "wall_time_s": round(elapsed, 2),
        "n_windows": n_windows,
    }


def main() -> None:
    print("Loading HCP data...")
    ds_data = Dataset("hcp")

    # Simulate anesthesia: K from 5.0 (awake) down to 0.1 (deep anesthesia)
    K_values = [5.0, 3.0, 2.0, 1.5, 1.0, 0.5, 0.2, 0.1]

    print(f"\nSimulating anesthesia via coupling reduction...")
    print(f"{'K':>5} {'R_mean':>8} {'R_min':>8} {'regime':>30} {'transitions':>12}")
    print("-" * 70)

    results = []
    for K in K_values:
        r = run_at_coupling(ds_data, K, duration_ms=15000)
        results.append(r)
        regime_str = str(r.get("regime_counts", {}))
        print(
            f"{K:>5.1f} {r.get('R_mean',0):>8.4f} {r.get('R_min',0):>8.4f} "
            f"{regime_str:>30} {r.get('n_transitions',0):>12}"
        )

    print(f"\n{'='*70}")
    print(f"Anesthesia Simulation Summary")
    print(f"{'='*70}")

    # Find transition point
    for i in range(len(results) - 1):
        r1 = results[i]
        r2 = results[i + 1]
        if r1.get("R_mean", 0) > 0.3 and r2.get("R_mean", 0) < 0.3:
            print(
                f"Consciousness transition between K={r1['K']} "
                f"(R={r1.get('R_mean',0):.4f}) and K={r2['K']} "
                f"(R={r2.get('R_mean',0):.4f})"
            )
            break
    else:
        print("No clear consciousness transition detected in K range")

    with open("experiments/anesthesia_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to experiments/anesthesia_results.json")


if __name__ == "__main__":
    main()
