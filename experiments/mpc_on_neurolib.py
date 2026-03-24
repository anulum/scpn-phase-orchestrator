# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — MPC predictive supervisor on neurolib output
#
# Tests PredictiveSupervisor on real neurolib ALN dynamics.
# Measures: how many degradation events does MPC predict before they happen?

"""MPC supervision experiment on neurolib ALN output.

Usage:
    python experiments/mpc_on_neurolib.py
"""

from __future__ import annotations

import json
import time

import numpy as np
from scipy.signal import hilbert

from neurolib.models.aln import ALNModel
from neurolib.utils.loadData import Dataset

from scpn_phase_orchestrator.coupling.knm import CouplingBuilder
from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
from scpn_phase_orchestrator.supervisor.predictive import PredictiveSupervisor
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter


def extract_phases(rates):
    n_regions, n_t = rates.shape
    phases = np.zeros((n_regions, n_t))
    for i in range(n_regions):
        analytic = hilbert(rates[i])
        phases[i] = np.angle(analytic) % (2 * np.pi)
    return phases


def main():
    print("Loading HCP + running neurolib ALN...")
    ds = Dataset("hcp")
    model = ALNModel(Cmat=ds.Cmat, Dmat=ds.Dmat)
    model.params["duration"] = 30000
    model.params["Ke_gl"] = 2.0

    t0 = time.perf_counter()
    model.run()
    sim_time = time.perf_counter() - t0
    print(f"  neurolib simulation: {sim_time:.1f}s")

    rates = model.rates_exc
    phases = extract_phases(rates)
    n_regions = phases.shape[0]

    # Build coupling matrix for MPC (simple exponential for 80 regions)
    coupling = CouplingBuilder().build(n_regions, 0.47, 0.25)
    omegas = np.ones(n_regions) * 2.0 * np.pi * 10.0  # ~10 Hz alpha

    # MPC supervisor
    mpc = PredictiveSupervisor(
        n_oscillators=n_regions, dt=0.01, horizon=20, divergence_threshold=0.5
    )

    ds_factor = 100  # 10ms windows
    n_windows = phases.shape[1] // ds_factor

    predictions = []
    R_history = []
    actions_taken = []
    prediction_hits = 0
    prediction_misses = 0
    false_alarms = 0

    print(f"\nRunning MPC supervision on {n_windows} windows...")
    for w in range(n_windows):
        end = (w + 1) * ds_factor
        window_phases = phases[:, end - 1]

        R, psi = compute_order_parameter(window_phases)
        R_history.append(float(R))

        # MPC prediction
        pred = mpc.predict(window_phases, omegas, coupling.knm, coupling.alpha)
        predictions.append({
            "window": w,
            "R_current": round(float(R), 4),
            "R_predicted_final": round(pred.R_predicted[-1], 4),
            "will_degrade": pred.will_degrade,
            "will_critical": pred.will_critical,
            "steps_to_degradation": pred.steps_to_degradation,
        })

        # MPC control decision
        upde_state = UPDEState(
            layers=[LayerState(R=R, psi=psi)],
            cross_layer_alignment=np.eye(1),
            stability_proxy=R,
            regime_id=0,
        )
        actions = mpc.decide(
            window_phases, omegas, coupling.knm, coupling.alpha,
            upde_state, BoundaryState(),
        )
        if actions:
            actions_taken.append({
                "window": w,
                "R_current": round(float(R), 4),
                "action": actions[0].justification,
            })

    # Evaluate prediction accuracy: did "will_degrade" at t predict R < 0.6 at t+horizon?
    R_arr = np.array(R_history)
    for i, p in enumerate(predictions):
        if p["will_degrade"]:
            future_end = min(i + 20, len(R_arr))
            future_R = R_arr[i:future_end]
            if len(future_R) > 0 and np.any(future_R < 0.6):
                prediction_hits += 1
            else:
                false_alarms += 1

    # Count actual degradation events that were NOT predicted
    for i in range(len(R_arr) - 20):
        actual_degrade = np.any(R_arr[i:i + 20] < 0.6)
        predicted = predictions[i]["will_degrade"]
        if actual_degrade and not predicted:
            prediction_misses += 1

    total_predictions = sum(1 for p in predictions if p["will_degrade"])
    total_actual = sum(1 for i in range(len(R_arr) - 20)
                       if np.any(R_arr[i:i + 20] < 0.6))

    results = {
        "n_windows": n_windows,
        "R_mean": round(float(np.mean(R_history)), 4),
        "R_std": round(float(np.std(R_history)), 4),
        "R_min": round(float(np.min(R_history)), 4),
        "R_max": round(float(np.max(R_history)), 4),
        "total_degradation_predictions": total_predictions,
        "total_actual_degradations": total_actual,
        "prediction_hits": prediction_hits,
        "false_alarms": false_alarms,
        "prediction_misses": prediction_misses,
        "precision": round(prediction_hits / max(1, total_predictions), 4),
        "recall": round(prediction_hits / max(1, total_actual), 4),
        "n_control_actions": len(actions_taken),
        "sim_duration_ms": 30000,
        "K": 2.0,
        "mpc_horizon": 20,
    }

    print(f"\n{'='*60}")
    print(f"MPC Predictive Supervision Results")
    print(f"{'='*60}")
    print(f"Windows: {n_windows}")
    print(f"R: {results['R_mean']:.4f} ± {results['R_std']:.4f}")
    print(f"Degradation predictions: {total_predictions}")
    print(f"Actual degradation windows: {total_actual}")
    print(f"Hits: {prediction_hits}, False alarms: {false_alarms}, Misses: {prediction_misses}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"Control actions triggered: {len(actions_taken)}")

    with open("experiments/mpc_on_neurolib_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to experiments/mpc_on_neurolib_results.json")


if __name__ == "__main__":
    main()
