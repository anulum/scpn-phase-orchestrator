# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SPO supervision on neurolib output
#
# Demonstrates SPO's unique value: regime detection, TCBO consciousness
# gate, and predictive supervision on top of neurolib's neural dynamics.

"""Feed neurolib ALN output into SPO's supervision layer.

Shows what neurolib CAN'T do: regime classification, boundary monitoring,
TCBO consciousness boundary, predictive control.

Usage:
    python experiments/regime_on_neurolib.py
"""

from __future__ import annotations

import json
import time

import numpy as np
from scipy.signal import hilbert

from neurolib.models.aln import ALNModel
from neurolib.utils.loadData import Dataset

from scpn_phase_orchestrator.monitor.boundaries import BoundaryObserver, BoundaryState
from scpn_phase_orchestrator.monitor.npe import compute_npe
from scpn_phase_orchestrator.ssgf.tcbo import TCBOObserver
from scpn_phase_orchestrator.supervisor.events import EventBus
from scpn_phase_orchestrator.supervisor.regimes import Regime, RegimeManager
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter


def extract_phases_from_rates(rates: np.ndarray, window: int = 100) -> np.ndarray:
    """Extract instantaneous phases from neural firing rates via Hilbert."""
    n_regions, n_t = rates.shape
    phases = np.zeros((n_regions, n_t))
    for i in range(n_regions):
        analytic = hilbert(rates[i])
        phases[i] = np.angle(analytic) % (2 * np.pi)
    return phases


def main() -> None:
    print("Loading HCP + running neurolib ALN...")
    ds = Dataset("hcp")
    model = ALNModel(Cmat=ds.Cmat, Dmat=ds.Dmat)
    model.params["duration"] = 30000  # 30s
    model.params["Ke_gl"] = 2.0  # optimal from baseline

    t0 = time.perf_counter()
    model.run()
    sim_time = time.perf_counter() - t0
    print(f"  neurolib simulation: {sim_time:.1f}s")

    rates = model.rates_exc  # (80, T)
    print(f"  rates shape: {rates.shape}")

    # Extract phases
    phases = extract_phases_from_rates(rates)
    n_regions, n_t = phases.shape

    # Downsample for SPO processing (every 100 steps = 10ms windows)
    ds_factor = 100
    n_windows = n_t // ds_factor

    # SPO supervision components
    event_bus = EventBus()
    regime_manager = RegimeManager(event_bus=event_bus)
    tcbo = TCBOObserver(tau_h1=0.72, window_size=30, embed_dim=2, embed_delay=1)

    regime_history = []
    R_history = []
    npe_history = []
    tcbo_history = []

    print(f"\nRunning SPO supervision on {n_windows} windows...")
    for w in range(n_windows):
        start = w * ds_factor
        end = start + ds_factor
        window_phases = phases[:, end - 1]  # phase snapshot at window end

        # Kuramoto R across all regions
        R, psi = compute_order_parameter(window_phases)
        R_history.append(float(R))

        # NPE
        npe = compute_npe(window_phases)
        npe_history.append(float(npe))

        # TCBO
        tcbo_state = tcbo.observe(window_phases)
        tcbo_history.append(tcbo_state.p_h1)

        # Regime classification
        layer_states = [LayerState(R=R, psi=psi)]
        upde_state = UPDEState(
            layers=layer_states,
            cross_layer_alignment=np.eye(1),
            stability_proxy=R,
            regime_id=regime_manager.current_regime.value,
        )
        proposed = regime_manager.evaluate(upde_state, BoundaryState())
        regime_manager.transition(proposed)
        regime_history.append(regime_manager.current_regime.value)

    # Report
    regime_counts = {}
    for r in regime_history:
        regime_counts[r] = regime_counts.get(r, 0) + 1

    transitions = len(regime_manager.transition_history)

    print(f"\n{'='*60}")
    print(f"SPO Supervision Results on neurolib ALN output")
    print(f"{'='*60}")
    print(f"Windows analyzed: {n_windows}")
    print(f"R range: [{min(R_history):.4f}, {max(R_history):.4f}]")
    print(f"R mean: {np.mean(R_history):.4f}")
    print(f"NPE range: [{min(npe_history):.4f}, {max(npe_history):.4f}]")
    print(f"TCBO p_h1 mean: {np.mean(tcbo_history):.4f}")
    print(f"Regime distribution: {regime_counts}")
    print(f"Regime transitions: {transitions}")
    print(f"Events logged: {event_bus.count}")

    results = {
        "n_windows": n_windows,
        "R_mean": round(float(np.mean(R_history)), 4),
        "R_std": round(float(np.std(R_history)), 4),
        "R_min": round(float(min(R_history)), 4),
        "R_max": round(float(max(R_history)), 4),
        "NPE_mean": round(float(np.mean(npe_history)), 4),
        "TCBO_p_h1_mean": round(float(np.mean(tcbo_history)), 4),
        "regime_counts": regime_counts,
        "n_transitions": transitions,
        "n_events": event_bus.count,
        "sim_duration_ms": 30000,
        "K": 2.0,
    }

    with open("experiments/regime_on_neurolib_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to experiments/regime_on_neurolib_results.json")


if __name__ == "__main__":
    main()
