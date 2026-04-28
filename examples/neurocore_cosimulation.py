#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SCPN Phase Orchestrator — Example: sc-neurocore Co-Simulation
#
# REAL integration: sc-neurocore synapse objects run alongside SPO.
# STDP weight changes drive coupling adaptation. Gap junctions provide
# direct phase coupling. Astrocyte Ca²⁺ modulates imprint memory.
#
# This is NOT a model or mock — it instantiates real sc-neurocore
# synapse objects and feeds their state into SPO every tick.
#
# Usage: python examples/neurocore_cosimulation.py
# Requires: pip install scpn-phase-orchestrator sc-neurocore>=3.13.0

from __future__ import annotations

import numpy as np
from sc_neurocore.synapses.gap_junction import GapJunction
from sc_neurocore.synapses.tripartite import TripartiteSynapse
from sc_neurocore.synapses.triplet_stdp import TripletSTDP

from scpn_phase_orchestrator.adapters.synapse_coupling_bridge import (
    SynapseCouplingBridge,
)
from scpn_phase_orchestrator.imprint.state import ImprintState
from scpn_phase_orchestrator.imprint.update import ImprintModel
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

TWO_PI = 2.0 * np.pi


def main() -> None:
    n = 4
    rng = np.random.default_rng(42)
    omegas = rng.uniform(-0.5, 0.5, n)
    knm_base = np.ones((n, n)) * 0.5
    np.fill_diagonal(knm_base, 0.0)
    alpha = np.zeros((n, n))

    # ── Instantiate real sc-neurocore synapses ───────────────────────
    # One STDP synapse per pair (i,j)
    stdp_synapses: dict[tuple[int, int], TripletSTDP] = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                stdp_synapses[(i, j)] = TripletSTDP(
                    tau_plus=16.8,
                    tau_minus=33.7,
                    tau_x=101.0,
                    tau_y=125.0,
                )

    # Gap junctions between neighbours
    gap_junctions: dict[tuple[int, int], GapJunction] = {}
    for i in range(n - 1):
        gap_junctions[(i, i + 1)] = GapJunction(conductance=0.2)

    # One tripartite synapse per oscillator (astrocyte modulation)
    astrocytes = [TripartiteSynapse() for _ in range(n)]

    # ── SPO setup ────────────────────────────────────────────────────
    eng = UPDEEngine(n, dt=0.01)
    bridge = SynapseCouplingBridge(n, stdp_scale=5.0, gap_scale=2.0, ca_scale=1.0)
    ImprintModel(decay_rate=0.01, saturation=3.0)
    imprint_state = ImprintState(m_k=np.zeros(n), last_update=0.0)
    phases = rng.uniform(0, TWO_PI, n)

    print("sc-neurocore ↔ SPO Co-Simulation")
    print("=" * 55)
    print(f"  {n} oscillators, {len(stdp_synapses)} STDP synapses,")
    print(f"  {len(gap_junctions)} gap junctions, {len(astrocytes)} astrocytes")
    print()
    print(f"{'Tick':>5s}  {'R':>6s}  {'dW_mean':>8s}  {'Ca_mean':>8s}  {'K_eff':>6s}")
    print("-" * 40)

    for tick in range(50):
        # ── Step 1: Generate spikes from phase crossings ─────────
        # Phase crossing 0 → spike (simplified spike model)
        prev_phases = phases.copy()
        phases = eng.step(phases, omegas, knm_base, 0.0, 0.0, alpha)
        spikes = (prev_phases > 5.5) & (phases < 1.0)  # wrapped past 2π

        # ── Step 2: Feed spikes to sc-neurocore synapses ─────────
        weight_matrix = np.zeros((n, n))
        for (i, j), syn in stdp_synapses.items():
            syn.step(
                pre_spike=bool(spikes[i]),
                post_spike=bool(spikes[j]),
                dt=0.01,
            )
            weight_matrix[i, j] = syn.weight

        # ── Step 3: Update gap junction conductances ─────────────
        conductance_matrix = np.zeros((n, n))
        for (i, j), gj in gap_junctions.items():
            # Use phase difference as proxy for voltage difference
            v_diff = np.sin(phases[i] - phases[j])
            conductance_matrix[i, j] = abs(gj.current(v_pre=v_diff, v_post=0.0))
            conductance_matrix[j, i] = conductance_matrix[i, j]

        # ── Step 4: Update astrocyte Ca²⁺ ────────────────────────
        ca_levels = np.zeros(n)
        for k, astro in enumerate(astrocytes):
            astro.step(
                pre_spike=bool(spikes[k]),
                post_spike=bool(spikes[(k + 1) % n]),
                dt=0.01,
            )
            ca_levels[k] = astro.ca

        # ── Step 5: Feed to bridge → get SPO coupling ────────────
        bridge.update_stdp_weights(weight_matrix)
        bridge.update_gap_conductances(conductance_matrix)
        bridge.update_astrocyte_ca(ca_levels)

        knm_adapted = bridge.apply_to_knm(knm_base)
        imprint_state = ImprintState(
            m_k=bridge.apply_to_imprint(imprint_state.m_k),
            last_update=imprint_state.last_update + 0.01,
        )

        # ── Step 6: Run SPO with adapted coupling ────────────────
        phases = eng.step(phases, omegas, knm_adapted, 0.0, 0.0, alpha)

        R, _ = compute_order_parameter(phases)
        snap = bridge.snapshot()

        if tick % 10 == 0 or tick == 49:
            print(
                f"{tick + 1:>5d}  {R:>6.3f}  "
                f"{snap.mean_weight_change:>8.5f}  "
                f"{snap.mean_ca:>8.4f}  "
                f"{knm_adapted[knm_adapted > 0].mean():>6.3f}"
            )

    print("\nFinal STDP weights (sample):")
    for (i, j), syn in list(stdp_synapses.items())[:4]:
        print(f"  {i}→{j}: w={syn.weight:.4f}")

    print("\nAstrocyte Ca²⁺ levels:")
    for k, astro in enumerate(astrocytes):
        print(f"  Oscillator {k}: Ca²⁺={astro.ca:.4f}")

    print("\nThis is real co-simulation: sc-neurocore synapses drive SPO coupling.")


if __name__ == "__main__":
    main()
