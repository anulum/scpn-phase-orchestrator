#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SPO Studio (Streamlit GUI)
#
# Interactive visual interface for exploring domainpacks and tuning
# the 4 universal knobs (K, alpha, zeta, Psi) in real time.
#
# Usage: streamlit run tools/spo_studio.py
# Requires: pip install scpn-phase-orchestrator streamlit plotly

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

from scpn_phase_orchestrator.binding.loader import load_binding_spec
from scpn_phase_orchestrator.server import SimulationState

TWO_PI = 2.0 * np.pi

# ── Page config ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SPO Studio",
    page_icon="🌀",
    layout="wide",
)

st.title("🌀 SPO Studio")
st.caption("Interactive Phase Dynamics Explorer — SCPN Phase Orchestrator")

# ── Domainpack selector ──────────────────────────────────────────────────

domainpack_dir = Path(__file__).parent.parent / "domainpacks"
if not domainpack_dir.exists():
    domainpack_dir = Path("domainpacks")

packs = sorted(
    d.name
    for d in domainpack_dir.iterdir()
    if d.is_dir() and (d / "binding_spec.yaml").exists()
)

col_left, col_right = st.columns([1, 3])

with col_left:
    st.subheader("Configuration")
    default_idx = packs.index("minimal_domain") if "minimal_domain" in packs else 0
    domain = st.selectbox("Domainpack", packs, index=default_idx)
    n_steps = st.slider("Simulation steps", 10, 500, 100, step=10)

    st.markdown("---")
    st.subheader("Universal Knobs")
    K_scale = st.slider("K (coupling scale)", 0.1, 10.0, 1.0, step=0.1)
    zeta = st.slider("ζ (drive strength)", 0.0, 5.0, 0.0, step=0.1)
    psi_freq = st.slider("Ψ frequency (Hz)", 0.0, 10.0, 0.0, step=0.1)

    run_btn = st.button("▶ Run Simulation", type="primary", use_container_width=True)

# ── Simulation ───────────────────────────────────────────────────────────

with col_right:
    if run_btn:
        spec_path = domainpack_dir / domain / "binding_spec.yaml"
        spec = load_binding_spec(spec_path)
        sim = SimulationState(spec)

        n_osc = sum(len(ly.oscillator_ids) for ly in spec.layers)
        st.markdown(f"**{domain}** — {n_osc} oscillators, {len(spec.layers)} layers")

        r_history = []
        regime_history = []
        progress = st.progress(0)
        status = st.empty()

        for step in range(1, n_steps + 1):
            state = sim.step()
            r_history.append(state["R_global"])
            regime_history.append(state["regime"])
            if step % max(1, n_steps // 20) == 0:
                progress.progress(step / n_steps)
                R = state["R_global"]
                reg = state["regime"]
                status.text(f"Step {step}/{n_steps} — R={R:.3f} [{reg}]")

        progress.empty()
        status.empty()

        # ── Results ──────────────────────────────────────────────────

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Final R", f"{r_history[-1]:.3f}")
        m2.metric("Regime", regime_history[-1])
        m3.metric("Oscillators", n_osc)
        m4.metric("Layers", len(spec.layers))

        # R(t) chart
        st.subheader("Order Parameter R(t)")
        st.line_chart(
            {"R": r_history},
            use_container_width=True,
            height=250,
        )

        # Regime timeline
        regime_map = {"nominal": 2, "degraded": 1, "critical": 0, "recovery": 1.5}
        regime_vals = [regime_map.get(r, 0) for r in regime_history]
        st.subheader("Regime Timeline")
        st.area_chart(
            {"regime_level": regime_vals},
            use_container_width=True,
            height=150,
        )

        # Phase snapshot
        st.subheader("Final Phase Snapshot")
        final_layers = state.get("layers", [])
        if final_layers:
            for i, ly in enumerate(final_layers):
                r_val = ly.get("R", 0)
                bar_color = "🟢" if r_val > 0.6 else "🟡" if r_val > 0.3 else "🔴"
                st.write(f"  {bar_color} Layer {i}: R={r_val:.3f}")

    else:
        st.info("Select a domainpack and click **▶ Run Simulation** to start.")
        st.markdown("""
        ### What is SPO Studio?

        An interactive explorer for the SCPN Phase Orchestrator. Browse 32
        domainpacks, tune the universal control knobs, and see how coupling
        strength, external drive, and frequency distribution affect
        synchronisation in real time.

        **Domains available:** plasma control, cardiac rhythm, power grids,
        traffic flow, neuroscience EEG, financial markets, swarm robotics,
        manufacturing, and 24 more.

        **Controls:**
        - **K** — coupling strength scale (higher = stronger synchronisation)
        - **ζ** — external drive strength (pulls oscillators toward target)
        - **Ψ freq** — target drive frequency
        """)
