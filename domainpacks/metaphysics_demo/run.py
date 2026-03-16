# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Metaphysics demo example

"""Metaphysics demo: P/I/S + Imprint + Geometry ablation.

Runs the binding spec twice — once with imprint enabled, once without —
and prints R_good / R_bad trajectories for comparison.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scpn_phase_orchestrator.binding import load_binding_spec, validate_binding_spec
from scpn_phase_orchestrator.coupling.geometry_constraints import (
    NonNegativeConstraint,
    SymmetryConstraint,
    project_knm,
)
from scpn_phase_orchestrator.coupling.knm import CouplingBuilder, CouplingState
from scpn_phase_orchestrator.imprint.state import ImprintState
from scpn_phase_orchestrator.imprint.update import ImprintModel
from scpn_phase_orchestrator.monitor.boundaries import BoundaryObserver
from scpn_phase_orchestrator.supervisor.policy import SupervisorPolicy
from scpn_phase_orchestrator.supervisor.policy_rules import (
    PolicyEngine,
    load_policy_rules,
)
from scpn_phase_orchestrator.supervisor.regimes import RegimeManager
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState
from scpn_phase_orchestrator.upde.order_params import (
    compute_order_parameter,
    compute_plv,
)

STEPS = 300
SPEC_PATH = Path(__file__).parent / "binding_spec.yaml"


def _run_sim(use_imprint: bool) -> dict:
    spec = load_binding_spec(SPEC_PATH)
    errors = validate_binding_spec(spec)
    if errors:
        raise RuntimeError(f"Invalid spec: {errors}")

    n_osc = sum(len(layer.oscillator_ids) for layer in spec.layers)
    builder = CouplingBuilder()
    coupling = builder.build(
        n_osc,
        spec.coupling.base_strength,
        spec.coupling.decay_alpha,
    )
    engine = UPDEEngine(n_osc, dt=spec.sample_period_s)
    boundary_observer = BoundaryObserver(spec.boundaries)
    regime_manager = RegimeManager()
    supervisor = SupervisorPolicy(regime_manager)

    policy_path = SPEC_PATH.parent / "policy.yaml"
    policy_engine = None
    if policy_path.exists():
        rules = load_policy_rules(policy_path)
        if rules:
            policy_engine = PolicyEngine(rules)

    imprint_model = None
    imprint_state = None
    if use_imprint and spec.imprint_model is not None:
        imprint_model = ImprintModel(
            spec.imprint_model.decay_rate, spec.imprint_model.saturation
        )
        imprint_state = ImprintState(m_k=np.zeros(n_osc), last_update=0.0)

    geo_constraints = []
    if spec.geometry_prior is not None:
        ct = spec.geometry_prior.constraint_type.lower()
        if "symmetric" in ct:
            geo_constraints.append(SymmetryConstraint())
        if "non_negative" in ct or "nonneg" in ct:
            geo_constraints.append(NonNegativeConstraint())

    rng = np.random.default_rng(42)
    phases = rng.uniform(0, 2 * np.pi, n_osc)
    omegas = np.array(
        [1.0 + 0.1 * layer.index for layer in spec.layers for _ in layer.oscillator_ids]
    )

    layer_osc_ranges = {}
    osc_idx = 0
    for layer in spec.layers:
        n_layer = len(layer.oscillator_ids)
        layer_osc_ranges[layer.index] = list(range(osc_idx, osc_idx + n_layer))
        osc_idx += n_layer

    zeta = max(
        spec.drivers.physical.get("zeta", 0.0),
        spec.drivers.informational.get("zeta", 0.0),
        spec.drivers.symbolic.get("zeta", 0.0),
    )
    psi_target = spec.drivers.physical.get("psi", 0.0)

    r_good_trace = []
    r_bad_trace = []

    for _ in range(STEPS):
        eff_knm = coupling.knm
        eff_alpha = coupling.alpha
        if imprint_model is not None and imprint_state is not None:
            eff_knm = imprint_model.modulate_coupling(eff_knm, imprint_state)
            eff_alpha = imprint_model.modulate_lag(eff_alpha, imprint_state)
        if geo_constraints:
            eff_knm = project_knm(eff_knm, geo_constraints)

        phases = engine.step(phases, omegas, eff_knm, zeta, psi_target, eff_alpha)

        layer_states = []
        for layer in spec.layers:
            osc_ids = layer_osc_ranges[layer.index]
            if osc_ids:
                r, psi = compute_order_parameter(phases[osc_ids])
            else:
                r, psi = 0.0, 0.0
            layer_states.append(LayerState(R=r, psi=psi))

        n_layers = len(spec.layers)
        cla = np.zeros((n_layers, n_layers))
        for li in range(n_layers):
            for lj in range(li + 1, n_layers):
                ids_i = layer_osc_ranges[spec.layers[li].index]
                ids_j = layer_osc_ranges[spec.layers[lj].index]
                if ids_i and ids_j:
                    pi, pj = phases[ids_i], phases[ids_j]
                    mn = min(len(pi), len(pj))
                    plv = compute_plv(pi[:mn], pj[:mn])
                    cla[li, lj] = plv
                    cla[lj, li] = plv

        mean_r = float(np.mean([ls.R for ls in layer_states]))
        upde_state = UPDEState(
            layers=layer_states,
            cross_layer_alignment=cla,
            stability_proxy=mean_r,
            regime_id=regime_manager.current_regime.value,
        )

        obs_values = {"R": mean_r}
        for i, ls in enumerate(layer_states):
            obs_values[f"R_{i}"] = ls.R
        boundary_state = boundary_observer.observe(obs_values)
        actions = supervisor.decide(upde_state, boundary_state)

        if policy_engine is not None:
            actions.extend(
                policy_engine.evaluate(
                    regime_manager.current_regime,
                    upde_state,
                    spec.objectives.good_layers,
                    spec.objectives.bad_layers,
                )
            )

        for act in actions:
            if act.knob == "zeta":
                zeta = min(zeta + act.value, 0.5)
            elif act.knob == "K" and act.scope == "global":
                coupling = CouplingState(
                    knm=coupling.knm * (1.0 + act.value),
                    alpha=coupling.alpha,
                    active_template=coupling.active_template,
                )
            elif act.knob == "Psi":
                psi_target = act.value

        if imprint_model is not None and imprint_state is not None:
            exposure = np.array(
                [
                    layer_states[i].R
                    for i, layer in enumerate(spec.layers)
                    for _ in layer.oscillator_ids
                ]
            )
            imprint_state = imprint_model.update(
                imprint_state, exposure, spec.sample_period_s
            )

        good_ph = [
            phases[j]
            for idx in spec.objectives.good_layers
            for j in layer_osc_ranges.get(idx, [])
        ]
        bad_ph = [
            phases[j]
            for idx in spec.objectives.bad_layers
            for j in layer_osc_ranges.get(idx, [])
        ]
        r_good = compute_order_parameter(np.array(good_ph))[0] if good_ph else 0.0
        r_bad = compute_order_parameter(np.array(bad_ph))[0] if bad_ph else 0.0
        r_good_trace.append(r_good)
        r_bad_trace.append(r_bad)

    return {
        "r_good_final": r_good_trace[-1],
        "r_bad_final": r_bad_trace[-1],
        "r_good_trace": r_good_trace,
        "r_bad_trace": r_bad_trace,
        "regime": regime_manager.current_regime.value,
    }


def main():
    print("=== Metaphysics Demo: Imprint + Geometry Ablation ===\n")

    print(f"Running {STEPS} steps WITH imprint + geometry...")
    with_imprint = _run_sim(use_imprint=True)
    print(
        f"  R_good={with_imprint['r_good_final']:.4f}  "
        f"R_bad={with_imprint['r_bad_final']:.4f}  "
        f"regime={with_imprint['regime']}"
    )

    print(f"\nRunning {STEPS} steps WITHOUT imprint (geometry only)...")
    without_imprint = _run_sim(use_imprint=False)
    print(
        f"  R_good={without_imprint['r_good_final']:.4f}  "
        f"R_bad={without_imprint['r_bad_final']:.4f}  "
        f"regime={without_imprint['regime']}"
    )

    delta_good = with_imprint["r_good_final"] - without_imprint["r_good_final"]
    delta_bad = with_imprint["r_bad_final"] - without_imprint["r_bad_final"]
    print(f"\nAblation delta: dR_good={delta_good:+.4f}  dR_bad={delta_bad:+.4f}")

    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        steps = range(STEPS)

        ax1.plot(steps, with_imprint["r_good_trace"], label="imprint ON")
        ax1.plot(
            steps,
            without_imprint["r_good_trace"],
            label="imprint OFF",
            linestyle="--",
        )
        ax1.set_title("R_good (informational + symbolic)")
        ax1.set_xlabel("step")
        ax1.set_ylabel("R")
        ax1.legend()

        ax2.plot(steps, with_imprint["r_bad_trace"], label="imprint ON")
        ax2.plot(
            steps,
            without_imprint["r_bad_trace"],
            label="imprint OFF",
            linestyle="--",
        )
        ax2.set_title("R_bad (physical)")
        ax2.set_xlabel("step")
        ax2.set_ylabel("R")
        ax2.legend()

        fig.tight_layout()
        out = Path(__file__).parent / "ablation.png"
        fig.savefig(out, dpi=120)
        print(f"\nPlot saved to {out}")
        plt.close(fig)
    except ImportError:
        print("\n(matplotlib not installed — skipping plot)")


if __name__ == "__main__":
    main()
