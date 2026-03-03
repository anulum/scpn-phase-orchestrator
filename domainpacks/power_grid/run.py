# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

"""Power grid: load step -> renewable ramp -> generator trip -> AGC restore."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scpn_phase_orchestrator.binding import load_binding_spec, validate_binding_spec
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

STEPS = 250
SPEC_PATH = Path(__file__).parent / "binding_spec.yaml"
TWO_PI = 2.0 * np.pi

# Normalised angular velocities for each oscillator.
# Generators near 1.0 (synchronous), loads/renewables deviate.
OMEGAS = np.array(
    [
        1.0,
        1.02,
        0.98,  # generator_rotor
        0.5,
        0.5,  # area_frequency
        0.3,
        0.3,  # tie_line
        0.1,
        0.05,
        0.2,  # load_demand
        0.15,
        0.08,  # renewable_intermittency
    ]
)


def _build_layer_map(spec):
    osc_idx = 0
    ranges = {}
    for layer in spec.layers:
        n = len(layer.oscillator_ids)
        ranges[layer.index] = list(range(osc_idx, osc_idx + n))
        osc_idx += n
    return ranges


def main():
    spec = load_binding_spec(SPEC_PATH)
    errors = validate_binding_spec(spec)
    if errors:
        raise RuntimeError(f"Invalid spec: {errors}")

    n_osc = sum(len(layer.oscillator_ids) for layer in spec.layers)
    builder = CouplingBuilder()
    coupling = builder.build(
        n_osc, spec.coupling.base_strength, spec.coupling.decay_alpha
    )
    engine = UPDEEngine(n_osc, dt=spec.sample_period_s)
    boundary_observer = BoundaryObserver(spec.boundaries)
    regime_manager = RegimeManager()
    supervisor = SupervisorPolicy(regime_manager)

    policy_path = SPEC_PATH.parent / "policy.yaml"
    rules = load_policy_rules(policy_path)
    policy_engine = PolicyEngine(rules) if rules else None

    imprint_model = ImprintModel(
        spec.imprint_model.decay_rate, spec.imprint_model.saturation
    )
    imprint_state = ImprintState(m_k=np.zeros(n_osc), last_update=0.0)

    rng = np.random.default_rng(42)
    phases = rng.uniform(0, TWO_PI, n_osc)
    omegas = OMEGAS[:n_osc].copy()
    layer_map = _build_layer_map(spec)

    zeta = spec.drivers.physical.get("zeta", 0.0)
    psi_target = spec.drivers.physical.get("psi", 0.0)

    print("=== Power Grid: Load Step -> Renewable Ramp -> Gen Trip -> AGC ===\n")
    print(f"{'step':>5}  {'R_good':>6}  {'R_bad':>5}  {'regime':>10}  phase")
    print("-" * 55)

    for step in range(STEPS):
        # Phase 1 (0-49): steady-state
        # Phase 2 (50-99): sudden load step
        if step == 50:
            omegas[layer_map[3]] *= 2.5

        # Phase 3 (100-149): renewable ramp
        if 100 <= step < 150:
            omegas[layer_map[4]] += 0.01 * rng.standard_normal(len(layer_map[4]))

        # Phase 4 (150-174): generator trip (gen_coal drops out)
        if step == 150:
            omegas[layer_map[0][0]] = 0.0
            coupling_knm = coupling.knm.copy()
            coupling_knm[layer_map[0][0], :] *= 0.1
            coupling_knm[:, layer_map[0][0]] *= 0.1
            coupling = CouplingState(
                knm=coupling_knm,
                alpha=coupling.alpha,
                active_template=coupling.active_template,
            )

        # Phase 5 (175-249): AGC + policy restore
        if step == 175:
            omegas[layer_map[3]] = OMEGAS[layer_map[3][0] : layer_map[3][-1] + 1]
            zeta = 0.2

        eff_knm = imprint_model.modulate_coupling(coupling.knm, imprint_state)
        eff_alpha = imprint_model.modulate_lag(coupling.alpha, imprint_state)

        phases = engine.step(phases, omegas, eff_knm, zeta, psi_target, eff_alpha)

        layer_states = []
        for layer in spec.layers:
            ids = layer_map[layer.index]
            r, psi = compute_order_parameter(phases[ids]) if ids else (0.0, 0.0)
            layer_states.append(LayerState(R=r, psi=psi))

        n_layers = len(spec.layers)
        cla = np.zeros((n_layers, n_layers))
        for li in range(n_layers):
            for lj in range(li + 1, n_layers):
                pi, pj = phases[layer_map[li]], phases[layer_map[lj]]
                mn = min(len(pi), len(pj))
                plv = compute_plv(pi[:mn], pj[:mn])
                cla[li, lj] = cla[lj, li] = plv

        mean_r = float(np.mean([ls.R for ls in layer_states]))
        upde_state = UPDEState(
            layers=layer_states,
            cross_layer_alignment=cla,
            stability_proxy=mean_r,
            regime_id=regime_manager.current_regime.value,
        )

        gen_r = layer_states[0].R
        obs_values = {
            "R": mean_r,
            "frequency_dev": 0.5 * (1.0 - gen_r),
            "voltage_pu": 0.95 + 0.1 * gen_r,
            "rotor_angle_deg": 90.0 * (1.0 - gen_r),
        }
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
                zeta = min(zeta + act.value, 2.0)
            elif act.knob == "K" and act.scope == "global":
                coupling = CouplingState(
                    knm=coupling.knm * (1.0 + act.value),
                    alpha=coupling.alpha,
                    active_template=coupling.active_template,
                )
            elif act.knob == "Psi":
                psi_target = act.value

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
            for j in layer_map.get(idx, [])
        ]
        bad_ph = [
            phases[j]
            for idx in spec.objectives.bad_layers
            for j in layer_map.get(idx, [])
        ]
        r_good = compute_order_parameter(np.array(good_ph))[0] if good_ph else 0.0
        r_bad = compute_order_parameter(np.array(bad_ph))[0] if bad_ph else 0.0

        if step % 25 == 0:
            if step < 50:
                label = "steady"
            elif step < 100:
                label = "load-step"
            elif step < 150:
                label = "renewable"
            elif step < 175:
                label = "gen-trip"
            else:
                label = "AGC"
            print(
                f"{step:5d}  {r_good:.4f}  {r_bad:.4f}  "
                f"{regime_manager.current_regime.value:>10}  {label}"
            )

    good_ph = [
        phases[j] for idx in spec.objectives.good_layers for j in layer_map.get(idx, [])
    ]
    bad_ph = [
        phases[j] for idx in spec.objectives.bad_layers for j in layer_map.get(idx, [])
    ]
    r_good_f = compute_order_parameter(np.array(good_ph))[0] if good_ph else 0.0
    r_bad_f = compute_order_parameter(np.array(bad_ph))[0] if bad_ph else 0.0
    print(
        f"\nFinal  R_good={r_good_f:.4f}  R_bad={r_bad_f:.4f}"
        f"  regime={regime_manager.current_regime.value}"
    )


if __name__ == "__main__":
    main()
