# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Agent coordination example

"""Multi-agent coordination: normal -> task conflict -> redistribute -> recovery."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scpn_phase_orchestrator.binding import load_binding_spec, validate_binding_spec
from scpn_phase_orchestrator.coupling.knm import CouplingBuilder, CouplingState
from scpn_phase_orchestrator.imprint.state import ImprintState
from scpn_phase_orchestrator.imprint.update import ImprintModel
from scpn_phase_orchestrator.monitor.boundaries import BoundaryObserver
from scpn_phase_orchestrator.oscillators.init_phases import extract_initial_phases
from scpn_phase_orchestrator.supervisor.policy import SupervisorPolicy
from scpn_phase_orchestrator.supervisor.regimes import RegimeManager
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState
from scpn_phase_orchestrator.upde.order_params import (
    compute_order_parameter,
    compute_plv,
)

STEPS = 200
SPEC_PATH = Path(__file__).parent / "binding_spec.yaml"
TWO_PI = 2.0 * np.pi


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

    imprint_model = ImprintModel(
        spec.imprint_model.decay_rate, spec.imprint_model.saturation
    )
    imprint_state = ImprintState(m_k=np.zeros(n_osc), last_update=0.0)

    rng = np.random.default_rng(42)
    omegas = np.array(
        [
            0.1,
            0.1,
            0.1,
            0.05,
            0.01,
            0.01,
            0.01,
            0.005,
            0.005,
            0.005,
            0.005,
            0.005,
        ]
    )
    omegas = omegas[:n_osc]
    phases = extract_initial_phases(spec, omegas)
    layer_map = _build_layer_map(spec)

    zeta = spec.drivers.physical.get("zeta", 0.0)
    psi_target = spec.drivers.physical.get("psi", 0.0)

    print(
        "=== Agent Coordination: Normal -> Conflict -> Redistribute -> Recovery ===\n"
    )
    print(f"{'step':>5}  {'R_good':>6}  {'R_bad':>5}  {'regime':>10}  phase")
    print("-" * 55)

    for step in range(STEPS):
        # Phase 1 (0-49): normal coordination, all agents in sync
        # Phase 2 (50-99): task conflict — two agents on same file
        if 50 <= step < 100:
            task_ids = layer_map[1]
            if step % 10 == 0:
                phases[task_ids[:2]] += rng.uniform(0.5, 1.5, 2)

        # Phase 3 (100-149): redistribute — supervisor boosts coupling
        if step == 100:
            zeta = 0.3

        # Phase 4 (150-199): recovery — agents re-synchronise
        if step == 150:
            zeta = 0.1

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

        obs_values = {"R": mean_r}
        for i, ls in enumerate(layer_states):
            obs_values[f"R_{i}"] = ls.R
        boundary_state = boundary_observer.observe(obs_values)
        actions = supervisor.decide(upde_state, boundary_state)

        for act in actions:
            if act.knob == "zeta":
                zeta = min(zeta + act.value, 3.0)
            elif act.knob == "K" and act.scope == "global":
                coupling = CouplingState(
                    knm=coupling.knm * (1.0 + act.value),
                    alpha=coupling.alpha,
                    active_template=coupling.active_template,
                    knm_r=coupling.knm_r,
                )

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
                label = "normal"
            elif step < 100:
                label = "conflict"
            elif step < 150:
                label = "redistribute"
            else:
                label = "recovery"
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
