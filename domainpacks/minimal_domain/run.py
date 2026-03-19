# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Minimal domain example

"""Minimal domain: baseline -> perturbation -> recovery (full pipeline)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scpn_phase_orchestrator.binding import load_binding_spec, validate_binding_spec
from scpn_phase_orchestrator.coupling.knm import CouplingBuilder, CouplingState
from scpn_phase_orchestrator.monitor.boundaries import BoundaryObserver
from scpn_phase_orchestrator.oscillators.init_phases import extract_initial_phases
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
from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

STEPS = 150
SPEC_PATH = Path(__file__).parent / "binding_spec.yaml"
TWO_PI = 2.0 * np.pi

OMEGAS = np.array([1.0, 1.05, 0.95, 1.0])


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
    amplitude_mode = spec.amplitude is not None
    if amplitude_mode:
        amp = spec.amplitude
        coupling = builder.build_with_amplitude(
            n_osc,
            spec.coupling.base_strength,
            spec.coupling.decay_alpha,
            amp.amp_coupling_strength,
            amp.amp_coupling_decay,
        )
        sl_engine = StuartLandauEngine(n_osc, dt=spec.sample_period_s)
        mu = np.full(n_osc, amp.mu)
    else:
        coupling = builder.build(
            n_osc, spec.coupling.base_strength, spec.coupling.decay_alpha
        )
        engine = UPDEEngine(n_osc, dt=spec.sample_period_s)
    boundary_observer = BoundaryObserver(spec.boundaries)
    regime_manager = RegimeManager()
    supervisor = SupervisorPolicy(regime_manager)

    policy_path = SPEC_PATH.parent / "policy.yaml"
    rules = load_policy_rules(policy_path) if policy_path.exists() else []
    policy_engine = PolicyEngine(rules) if rules else None

    rng = np.random.default_rng(42)
    omegas = OMEGAS[:n_osc].copy()
    phases = extract_initial_phases(spec, omegas)
    if amplitude_mode:
        sl_state = np.concatenate([phases, np.sqrt(np.maximum(mu, 0.0))])
    layer_map = _build_layer_map(spec)

    zeta = spec.drivers.physical.get("zeta", 0.0)
    psi_target = spec.drivers.physical.get("psi", 0.0)

    print("=== Minimal Domain: Baseline -> Perturbation -> Recovery ===\n")
    print(f"{'step':>5}  {'R_good':>6}  {'R_bad':>5}  {'regime':>10}  phase")
    print("-" * 55)

    for step in range(STEPS):
        # Phase 1 (0-49): baseline
        # Phase 2 (50-99): perturbation — scatter phases
        if step == 50:
            phases += rng.uniform(0, TWO_PI, n_osc)

        # Phase 3 (100-149): recovery — boost coupling
        if step == 100:
            coupling = CouplingState(
                knm=coupling.knm * 1.5,
                alpha=coupling.alpha,
                active_template=coupling.active_template,
                knm_r=coupling.knm_r,
            )
            zeta = 0.2

        if amplitude_mode:
            sl_state = sl_engine.step(
                sl_state,
                omegas,
                mu,
                coupling.knm,
                coupling.knm_r,
                zeta,
                psi_target,
                coupling.alpha,
                epsilon=amp.epsilon,
            )
            phases = sl_state[:n_osc]
        else:
            phases = engine.step(
                phases, omegas, coupling.knm, zeta, psi_target, coupling.alpha
            )

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
            if act.knob == "K" and act.scope == "global":
                coupling = CouplingState(
                    knm=coupling.knm * (1.0 + act.value),
                    alpha=coupling.alpha,
                    active_template=coupling.active_template,
                    knm_r=coupling.knm_r,
                )
            elif act.knob == "zeta":
                zeta = min(zeta + act.value, 1.0)

        good_ph = [
            phases[j]
            for idx in spec.objectives.good_layers
            for j in layer_map.get(idx, [])
        ]
        r_good = compute_order_parameter(np.array(good_ph))[0] if good_ph else 0.0
        r_bad = 0.0  # no bad_layers

        if step % 15 == 0:
            if step < 50:
                label = "baseline"
            elif step < 100:
                label = "perturbed"
            else:
                label = "recovery"
            print(
                f"{step:5d}  {r_good:.4f}  {r_bad:.4f}  "
                f"{regime_manager.current_regime.value:>10}  {label}"
            )

    good_ph = [
        phases[j] for idx in spec.objectives.good_layers for j in layer_map.get(idx, [])
    ]
    r_good_f = compute_order_parameter(np.array(good_ph))[0] if good_ph else 0.0
    print(
        f"\nFinal  R_good={r_good_f:.4f}  R_bad=0.0000"
        f"  regime={regime_manager.current_regime.value}"
    )


if __name__ == "__main__":
    main()
