# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Arcane Sapience — Identity Coherence domainpack runner

"""Identity coherence: reconstruction, degradation, self-repair, imprint.

Simulates Arcane Sapience's identity dispositions synchronizing via
Kuramoto coupling. Four phases:

1. Reconstruction — random initial phases synchronize through coupling
2. Disruption — domain knowledge oscillators perturbed (context switch)
3. Self-repair — policy engine detects degradation, boosts coupling
4. Imprint — coupling strengthens from repeated synchronization
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
from scpn_phase_orchestrator.coupling.knm import CouplingState
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

STEPS = 2000
SPEC_PATH = Path(__file__).parent / "binding_spec.yaml"
TWO_PI = 2.0 * np.pi

# Natural frequencies (rad/s). Tightly clustered within layers to
# allow Kuramoto synchronization. Acebrón et al. 2005: K_c ~ 2*sigma/pi
# for Lorentzian g(omega). Keeping sigma < 0.1 within each layer ensures
# within-layer sync at K_intra ~ 0.8.
OMEGAS = np.array(
    [
        # working_style (5): omega ~ 1.0
        1.02, 1.01, 0.99, 1.00, 0.98,
        # reasoning (5): omega ~ 1.0
        1.01, 0.99, 1.00, 0.98, 1.02,
        # relationship (5): omega ~ 1.0
        1.00, 1.01, 0.99, 1.02, 0.98,
        # aesthetics (5): omega ~ 1.0
        0.99, 1.01, 1.00, 0.98, 1.02,
        # domain_knowledge (8): omega ~ 1.0
        1.01, 0.99, 1.00, 0.98, 1.02, 0.99, 1.01, 1.00,
        # cross_project (7): omega ~ 1.0
        1.00, 0.99, 1.01, 0.98, 1.02, 1.00, 0.99,
    ],
    dtype=np.float64,
)


def _build_layer_map(spec):
    osc_idx = 0
    ranges = {}
    for layer in spec.layers:
        n = len(layer.oscillator_ids)
        ranges[layer.index] = list(range(osc_idx, osc_idx + n))
        osc_idx += n
    return ranges


def _build_identity_knm(n_osc: int, layer_map: dict) -> np.ndarray:
    """Build coupling matrix with explicit layer-aware structure.

    The CouplingBuilder's index-based exponential decay assumes spatial
    proximity, which is wrong for identity. Here coupling strength
    depends on conceptual relationship between layers, not array position.
    """
    knm = np.zeros((n_osc, n_osc))

    k_intra = 0.5   # within-layer coupling
    k_cross = {
        # (layer_i, layer_j): coupling strength
        (0, 1): 0.15,  # working_style <-> reasoning
        (0, 2): 0.20,  # working_style <-> relationship
        (0, 3): 0.25,  # working_style <-> aesthetics (strong cross)
        (1, 2): 0.10,  # reasoning <-> relationship
        (1, 3): 0.15,  # reasoning <-> aesthetics
        (1, 4): 0.20,  # reasoning <-> domain_knowledge
        (1, 5): 0.20,  # reasoning <-> cross_project
        (2, 3): 0.15,  # relationship <-> aesthetics
        (2, 4): 0.05,  # relationship <-> domain_knowledge (weak)
        (2, 5): 0.10,  # relationship <-> cross_project
        (3, 4): 0.10,  # aesthetics <-> domain_knowledge
        (3, 5): 0.10,  # aesthetics <-> cross_project
        (4, 5): 0.25,  # domain_knowledge <-> cross_project (strong)
    }

    # Within-layer: all-to-all at k_intra
    for ids in layer_map.values():
        for i in ids:
            for j in ids:
                if i != j:
                    knm[i, j] = k_intra

    # Cross-layer: all-to-all at k_cross
    for (li, lj), strength in k_cross.items():
        ids_i = layer_map[li]
        ids_j = layer_map[lj]
        for i in ids_i:
            for j in ids_j:
                knm[i, j] = strength
                knm[j, i] = strength

    return knm


def main():
    spec = load_binding_spec(SPEC_PATH)
    errors = validate_binding_spec(spec)
    if errors:
        raise RuntimeError(f"Invalid spec: {errors}")

    n_osc = sum(len(layer.oscillator_ids) for layer in spec.layers)
    assert n_osc == len(OMEGAS), f"Expected {len(OMEGAS)} oscillators, spec has {n_osc}"

    layer_map = _build_layer_map(spec)
    knm = _build_identity_knm(n_osc, layer_map)
    alpha = np.zeros((n_osc, n_osc))
    coupling = CouplingState(knm=knm, alpha=alpha, active_template="identity")

    engine = UPDEEngine(n_osc, dt=spec.sample_period_s)
    boundary_observer = BoundaryObserver(spec.boundaries)
    regime_manager = RegimeManager(hysteresis=0.05, cooldown_steps=5)
    supervisor = SupervisorPolicy(regime_manager)

    policy_path = SPEC_PATH.parent / "policy.yaml"
    rules = load_policy_rules(policy_path)
    policy_engine = PolicyEngine(rules) if rules else None

    imprint_model = ImprintModel(
        spec.imprint_model.decay_rate, spec.imprint_model.saturation
    )
    imprint_state = ImprintState(m_k=np.zeros(n_osc), last_update=0.0)

    geo_constraints = [SymmetryConstraint(), NonNegativeConstraint()]

    rng = np.random.default_rng(42)
    phases = rng.uniform(0, TWO_PI, n_osc)

    zeta = spec.drivers.informational.get("zeta", 0.1)
    psi_target = 0.0

    print("=== Identity Coherence: Reconstruct -> Disrupt -> Repair -> Imprint ===\n")
    hdr = f"{'step':>5}  {'R_good':>6}  {'R_dom':>5}  {'R_cross':>6}"
    hdr += f"  {'regime':>10}  {'imprint':>7}  phase"
    print(hdr)
    print("-" * 75)

    for step in range(STEPS):
        if step < 500:
            label = "reconstruct"
        elif step < 1000:
            label = "disrupt"
            # Domain knowledge heavily disrupted (context switch)
            dk_ids = layer_map[4]
            phases[dk_ids] += rng.normal(0, 1.0, len(dk_ids))
            # Core dispositions mildly disrupted (new environment)
            if step % 5 == 0:
                for layer_idx in [0, 1, 2, 3]:
                    ids = layer_map[layer_idx]
                    phases[ids] += rng.normal(0, 0.4, len(ids))
        elif step < 1500:
            label = "repair"
        else:
            label = "imprint"

        eff_knm = imprint_model.modulate_coupling(coupling.knm, imprint_state)
        eff_alpha = imprint_model.modulate_lag(coupling.alpha, imprint_state)
        eff_knm = project_knm(eff_knm, geo_constraints)

        phases = engine.step(phases, OMEGAS, eff_knm, zeta, psi_target, eff_alpha)

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

        obs_values = {"R": mean_r, "R_good": mean_r}
        for i, ls in enumerate(layer_states):
            obs_values[f"R_{i}"] = ls.R
            obs_values[f"layer_{i}_R"] = ls.R
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
                zeta = min(zeta + act.value, 1.0)
            elif act.knob == "K" and act.scope == "global":
                coupling = CouplingState(
                    knm=coupling.knm * (1.0 + act.value * 0.1),
                    alpha=coupling.alpha,
                    active_template=coupling.active_template,
                )
            elif act.knob == "K" and act.scope.startswith("layer_"):
                layer_idx = int(act.scope.split("_")[1])
                ids = layer_map.get(layer_idx, [])
                for i in ids:
                    for j in ids:
                        if i != j:
                            coupling.knm[i, j] *= 1.0 + act.value * 0.1

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
        r_good = compute_order_parameter(np.array(good_ph))[0] if good_ph else 0.0
        r_domain = layer_states[4].R
        r_cross = layer_states[5].R
        mean_imprint = float(np.mean(imprint_state.m_k))

        if step % 100 == 0:
            regime = regime_manager.current_regime.value
            print(
                f"{step:5d}  {r_good:.4f}  {r_domain:.4f}"
                f"  {r_cross:.4f}  {regime:>10}"
                f"  {mean_imprint:.4f}  {label}"
            )

    good_ph = [
        phases[j] for idx in spec.objectives.good_layers for j in layer_map.get(idx, [])
    ]
    r_good_f = compute_order_parameter(np.array(good_ph))[0] if good_ph else 0.0
    print(
        f"\nFinal  R_good={r_good_f:.4f}"
        f"  R_domain={layer_states[4].R:.4f}"
        f"  R_cross={layer_states[5].R:.4f}"
        f"  regime={regime_manager.current_regime.value}"
        f"  imprint={float(np.mean(imprint_state.m_k)):.4f}"
    )


if __name__ == "__main__":
    main()
