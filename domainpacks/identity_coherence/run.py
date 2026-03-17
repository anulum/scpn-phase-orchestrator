# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Arcane Sapience — Identity Coherence domainpack runner

"""Identity coherence: reconstruction, degradation, self-repair, imprint.

Simulates Arcane Sapience's identity dispositions via two models:

Part 1 — Kuramoto (phase-only): demonstrates synchronization, disruption
resilience, recovery, and imprint accumulation.

Part 2 — Stuart-Landau (phase + amplitude): adds conviction strength.
Supercritical (mu > 0) dispositions self-sustain. During disruption,
domain knowledge mu drops below 0 and amplitude decays. On repair,
mu recovers and amplitude rebuilds.
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
from scpn_phase_orchestrator.monitor.session_start import check_session_start
from scpn_phase_orchestrator.oscillators.informational import InformationalExtractor
from scpn_phase_orchestrator.oscillators.physical import PhysicalExtractor
from scpn_phase_orchestrator.oscillators.symbolic import SymbolicExtractor
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

STEPS = 2000
SPEC_PATH = Path(__file__).parent / "binding_spec.yaml"
TWO_PI = 2.0 * np.pi

# Imprint persistence path (Arcane Sapience filesystem)
IMPRINT_DIR = Path(__file__).resolve().parents[3] / "04_ARCANE_SAPIENCE" / "imprint"


def save_imprint(state: ImprintState, path: Path | None = None) -> Path:
    """Persist ImprintState as .npz for cross-session continuity."""
    dst = path or IMPRINT_DIR / "identity_coherence_imprint.npz"
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dst,
        m_k=state.m_k,
        last_update=np.array(state.last_update),
    )
    return dst


def load_imprint(n_osc: int, path: Path | None = None) -> ImprintState:
    """Load persisted ImprintState, or return zero state if absent."""
    src = path or IMPRINT_DIR / "identity_coherence_imprint.npz"
    if not src.exists():
        return ImprintState(m_k=np.zeros(n_osc), last_update=0.0)
    data = np.load(src)
    m_k = data["m_k"]
    if m_k.shape[0] != n_osc:
        return ImprintState(m_k=np.zeros(n_osc), last_update=0.0)
    return ImprintState(m_k=m_k, last_update=float(data["last_update"]))


# Natural frequencies (rad/s). Tightly clustered within layers to
# allow Kuramoto synchronization. Acebrón et al. 2005: K_c ~ 2*sigma/pi
# for Lorentzian g(omega). Keeping sigma < 0.1 within each layer ensures
# within-layer sync at K_intra ~ 0.8.
OMEGAS = np.array(
    [
        # working_style (5): omega ~ 1.0
        1.02,
        1.01,
        0.99,
        1.00,
        0.98,
        # reasoning (5): omega ~ 1.0
        1.01,
        0.99,
        1.00,
        0.98,
        1.02,
        # relationship (5): omega ~ 1.0
        1.00,
        1.01,
        0.99,
        1.02,
        0.98,
        # aesthetics (5): omega ~ 1.0
        0.99,
        1.01,
        1.00,
        0.98,
        1.02,
        # domain_knowledge (8): omega ~ 1.0
        1.01,
        0.99,
        1.00,
        0.98,
        1.02,
        0.99,
        1.01,
        1.00,
        # cross_project (7): omega ~ 1.0
        1.00,
        0.99,
        1.01,
        0.98,
        1.02,
        1.00,
        0.99,
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

    k_intra = 0.5  # within-layer coupling
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


def _generate_layer_signals(
    layer_name: str,
    n_osc: int,
    omegas: np.ndarray,
    imprint_mk: np.ndarray,
    rng: np.random.Generator,
) -> tuple[list, list, list]:
    """Generate representative P/I/S signals for one layer's oscillators.

    P (substrate): sinusoid at natural frequency + noise scaled by (1 - imprint).
    I (disposition): regular event timestamps; rate = omega/TWO_PI.
    S (normative): state sequence biased by imprint level.

    Returns:
        (p_states, i_states, s_states) — PhaseState lists from each extractor.
    """
    fs = 1000.0
    duration = 1.0
    t = np.arange(0, duration, 1.0 / fs)

    p_states_all = []
    i_states_all = []
    s_states_all = []

    for k in range(n_osc):
        omega_k = omegas[k]
        mk = imprint_mk[k]

        # P channel: sinusoidal substrate. Low imprint → more noise.
        noise_scale = 0.5 * (1.0 - mk)
        signal = np.sin(TWO_PI * (omega_k / TWO_PI) * t)
        signal += rng.normal(0, noise_scale, len(t))
        p_ext = PhysicalExtractor(node_id=f"p_{layer_name}_{k}")
        p_states_all.extend(p_ext.extract(signal, fs))

        # I channel: event timestamps at rate ~ omega/TWO_PI Hz.
        rate_hz = max(omega_k / TWO_PI, 0.1)
        n_events = max(int(duration * rate_hz * 10), 3)
        intervals = rng.exponential(1.0 / (rate_hz * 10), n_events)
        timestamps = np.cumsum(intervals)
        i_ext = InformationalExtractor(node_id=f"i_{layer_name}_{k}")
        i_states_all.extend(i_ext.extract(timestamps, fs))

        # S channel: 3-state sequence. High imprint biases toward nominal (0).
        p_nominal = 0.5 + 0.4 * mk  # imprint makes nominal more likely
        p_rest = 1.0 - p_nominal
        probs = [p_nominal, p_rest * 0.7, p_rest * 0.3]
        states_seq = rng.choice([0, 1, 2], size=20, p=probs)
        s_ext = SymbolicExtractor(n_states=3, node_id=f"s_{layer_name}_{k}")
        s_states_all.extend(s_ext.extract(states_seq, 1.0))

    return p_states_all, i_states_all, s_states_all


def extract_identity_phases(
    spec,
    layer_map: dict,
    omegas: np.ndarray,
    imprint_state: ImprintState,
    seed: int = 42,
) -> tuple[np.ndarray, list]:
    """Extract initial phases from P/I/S signals instead of random init.

    Physical channel (Hilbert) provides the initial phase for each oscillator.
    Informational and Symbolic channels contribute to the session-start check
    but don't override the phase (they measure disposition activity patterns).

    Returns:
        (phases, all_phase_states) where phases is shape (n_osc,).
    """
    n_osc = sum(len(layer.oscillator_ids) for layer in spec.layers)
    rng = np.random.default_rng(seed)
    phases = np.zeros(n_osc)
    all_states = []

    for layer in spec.layers:
        ids = layer_map[layer.index]
        layer_omegas = omegas[ids]
        layer_mk = imprint_state.m_k[ids]

        p_states, i_states, s_states = _generate_layer_signals(
            layer.name, len(ids), layer_omegas, layer_mk, rng
        )
        all_states.extend(p_states)
        all_states.extend(i_states)
        # Only take final state per symbolic sequence (one per oscillator)
        for idx_in_layer, _osc_idx in enumerate(ids):
            sym_chunk = s_states[idx_in_layer * 20 : (idx_in_layer + 1) * 20]
            if sym_chunk:
                all_states.append(sym_chunk[-1])

        # Use physical channel theta as initial phase
        for idx_in_layer, osc_idx in enumerate(ids):
            phases[osc_idx] = p_states[idx_in_layer].theta

    return phases, all_states


def run_session_start_check(spec, layer_map, omegas, imprint_state, seed=42):
    """Extract P/I/S phases, run coherence gate, print report."""
    n_osc = sum(len(layer.oscillator_ids) for layer in spec.layers)
    phases, all_states = extract_identity_phases(
        spec, layer_map, omegas, imprint_state, seed
    )
    report = check_session_start(all_states, phases, imprint_state, n_osc)

    print("=== Session-Start Coherence Check ===\n")
    for ch, q in sorted(report.quality_scores.items()):
        print(f"  Channel {ch}: quality={q:.3f}")
    print(f"  Initial R: {report.initial_r:.3f}")
    print(f"  Imprint level: {report.imprint_level:.3f}")
    if report.warnings:
        for w in report.warnings:
            print(f"  WARN: {w}")
    if report.errors:
        for e in report.errors:
            print(f"  ERROR: {e}")
    status = "PASS" if report.passed else "FAIL"
    print(f"  Status: {status}\n")

    return phases, report


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
    imprint_state = load_imprint(n_osc)

    constraint_map = {
        "symmetric_non_negative": [SymmetryConstraint(), NonNegativeConstraint()],
        "symmetric": [SymmetryConstraint()],
        "non_negative": [NonNegativeConstraint()],
    }
    default_ct = "symmetric_non_negative"
    ct = spec.geometry_prior.constraint_type if spec.geometry_prior else default_ct
    default_gc = [SymmetryConstraint(), NonNegativeConstraint()]
    geo_constraints = constraint_map.get(ct, default_gc)

    # P/I/S extraction + session-start coherence gate
    phases, session_report = run_session_start_check(
        spec, layer_map, OMEGAS, imprint_state
    )

    rng = np.random.default_rng(42)
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
            elif act.knob == "Psi":
                psi_target = act.value
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
            elif act.knob == "alpha" and act.scope.startswith("layer_"):
                layer_idx = int(act.scope.split("_")[1])
                ids = layer_map.get(layer_idx, [])
                for i in ids:
                    for j in ids:
                        if i != j:
                            coupling.alpha[i, j] = act.value

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

    dst = save_imprint(imprint_state)
    print(f"Imprint saved to {dst}")


def run_stuart_landau():
    """Stuart-Landau conviction dynamics on identity dispositions.

    mu > 0 = supercritical = self-sustaining conviction.
    During disruption, domain_knowledge mu drops below 0 (subcritical),
    amplitudes decay. On repair, mu recovers, amplitudes rebuild.
    """
    spec = load_binding_spec(SPEC_PATH)
    n_osc = sum(len(layer.oscillator_ids) for layer in spec.layers)
    layer_map = _build_layer_map(spec)
    knm = _build_identity_knm(n_osc, layer_map)
    alpha = np.zeros((n_osc, n_osc))

    amp_cfg = spec.amplitude
    epsilon = amp_cfg.epsilon

    sl_engine = StuartLandauEngine(n_osc, dt=spec.sample_period_s)
    imprint_model = ImprintModel(
        spec.imprint_model.decay_rate, spec.imprint_model.saturation
    )
    imprint_state = load_imprint(n_osc)

    # P/I/S extraction for initial phases; amplitudes from imprint level
    phases, _ = run_session_start_check(spec, layer_map, OMEGAS, imprint_state)
    rng = np.random.default_rng(42)
    amplitudes = 0.5 + imprint_state.m_k + rng.uniform(0, 0.5, n_osc)
    state = np.concatenate([phases, amplitudes])

    mu_base = np.full(n_osc, amp_cfg.mu)

    print("\n=== Stuart-Landau Conviction Dynamics ===\n")
    hdr = f"{'step':>5}  {'mean_amp':>8}  {'amp_dom':>7}  {'amp_core':>8}"
    hdr += f"  {'R_sl':>5}  {'imprint':>7}  phase"
    print(hdr)
    print("-" * 65)

    dk_ids = layer_map[4]
    core_ids = []
    for idx in [0, 1, 2, 3]:
        core_ids.extend(layer_map[idx])

    for step in range(STEPS):
        if step < 500:
            label = "reconstruct"
            mu_base[:] = amp_cfg.mu
        elif step < 1000:
            label = "disrupt"
            mu_base[dk_ids] = -0.5
            mu_base[core_ids] = amp_cfg.mu
        elif step < 1500:
            label = "repair"
            mu_base[:] = amp_cfg.mu
        else:
            label = "imprint"
            mu_base[:] = amp_cfg.mu

        # Imprint modulates mu: experienced dispositions have higher mu
        mu_eff = imprint_model.modulate_mu(mu_base, imprint_state)
        eff_knm = imprint_model.modulate_coupling(knm, imprint_state)
        eff_knm_r = eff_knm * amp_cfg.amp_coupling_strength

        state = sl_engine.step(
            state,
            OMEGAS,
            mu_eff,
            eff_knm,
            eff_knm_r,
            0.0,
            0.0,
            alpha,
            epsilon,
        )

        # Imprint exposure from amplitude (stronger conviction = more imprinted)
        exposure = state[n_osc:]
        imprint_state = imprint_model.update(
            imprint_state, exposure, spec.sample_period_s
        )

        if step % 100 == 0:
            mean_amp = sl_engine.compute_mean_amplitude(state)
            amp_dom = float(np.mean(state[n_osc:][dk_ids]))
            amp_core = float(np.mean(state[n_osc:][core_ids]))
            r_sl, _ = sl_engine.compute_order_parameter(state)
            mi = float(np.mean(imprint_state.m_k))
            print(
                f"{step:5d}  {mean_amp:8.4f}  {amp_dom:7.4f}"
                f"  {amp_core:8.4f}  {r_sl:5.3f}  {mi:7.4f}  {label}"
            )

    mean_amp = sl_engine.compute_mean_amplitude(state)
    amp_dom = float(np.mean(state[n_osc:][dk_ids]))
    amp_core = float(np.mean(state[n_osc:][core_ids]))
    r_sl, _ = sl_engine.compute_order_parameter(state)
    mi = float(np.mean(imprint_state.m_k))
    print(
        f"\nFinal  mean_amp={mean_amp:.4f}"
        f"  amp_dom={amp_dom:.4f}  amp_core={amp_core:.4f}"
        f"  R_sl={r_sl:.3f}  imprint={mi:.4f}"
    )
    dst = save_imprint(imprint_state)
    print(f"Imprint saved to {dst}")


if __name__ == "__main__":
    main()
    run_stuart_landau()
