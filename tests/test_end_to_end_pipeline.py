# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anylum.li
# SCPN Phase Orchestrator — End-to-end pipeline integration tests

"""Full pipeline: binding_spec → extraction → UPDE/SL → regime → policy."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scpn_phase_orchestrator.binding.loader import load_binding_spec
from scpn_phase_orchestrator.coupling.knm import CouplingBuilder
from scpn_phase_orchestrator.imprint.state import ImprintState
from scpn_phase_orchestrator.imprint.update import ImprintModel
from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
from scpn_phase_orchestrator.oscillators.physical import PhysicalExtractor
from scpn_phase_orchestrator.oscillators.symbolic import SymbolicExtractor
from scpn_phase_orchestrator.supervisor.policy_rules import (
    PolicyAction,
    PolicyCondition,
    PolicyEngine,
    PolicyRule,
)
from scpn_phase_orchestrator.supervisor.regimes import Regime, RegimeManager
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter
from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

TWO_PI = 2.0 * np.pi
DOMAINPACK_DIR = Path(__file__).parent.parent / "domainpacks"


def _load_spec(name: str):
    return load_binding_spec(DOMAINPACK_DIR / name / "binding_spec.yaml")


class TestEndToEndKuramoto:
    """Load a domainpack, run UPDE, evaluate regime, apply policy."""

    def test_cardiac_rhythm_pipeline(self):
        spec = _load_spec("cardiac_rhythm")
        n = sum(len(layer.oscillator_ids) for layer in spec.layers)
        assert n == 10  # 3+2+2+3

        coupling = CouplingBuilder().build(
            n,
            spec.coupling.base_strength,
            spec.coupling.decay_alpha,
        )

        rng = np.random.default_rng(0)
        phases = rng.uniform(0, TWO_PI, n)
        engine = UPDEEngine(n, dt=spec.control_period_s, method="rk4")

        r_initial, _ = compute_order_parameter(phases)

        for _ in range(500):
            phases = engine.step(
                phases,
                np.ones(n),
                coupling.knm,
                0.0,
                0.0,
                coupling.alpha,
            )

        r_final, _ = compute_order_parameter(phases)
        assert r_final > r_initial or r_final > 0.5

        layers = []
        offset = 0
        for layer_def in spec.layers:
            n_osc = len(layer_def.oscillator_ids)
            layer_phases = phases[offset : offset + n_osc]
            r, psi = compute_order_parameter(layer_phases)
            layers.append(LayerState(R=r, psi=psi))
            offset += n_osc

        state = UPDEState(
            layers=layers,
            cross_layer_alignment=np.zeros((len(layers), len(layers))),
            stability_proxy=r_final,
            regime_id="nominal",
        )

        boundary = BoundaryState()
        mgr = RegimeManager()
        regime = mgr.evaluate(state, boundary)
        mgr.transition(regime)
        assert isinstance(regime, Regime)

    def test_neuroscience_eeg_with_policy(self):
        spec = _load_spec("neuroscience_eeg")
        n = sum(len(layer.oscillator_ids) for layer in spec.layers)

        coupling = CouplingBuilder().build(
            n,
            spec.coupling.base_strength,
            spec.coupling.decay_alpha,
        )
        engine = UPDEEngine(n, dt=spec.control_period_s, method="rk4")

        rng = np.random.default_rng(1)
        phases = rng.uniform(0, TWO_PI, n)

        for _ in range(200):
            phases = engine.step(
                phases,
                np.ones(n),
                coupling.knm,
                0.0,
                0.0,
                coupling.alpha,
            )

        r, _ = compute_order_parameter(phases)

        rule = PolicyRule(
            name="boost_coupling",
            regimes=["DEGRADED"],
            condition=PolicyCondition(
                metric="stability_proxy",
                layer=None,
                op="<",
                threshold=0.6,
            ),
            actions=[PolicyAction(knob="K", scope="global", value=0.1, ttl_s=10.0)],
        )
        policy = PolicyEngine([rule])

        layers = [LayerState(R=r, psi=0.0)]
        state = UPDEState(
            layers=layers,
            cross_layer_alignment=np.zeros((1, 1)),
            stability_proxy=r,
            regime_id="nominal",
        )

        mgr = RegimeManager()
        regime = mgr.evaluate(state, BoundaryState())
        regime = mgr.transition(regime)
        actions = policy.evaluate(regime, state, [0], [])

        if regime == Regime.DEGRADED and r < 0.6:
            assert len(actions) >= 1
            assert actions[0].knob == "K"


class TestEndToEndStuartLandau:
    """Load a domainpack with amplitude config, run Stuart-Landau pipeline."""

    def test_cardiac_stuart_landau(self):
        spec = _load_spec("cardiac_rhythm")
        n = sum(len(layer.oscillator_ids) for layer in spec.layers)
        assert spec.amplitude is not None

        coupling = CouplingBuilder().build(
            n,
            spec.coupling.base_strength,
            spec.coupling.decay_alpha,
        )

        rng = np.random.default_rng(42)
        state = np.concatenate(
            [
                rng.uniform(0, TWO_PI, n),
                rng.uniform(0.5, 1.5, n),
            ]
        )
        omegas = np.ones(n)
        mu = np.full(n, spec.amplitude.mu)
        knm_r = coupling.knm * spec.amplitude.amp_coupling_strength
        alpha = coupling.alpha

        eng = StuartLandauEngine(n, dt=spec.control_period_s, method="rk4")
        for _ in range(1000):
            state = eng.step(
                state,
                omegas,
                mu,
                coupling.knm,
                knm_r,
                0.0,
                0.0,
                alpha,
                epsilon=spec.amplitude.epsilon,
            )

        phases = state[:n]
        amps = state[n:]
        assert np.all(phases >= 0.0)
        assert np.all(phases < TWO_PI + 1e-12)
        assert np.all(amps >= 0.0)
        assert np.all(np.isfinite(state))


class TestEndToEndImprint:
    """Imprint model accumulates exposure and modulates coupling."""

    def test_imprint_modulates_coupling(self):
        spec = _load_spec("cardiac_rhythm")
        n = sum(len(layer.oscillator_ids) for layer in spec.layers)

        coupling = CouplingBuilder().build(
            n,
            spec.coupling.base_strength,
            spec.coupling.decay_alpha,
        )

        model = ImprintModel(
            decay_rate=spec.imprint_model.decay_rate,
            saturation=spec.imprint_model.saturation,
        )
        imprint = ImprintState(m_k=np.zeros(n), last_update=0.0)

        exposure = np.ones(n) * 0.8
        imprint = model.update(imprint, exposure, dt=1.0)
        assert np.all(imprint.m_k > 0)

        knm_modulated = model.modulate_coupling(
            coupling.knm.copy(),
            imprint,
        )
        assert np.any(knm_modulated != coupling.knm)


class TestEndToEndChannels:
    """Verify P/I/S extractors produce valid phases."""

    def test_physical_extractor(self):
        rng = np.random.default_rng(0)
        signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 256))
        signal += rng.normal(0, 0.1, 256)
        ext = PhysicalExtractor()
        states = ext.extract(signal, sample_rate=256.0)
        assert len(states) >= 1
        assert 0.0 <= states[-1].theta < TWO_PI + 1e-6
        assert states[-1].quality >= 0.0

    def test_symbolic_extractor(self):
        ext = SymbolicExtractor(n_states=6)
        signal = np.array([2])  # state index 2
        states = ext.extract(signal, sample_rate=1.0)
        assert len(states) == 1
        expected = 2.0 * np.pi * 2 / 6
        assert abs(states[0].theta - expected) < 1e-10


class TestAllDomainpacksLoad:
    """Every domainpack loads without error."""

    def test_all_25_load(self):
        loaded = 0
        for pack_dir in sorted(DOMAINPACK_DIR.iterdir()):
            if not pack_dir.is_dir():
                continue
            spec_file = pack_dir / "binding_spec.yaml"
            if not spec_file.exists():
                continue
            spec = load_binding_spec(spec_file)
            assert spec.name == pack_dir.name
            assert len(spec.layers) > 0
            assert len(spec.oscillator_families) > 0
            loaded += 1
        assert loaded == 25
