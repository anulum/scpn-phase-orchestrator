# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for drivers, oscillators, coherence, metrics

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.drivers.psi_physical import PhysicalDriver
from scpn_phase_orchestrator.monitor.coherence import CoherenceMonitor
from scpn_phase_orchestrator.oscillators.base import PhaseState
from scpn_phase_orchestrator.oscillators.quality import PhaseQualityScorer
from scpn_phase_orchestrator.upde.metrics import (
    LayerState,
    LockSignature,
    UPDEState,
)

TWO_PI = 2.0 * np.pi


def _make_upde_state(r_values: list[float], cla: np.ndarray | None = None) -> UPDEState:
    layers = [LayerState(R=r, psi=0.0) for r in r_values]
    n = len(r_values)
    if cla is None:
        cla = np.eye(n)
    return UPDEState(
        layers=layers,
        cross_layer_alignment=cla,
        stability_proxy=float(np.mean(r_values)),
        regime_id="nominal",
    )


# ── PhysicalDriver ──────────────────────────────────────────────────────


class TestPhysicalDriver:
    def test_zero_time(self) -> None:
        d = PhysicalDriver(frequency=1.0, amplitude=2.0)
        assert d.compute(0.0) == 0.0

    def test_quarter_period(self) -> None:
        d = PhysicalDriver(frequency=1.0, amplitude=1.0)
        assert abs(d.compute(0.25) - 1.0) < 1e-10

    def test_negative_freq_raises(self) -> None:
        with pytest.raises(ValueError):
            PhysicalDriver(frequency=-1.0)

    def test_zero_freq_raises(self) -> None:
        with pytest.raises(ValueError):
            PhysicalDriver(frequency=0.0)

    def test_batch(self) -> None:
        d = PhysicalDriver(frequency=2.0, amplitude=3.0)
        t = np.array([0.0, 0.125, 0.25])
        result = d.compute_batch(t)
        assert result.shape == (3,)
        assert abs(result[0]) < 1e-10
        assert abs(result[1] - 3.0) < 1e-10

    def test_batch_bounded(self) -> None:
        d = PhysicalDriver(frequency=5.0, amplitude=2.0)
        t = np.linspace(0, 1, 100)
        result = d.compute_batch(t)
        assert np.all(np.abs(result) <= 2.0 + 1e-10)


# ── PhaseQualityScorer ──────────────────────────────────────────────────


def _ps(q: float, amp: float = 1.0) -> PhaseState:
    return PhaseState(
        theta=0.0, omega=1.0, amplitude=amp, quality=q, channel="P", node_id="x"
    )


class TestPhaseQualityScorer:
    def test_empty_score_zero(self) -> None:
        assert PhaseQualityScorer().score([]) == 0.0

    def test_uniform_quality(self) -> None:
        states = [_ps(0.8), _ps(0.8), _ps(0.8)]
        assert abs(PhaseQualityScorer().score(states) - 0.8) < 1e-10

    def test_weighted_by_amplitude(self) -> None:
        states = [_ps(1.0, amp=10.0), _ps(0.0, amp=0.001)]
        score = PhaseQualityScorer().score(states)
        assert score > 0.9

    def test_collapse_empty(self) -> None:
        assert PhaseQualityScorer().detect_collapse([]) is True

    def test_collapse_low_quality(self) -> None:
        states = [_ps(0.01), _ps(0.02), _ps(0.5)]
        assert PhaseQualityScorer().detect_collapse(states) is True

    def test_no_collapse_high_quality(self) -> None:
        states = [_ps(0.9), _ps(0.8), _ps(0.7)]
        assert PhaseQualityScorer().detect_collapse(states) is False

    def test_downweight_mask_shape(self) -> None:
        states = [_ps(0.1), _ps(0.5), _ps(0.9)]
        mask = PhaseQualityScorer().downweight_mask(states, min_quality=0.3)
        assert mask.shape == (3,)
        assert mask[0] == 0.0
        assert mask[1] == 0.5
        assert mask[2] == 0.9

    def test_downweight_mask_empty(self) -> None:
        mask = PhaseQualityScorer().downweight_mask([])
        assert len(mask) == 0


# ── CoherenceMonitor ─────────────────────────────────────────────────────


class TestCoherenceMonitor:
    def test_r_good(self) -> None:
        state = _make_upde_state([0.9, 0.8, 0.3, 0.2])
        cm = CoherenceMonitor(good_layers=[0, 1], bad_layers=[2, 3])
        assert abs(cm.compute_r_good(state) - 0.85) < 1e-10

    def test_r_bad(self) -> None:
        state = _make_upde_state([0.9, 0.8, 0.3, 0.2])
        cm = CoherenceMonitor(good_layers=[0, 1], bad_layers=[2, 3])
        assert abs(cm.compute_r_bad(state) - 0.25) < 1e-10

    def test_empty_layers(self) -> None:
        state = _make_upde_state([0.5])
        cm = CoherenceMonitor(good_layers=[], bad_layers=[])
        assert cm.compute_r_good(state) == 0.0

    def test_detect_phase_lock_high_cla(self) -> None:
        cla = np.array([[1.0, 0.95], [0.95, 1.0]])
        state = _make_upde_state([0.9, 0.9], cla=cla)
        cm = CoherenceMonitor(good_layers=[0], bad_layers=[1])
        locked = cm.detect_phase_lock(state, threshold=0.9)
        assert (0, 1) in locked

    def test_detect_phase_lock_low_cla(self) -> None:
        cla = np.array([[1.0, 0.1], [0.1, 1.0]])
        state = _make_upde_state([0.5, 0.5], cla=cla)
        cm = CoherenceMonitor(good_layers=[0], bad_layers=[1])
        locked = cm.detect_phase_lock(state, threshold=0.9)
        assert len(locked) == 0


# ── UPDEState / LayerState / LockSignature dataclasses ───────────────────


class TestMetricsDataclasses:
    def test_lock_signature_fields(self) -> None:
        sig = LockSignature(source_layer=0, target_layer=1, plv=0.95, mean_lag=0.1)
        assert sig.plv == 0.95

    def test_layer_state_defaults(self) -> None:
        ls = LayerState(R=0.8, psi=1.0)
        assert ls.mean_amplitude == 0.0
        assert ls.amplitude_spread == 0.0
        assert ls.lock_signatures == {}

    def test_upde_state_defaults(self) -> None:
        state = _make_upde_state([0.5])
        assert state.pac_max == 0.0
        assert state.boundary_violation_count == 0


class TestDriversOscillatorsPipelineWiring:
    """Pipeline: driver Ψ → engine zeta/psi → R."""

    def test_physical_driver_feeds_engine(self):
        """PhysicalDriver.compute → psi_drive → engine.step → R∈[0,1]."""
        import numpy as np

        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import (
            compute_order_parameter,
        )

        n = 4
        driver = PhysicalDriver(frequency=5.0, amplitude=0.3)
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = 0.5 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))

        for step in range(200):
            t = step * 0.01
            psi = driver.compute(t)
            phases = eng.step(
                phases,
                omegas,
                knm,
                0.3,
                psi,
                alpha,
            )
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0
