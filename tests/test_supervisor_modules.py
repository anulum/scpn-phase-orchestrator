# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for events, regimes, drivers (I/S)

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.drivers.psi_informational import InformationalDriver
from scpn_phase_orchestrator.drivers.psi_symbolic import SymbolicDriver
from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
from scpn_phase_orchestrator.supervisor.events import EventBus, RegimeEvent
from scpn_phase_orchestrator.supervisor.regimes import Regime, RegimeManager
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

TWO_PI = 2.0 * np.pi


def _state(r: float) -> UPDEState:
    return UPDEState(
        layers=[LayerState(R=r, psi=0.0)],
        cross_layer_alignment=np.eye(1),
        stability_proxy=r,
        regime_id="nominal",
    )


# ── InformationalDriver ─────────────────────────────────────────────────


class TestInformationalDriver:
    def test_zero_time(self) -> None:
        d = InformationalDriver(cadence_hz=10.0)
        assert d.compute(0.0) == 0.0

    def test_output_wrapped(self) -> None:
        d = InformationalDriver(cadence_hz=1.0)
        val = d.compute(1.5)
        assert 0.0 <= val < TWO_PI

    def test_negative_cadence_raises(self) -> None:
        with pytest.raises(ValueError):
            InformationalDriver(cadence_hz=-1.0)

    def test_batch_shape(self) -> None:
        d = InformationalDriver(cadence_hz=5.0)
        t = np.linspace(0, 1, 20)
        out = d.compute_batch(t)
        assert out.shape == (20,)
        assert np.all(out >= 0)
        assert np.all(out < TWO_PI)


# ── SymbolicDriver ──────────────────────────────────────────────────────


class TestSymbolicDriver:
    def test_cyclic_access(self) -> None:
        d = SymbolicDriver([1.0, 2.0, 3.0])
        assert d.compute(0) == 1.0
        assert d.compute(3) == 1.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            SymbolicDriver([])

    def test_batch(self) -> None:
        d = SymbolicDriver([10.0, 20.0])
        steps = np.array([0, 1, 2, 3])
        out = d.compute_batch(steps)
        np.testing.assert_array_equal(out, [10.0, 20.0, 10.0, 20.0])


# ── EventBus ─────────────────────────────────────────────────────────────


class TestEventBus:
    def test_post_and_history(self) -> None:
        bus = EventBus()
        e = RegimeEvent(kind="boundary_breach", step=1)
        bus.post(e)
        assert bus.count == 1
        assert bus.history[0] is e

    def test_subscribe_callback(self) -> None:
        bus = EventBus()
        received = []
        bus.subscribe(received.append)
        bus.post(RegimeEvent(kind="manual", step=0))
        assert len(received) == 1

    def test_unsubscribe(self) -> None:
        bus = EventBus()
        received = []
        bus.subscribe(received.append)
        bus.unsubscribe(received.append)
        bus.post(RegimeEvent(kind="manual", step=0))
        assert len(received) == 0

    def test_clear(self) -> None:
        bus = EventBus()
        bus.post(RegimeEvent(kind="manual", step=0))
        bus.clear()
        assert bus.count == 0

    def test_bounded_history(self) -> None:
        bus = EventBus(maxlen=3)
        for i in range(5):
            bus.post(RegimeEvent(kind="manual", step=i))
        assert bus.count == 3

    def test_invalid_event_kind(self) -> None:
        with pytest.raises(ValueError, match="invalid event kind"):
            RegimeEvent(kind="unknown", step=0)


# ── RegimeManager ────────────────────────────────────────────────────────


class TestRegimeManager:
    def test_initial_nominal(self) -> None:
        rm = RegimeManager()
        assert rm.current_regime == Regime.NOMINAL

    def test_evaluate_critical_low_r(self) -> None:
        rm = RegimeManager()
        result = rm.evaluate(_state(0.1), BoundaryState())
        assert result == Regime.CRITICAL

    def test_evaluate_degraded(self) -> None:
        rm = RegimeManager()
        result = rm.evaluate(_state(0.5), BoundaryState())
        assert result == Regime.DEGRADED

    def test_evaluate_nominal_high_r(self) -> None:
        rm = RegimeManager()
        result = rm.evaluate(_state(0.9), BoundaryState())
        assert result == Regime.NOMINAL

    def test_evaluate_hard_violation_critical(self) -> None:
        rm = RegimeManager()
        bs = BoundaryState(hard_violations=["test"])
        result = rm.evaluate(_state(0.9), bs)
        assert result == Regime.CRITICAL

    def test_transition_cooldown(self) -> None:
        rm = RegimeManager(cooldown_steps=5)
        rm.transition(Regime.DEGRADED)
        result = rm.transition(Regime.NOMINAL)
        assert result == Regime.DEGRADED  # in cooldown

    def test_force_transition_bypasses_cooldown(self) -> None:
        rm = RegimeManager(cooldown_steps=100)
        rm.transition(Regime.DEGRADED)
        result = rm.force_transition(Regime.NOMINAL)
        assert result == Regime.NOMINAL

    def test_transition_history(self) -> None:
        rm = RegimeManager(cooldown_steps=0)
        rm.transition(Regime.DEGRADED)
        rm.transition(Regime.CRITICAL)
        assert len(rm.transition_history) == 2

    def test_event_bus_emission(self) -> None:
        bus = EventBus()
        rm = RegimeManager(cooldown_steps=0, event_bus=bus)
        rm.transition(Regime.CRITICAL)
        assert bus.count == 1
        assert "nominal->critical" in bus.history[0].detail

    def test_same_regime_no_transition(self) -> None:
        rm = RegimeManager()
        result = rm.transition(Regime.NOMINAL)
        assert result == Regime.NOMINAL
        assert len(rm.transition_history) == 0
