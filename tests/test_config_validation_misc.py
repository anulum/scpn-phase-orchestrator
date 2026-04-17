# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — constructor validation tests across subsystems

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.apps.queuewaves.alerter import WebhookAlerter
from scpn_phase_orchestrator.apps.queuewaves.collector import MetricBuffer
from scpn_phase_orchestrator.autotune.sindy import PhaseSINDy
from scpn_phase_orchestrator.monitor.lyapunov import LyapunovGuard
from scpn_phase_orchestrator.ssgf.pgbo import PGBO
from scpn_phase_orchestrator.supervisor.events import EventBus


class TestWebhookAlerterValidation:
    def test_rejects_negative_cooldown(self) -> None:
        with pytest.raises(ValueError, match="cooldown_seconds must be non-negative"):
            WebhookAlerter(sinks=[], cooldown_seconds=-1.0)

    def test_accepts_zero_cooldown(self) -> None:
        WebhookAlerter(sinks=[], cooldown_seconds=0.0)


class TestMetricBufferValidation:
    def test_rejects_zero_maxlen(self) -> None:
        with pytest.raises(ValueError, match="maxlen must be >= 1"):
            MetricBuffer(maxlen=0)

    def test_rejects_negative_maxlen(self) -> None:
        with pytest.raises(ValueError, match="maxlen must be >= 1"):
            MetricBuffer(maxlen=-5)


class TestSindyValidation:
    def test_rejects_negative_threshold(self) -> None:
        with pytest.raises(ValueError, match="threshold must be non-negative"):
            PhaseSINDy(threshold=-0.01)

    def test_rejects_zero_max_iter(self) -> None:
        with pytest.raises(ValueError, match="max_iter must be >= 1"):
            PhaseSINDy(max_iter=0)

    def test_rejects_negative_max_iter(self) -> None:
        with pytest.raises(ValueError, match="max_iter must be >= 1"):
            PhaseSINDy(max_iter=-3)


class TestLyapunovGuardValidation:
    def test_rejects_zero_basin_threshold(self) -> None:
        with pytest.raises(ValueError, match="basin_threshold must be positive"):
            LyapunovGuard(basin_threshold=0.0)

    def test_rejects_negative_basin_threshold(self) -> None:
        with pytest.raises(ValueError, match="basin_threshold must be positive"):
            LyapunovGuard(basin_threshold=-0.5)

    def test_default_basin_threshold_is_half_pi(self) -> None:
        m = LyapunovGuard()
        assert abs(m._basin_threshold - np.pi / 2.0) < 1e-12


class TestEventBusValidation:
    def test_rejects_zero_maxlen(self) -> None:
        with pytest.raises(ValueError, match="maxlen must be >= 1"):
            EventBus(maxlen=0)

    def test_rejects_negative_maxlen(self) -> None:
        with pytest.raises(ValueError, match="maxlen must be >= 1"):
            EventBus(maxlen=-10)


class TestPgboValidation:
    def test_rejects_empty_cost_weights(self) -> None:
        with pytest.raises(ValueError, match="at least one weight"):
            PGBO(cost_weights=())

    def test_rejects_negative_cost_weight(self) -> None:
        with pytest.raises(ValueError, match="cost_weights must be non-negative"):
            PGBO(cost_weights=(1.0, -0.2, 0.1))

    def test_accepts_zero_cost_weight(self) -> None:
        PGBO(cost_weights=(1.0, 0.0, 0.1))


# Pipeline wiring: these constructors sit on the boundary between binding
# spec / CLI flags / YAML config and the runtime subsystems. A ValueError
# at the boundary beats a confusing NaN or a zero-size queue 300 seconds
# into the simulation.
