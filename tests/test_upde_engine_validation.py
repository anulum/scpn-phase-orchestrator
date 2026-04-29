# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — upde engine constructor validation tests

from __future__ import annotations

import importlib.util

import pytest

from scpn_phase_orchestrator.upde.delay import DelayBuffer, DelayedEngine
from scpn_phase_orchestrator.upde.inertial import InertialKuramotoEngine
from scpn_phase_orchestrator.upde.reduction import OttAntonsenReduction
from scpn_phase_orchestrator.upde.simplicial import SimplicialEngine
from scpn_phase_orchestrator.upde.swarmalator import SwarmalatorEngine

_HAS_JAX = importlib.util.find_spec("jax") is not None


class TestSwarmalatorEngineValidation:
    def test_rejects_zero_agents(self) -> None:
        with pytest.raises(ValueError, match="n_agents must be >= 1"):
            SwarmalatorEngine(n_agents=0)

    def test_rejects_zero_dimension(self) -> None:
        with pytest.raises(ValueError, match="dim must be >= 1"):
            SwarmalatorEngine(n_agents=4, dim=0)

    def test_rejects_zero_dt(self) -> None:
        with pytest.raises(ValueError, match="dt must be positive"):
            SwarmalatorEngine(n_agents=4, dt=0.0)

    def test_rejects_negative_dt(self) -> None:
        with pytest.raises(ValueError, match="dt must be positive"):
            SwarmalatorEngine(n_agents=4, dt=-0.01)


class TestSimplicialEngineValidation:
    def test_rejects_zero_oscillators(self) -> None:
        with pytest.raises(ValueError, match="n_oscillators must be >= 1"):
            SimplicialEngine(n_oscillators=0, dt=0.01)

    def test_rejects_zero_dt(self) -> None:
        with pytest.raises(ValueError, match="dt must be positive"):
            SimplicialEngine(n_oscillators=4, dt=0.0)

    def test_rejects_negative_sigma2(self) -> None:
        with pytest.raises(ValueError, match="sigma2 must be non-negative"):
            SimplicialEngine(n_oscillators=4, dt=0.01, sigma2=-0.1)


class TestDelayBufferValidation:
    def test_rejects_zero_oscillators(self) -> None:
        with pytest.raises(
            ValueError, match="n_oscillators must be a positive integer"
        ):
            DelayBuffer(n_oscillators=0, max_delay_steps=5)

    def test_rejects_zero_max_delay(self) -> None:
        with pytest.raises(
            ValueError, match="max_delay_steps must be a positive integer"
        ):
            DelayBuffer(n_oscillators=3, max_delay_steps=0)


class TestDelayedEngineValidation:
    def test_rejects_zero_oscillators(self) -> None:
        with pytest.raises(
            ValueError, match="n_oscillators must be a positive integer"
        ):
            DelayedEngine(n_oscillators=0, dt=0.01)

    def test_rejects_zero_dt(self) -> None:
        with pytest.raises(ValueError, match="dt must be a finite positive real"):
            DelayedEngine(n_oscillators=4, dt=0.0)

    def test_rejects_zero_delay_steps(self) -> None:
        with pytest.raises(ValueError, match="delay_steps must be a positive integer"):
            DelayedEngine(n_oscillators=4, dt=0.01, delay_steps=0)


class TestInertialEngineValidation:
    def test_rejects_zero_n(self) -> None:
        with pytest.raises(ValueError, match="n must be >= 1"):
            InertialKuramotoEngine(n=0)

    def test_rejects_zero_dt(self) -> None:
        with pytest.raises(ValueError, match="dt must be positive"):
            InertialKuramotoEngine(n=4, dt=0.0)


class TestOttAntonsenReductionValidation:
    def test_rejects_negative_delta(self) -> None:
        with pytest.raises(ValueError, match="delta .* must be non-negative"):
            OttAntonsenReduction(omega_0=1.0, delta=-0.1, K=0.5)

    def test_rejects_zero_dt(self) -> None:
        with pytest.raises(ValueError, match="dt must be positive"):
            OttAntonsenReduction(omega_0=1.0, delta=0.1, K=0.5, dt=0.0)


@pytest.mark.skipif(not _HAS_JAX, reason="jax not installed")
class TestJaxEngineValidation:
    def test_rejects_zero_n(self) -> None:
        from scpn_phase_orchestrator.upde.jax_engine import JaxUPDEEngine

        with pytest.raises(ValueError, match="n must be >= 1"):
            JaxUPDEEngine(n=0)

    def test_rejects_zero_dt(self) -> None:
        from scpn_phase_orchestrator.upde.jax_engine import JaxUPDEEngine

        with pytest.raises(ValueError, match="dt must be positive"):
            JaxUPDEEngine(n=4, dt=0.0)

    def test_rejects_unknown_method(self) -> None:
        from scpn_phase_orchestrator.upde.jax_engine import JaxUPDEEngine

        with pytest.raises(ValueError, match="unsupported method"):
            JaxUPDEEngine(n=4, method="adams-bashforth")


# Pipeline wiring: engine constructors guard the boundary between user
# configuration (YAML spec, CLI flag, API call) and numpy/rust state. A
# silent misconfiguration is a class of bug the supervisor cannot detect
# because the numbers still "compute" — these validators convert the
# silent failure into an early, traceable ValueError at load time.
