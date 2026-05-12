# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — upde engine constructor validation tests

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.delay import DelayBuffer, DelayedEngine
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.inertial import InertialKuramotoEngine
from scpn_phase_orchestrator.upde.reduction import OttAntonsenReduction
from scpn_phase_orchestrator.upde.sheaf_engine import SheafUPDEEngine
from scpn_phase_orchestrator.upde.simplicial import SimplicialEngine
from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine
from scpn_phase_orchestrator.upde.swarmalator import SwarmalatorEngine

_HAS_JAX = importlib.util.find_spec("jax") is not None


class TestUPDEEngineValidation:
    @pytest.mark.parametrize("n_oscillators", [False, 0, -1, 1.5, "4"])
    def test_rejects_invalid_oscillator_count(self, n_oscillators: Any) -> None:
        with pytest.raises(ValueError, match="n_oscillators must be >= 1"):
            UPDEEngine(n_oscillators=n_oscillators, dt=0.01)

    @pytest.mark.parametrize(
        "dt",
        [False, 0.0, -0.01, float("nan"), float("inf"), "0.01"],
    )
    def test_rejects_invalid_dt(self, dt: Any) -> None:
        with pytest.raises(ValueError, match="dt must be positive"):
            UPDEEngine(n_oscillators=4, dt=dt)

    @pytest.mark.parametrize("atol", [False, 0.0, -1e-6, float("nan"), "1e-6"])
    def test_rejects_invalid_atol(self, atol: Any) -> None:
        with pytest.raises(ValueError, match="atol must be positive"):
            UPDEEngine(n_oscillators=4, dt=0.01, atol=atol)

    @pytest.mark.parametrize("rtol", [False, 0.0, -1e-3, float("inf"), "1e-3"])
    def test_rejects_invalid_rtol(self, rtol: Any) -> None:
        with pytest.raises(ValueError, match="rtol must be positive"):
            UPDEEngine(n_oscillators=4, dt=0.01, rtol=rtol)

    @pytest.mark.parametrize("n_steps", [False, 0, -1, 1.5, "10"])
    def test_run_rejects_invalid_step_count(self, n_steps: Any) -> None:
        engine = UPDEEngine(n_oscillators=4, dt=0.01)
        phases = np.zeros(4, dtype=np.float64)
        omegas = np.ones(4, dtype=np.float64)
        knm = np.zeros((4, 4), dtype=np.float64)
        alpha = np.zeros((4, 4), dtype=np.float64)

        with pytest.raises(ValueError, match="n_steps must be >= 1"):
            engine.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=n_steps)

    def test_normalises_accepted_numpy_scalars(self) -> None:
        engine = UPDEEngine(
            n_oscillators=np.int64(4),
            dt=np.float64(0.01),
            atol=np.float64(1e-6),
            rtol=np.float64(1e-3),
        )

        assert engine._n == 4
        assert pytest.approx(0.01) == engine._dt
        assert pytest.approx(1e-6) == engine._atol
        assert pytest.approx(1e-3) == engine._rtol


class TestStuartLandauEngineValidation:
    @pytest.mark.parametrize("n_oscillators", [False, 0, -1, 1.5, "4"])
    def test_rejects_invalid_oscillator_count(self, n_oscillators: Any) -> None:
        with pytest.raises(ValueError, match="n_oscillators must be >= 1"):
            StuartLandauEngine(n_oscillators=n_oscillators, dt=0.01)

    @pytest.mark.parametrize(
        "dt",
        [False, 0.0, -0.01, float("nan"), float("inf"), "0.01"],
    )
    def test_rejects_invalid_dt(self, dt: Any) -> None:
        with pytest.raises(ValueError, match="dt must be positive"):
            StuartLandauEngine(n_oscillators=4, dt=dt)

    @pytest.mark.parametrize("atol", [False, 0.0, -1e-6, float("nan"), "1e-6"])
    def test_rejects_invalid_atol(self, atol: Any) -> None:
        with pytest.raises(ValueError, match="atol must be positive"):
            StuartLandauEngine(n_oscillators=4, dt=0.01, atol=atol)

    @pytest.mark.parametrize("rtol", [False, 0.0, -1e-3, float("inf"), "1e-3"])
    def test_rejects_invalid_rtol(self, rtol: Any) -> None:
        with pytest.raises(ValueError, match="rtol must be positive"):
            StuartLandauEngine(n_oscillators=4, dt=0.01, rtol=rtol)

    def test_normalises_accepted_numpy_scalars(self) -> None:
        engine = StuartLandauEngine(
            n_oscillators=np.int64(4),
            dt=np.float64(0.01),
            atol=np.float64(1e-6),
            rtol=np.float64(1e-3),
        )

        assert engine._n == 4
        assert pytest.approx(0.01) == engine._dt
        assert pytest.approx(1e-6) == engine._atol
        assert pytest.approx(1e-3) == engine._rtol


class TestSheafUPDEEngineValidation:
    @pytest.mark.parametrize("n_oscillators", [False, 0, -1, 1.5, "4"])
    def test_rejects_invalid_oscillator_count(self, n_oscillators: Any) -> None:
        with pytest.raises(ValueError, match="n_oscillators must be >= 1"):
            SheafUPDEEngine(n_oscillators=n_oscillators, d_dimensions=2, dt=0.01)

    @pytest.mark.parametrize("d_dimensions", [False, 0, -1, 1.5, "2"])
    def test_rejects_invalid_dimension_count(self, d_dimensions: Any) -> None:
        with pytest.raises(ValueError, match="d_dimensions must be >= 1"):
            SheafUPDEEngine(n_oscillators=4, d_dimensions=d_dimensions, dt=0.01)

    @pytest.mark.parametrize(
        "dt",
        [False, 0.0, -0.01, float("nan"), float("inf"), "0.01"],
    )
    def test_rejects_invalid_dt(self, dt: Any) -> None:
        with pytest.raises(ValueError, match="dt must be positive"):
            SheafUPDEEngine(n_oscillators=4, d_dimensions=2, dt=dt)

    @pytest.mark.parametrize("atol", [False, 0.0, -1e-6, float("nan"), "1e-6"])
    def test_rejects_invalid_atol(self, atol: Any) -> None:
        with pytest.raises(ValueError, match="atol must be positive"):
            SheafUPDEEngine(n_oscillators=4, d_dimensions=2, dt=0.01, atol=atol)

    @pytest.mark.parametrize("rtol", [False, 0.0, -1e-3, float("inf"), "1e-3"])
    def test_rejects_invalid_rtol(self, rtol: Any) -> None:
        with pytest.raises(ValueError, match="rtol must be positive"):
            SheafUPDEEngine(n_oscillators=4, d_dimensions=2, dt=0.01, rtol=rtol)

    @pytest.mark.parametrize("n_steps", [False, -1, 1.5, "10"])
    def test_run_rejects_invalid_step_count(self, n_steps: Any) -> None:
        engine = SheafUPDEEngine(n_oscillators=4, d_dimensions=2, dt=0.01)
        phases = np.zeros((4, 2), dtype=np.float64)
        omegas = np.ones((4, 2), dtype=np.float64)
        restriction_maps = np.zeros((4, 4, 2, 2), dtype=np.float64)
        psi = np.zeros(2, dtype=np.float64)

        with pytest.raises(ValueError, match="n_steps must be >= 0"):
            engine.run(phases, omegas, restriction_maps, 0.0, psi, n_steps=n_steps)

    def test_run_accepts_zero_step_copy(self) -> None:
        engine = SheafUPDEEngine(n_oscillators=4, d_dimensions=2, dt=0.01)
        phases = np.arange(8, dtype=np.float64).reshape(4, 2)
        omegas = np.ones((4, 2), dtype=np.float64)
        restriction_maps = np.zeros((4, 4, 2, 2), dtype=np.float64)
        psi = np.zeros(2, dtype=np.float64)

        out = engine.run(phases, omegas, restriction_maps, 0.0, psi, n_steps=0)

        np.testing.assert_array_equal(out, phases)
        assert not np.shares_memory(out, phases)

    def test_normalises_accepted_numpy_scalars(self) -> None:
        engine = SheafUPDEEngine(
            n_oscillators=np.int64(4),
            d_dimensions=np.int64(2),
            dt=np.float64(0.01),
            atol=np.float64(1e-6),
            rtol=np.float64(1e-3),
        )

        assert engine._n == 4
        assert engine._d == 2
        assert pytest.approx(0.01) == engine._dt
        assert pytest.approx(1e-6) == engine._atol
        assert pytest.approx(1e-3) == engine._rtol


class TestSwarmalatorEngineValidation:
    def test_rejects_zero_agents(self) -> None:
        with pytest.raises(ValueError, match="n_agents must be >= 1"):
            SwarmalatorEngine(n_agents=0)

    @pytest.mark.parametrize("n_agents", [False, -1, 1.5, "4"])
    def test_rejects_invalid_agent_count(self, n_agents: Any) -> None:
        with pytest.raises(ValueError, match="n_agents must be >= 1"):
            SwarmalatorEngine(n_agents=n_agents)

    def test_rejects_zero_dimension(self) -> None:
        with pytest.raises(ValueError, match="dim must be >= 1"):
            SwarmalatorEngine(n_agents=4, dim=0)

    @pytest.mark.parametrize("dim", [False, -1, 1.5, "2"])
    def test_rejects_invalid_dimension(self, dim: Any) -> None:
        with pytest.raises(ValueError, match="dim must be >= 1"):
            SwarmalatorEngine(n_agents=4, dim=dim)

    def test_rejects_zero_dt(self) -> None:
        with pytest.raises(ValueError, match="dt must be positive"):
            SwarmalatorEngine(n_agents=4, dt=0.0)

    def test_rejects_negative_dt(self) -> None:
        with pytest.raises(ValueError, match="dt must be positive"):
            SwarmalatorEngine(n_agents=4, dt=-0.01)

    @pytest.mark.parametrize("dt", [False, float("nan"), float("inf"), "0.01"])
    def test_rejects_invalid_dt(self, dt: Any) -> None:
        with pytest.raises(ValueError, match="dt must be positive"):
            SwarmalatorEngine(n_agents=4, dt=dt)

    @pytest.mark.parametrize("n_steps", [False, 0, -1, 1.5, "10"])
    def test_run_rejects_invalid_step_count(self, n_steps: Any) -> None:
        engine = SwarmalatorEngine(n_agents=4, dim=2, dt=0.01)
        pos = np.zeros((4, 2), dtype=np.float64)
        phases = np.zeros(4, dtype=np.float64)
        omegas = np.ones(4, dtype=np.float64)

        with pytest.raises(ValueError, match="n_steps must be >= 1"):
            engine.run(pos, phases, omegas, n_steps=n_steps)

    def test_normalises_accepted_numpy_scalars(self) -> None:
        engine = SwarmalatorEngine(
            n_agents=np.int64(4),
            dim=np.int64(2),
            dt=np.float64(0.01),
        )

        assert engine._n == 4
        assert engine._dim == 2
        assert pytest.approx(0.01) == engine._dt

    @pytest.mark.parametrize(
        ("field", "bad_value", "match"),
        [
            ("pos", np.zeros((4, 3), dtype=np.float64), "pos shape"),
            ("phases", np.zeros((5,), dtype=np.float64), "phases shape"),
            ("omegas", np.zeros((3,), dtype=np.float64), "omegas shape"),
        ],
    )
    def test_step_rejects_state_shape_mismatch(
        self,
        field: str,
        bad_value: np.ndarray,
        match: str,
    ) -> None:
        engine = SwarmalatorEngine(n_agents=4, dim=2, dt=0.01)
        pos = np.zeros((4, 2), dtype=np.float64)
        phases = np.zeros(4, dtype=np.float64)
        omegas = np.ones(4, dtype=np.float64)
        values = {"pos": pos, "phases": phases, "omegas": omegas}
        values[field] = bad_value

        with pytest.raises(ValueError, match=match):
            engine.step(values["pos"], values["phases"], values["omegas"])

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("pos", np.inf),
            ("phases", np.nan),
            ("omegas", np.inf),
        ],
    )
    def test_step_rejects_non_finite_state_arrays(
        self,
        field: str,
        bad_value: float,
    ) -> None:
        engine = SwarmalatorEngine(n_agents=4, dim=2, dt=0.01)
        pos = np.zeros((4, 2), dtype=np.float64)
        phases = np.zeros(4, dtype=np.float64)
        omegas = np.ones(4, dtype=np.float64)
        if field == "pos":
            pos[0, 0] = bad_value
        elif field == "phases":
            phases[0] = bad_value
        else:
            omegas[0] = bad_value

        with pytest.raises(ValueError, match=field):
            engine.step(pos, phases, omegas)

    @pytest.mark.parametrize("coefficient", ["a", "b", "j", "k"])
    @pytest.mark.parametrize("bad_value", [False, np.nan, np.inf, "1.0"])
    def test_step_rejects_invalid_coefficients(
        self,
        coefficient: str,
        bad_value: Any,
    ) -> None:
        engine = SwarmalatorEngine(n_agents=4, dim=2, dt=0.01)
        pos = np.zeros((4, 2), dtype=np.float64)
        phases = np.zeros(4, dtype=np.float64)
        omegas = np.ones(4, dtype=np.float64)
        kwargs = {coefficient: bad_value}

        with pytest.raises(ValueError, match=coefficient):
            engine.step(pos, phases, omegas, **kwargs)


class TestSimplicialEngineValidation:
    def test_rejects_zero_oscillators(self) -> None:
        with pytest.raises(ValueError, match="n_oscillators must be >= 1"):
            SimplicialEngine(n_oscillators=0, dt=0.01)

    @pytest.mark.parametrize("n_oscillators", [False, -1, 1.5, "4"])
    def test_rejects_invalid_oscillator_count(self, n_oscillators: Any) -> None:
        with pytest.raises(ValueError, match="n_oscillators must be >= 1"):
            SimplicialEngine(n_oscillators=n_oscillators, dt=0.01)

    def test_rejects_zero_dt(self) -> None:
        with pytest.raises(ValueError, match="dt must be positive"):
            SimplicialEngine(n_oscillators=4, dt=0.0)

    @pytest.mark.parametrize("dt", [False, -0.01, float("nan"), float("inf"), "0.01"])
    def test_rejects_invalid_dt(self, dt: Any) -> None:
        with pytest.raises(ValueError, match="dt must be positive"):
            SimplicialEngine(n_oscillators=4, dt=dt)

    def test_rejects_negative_sigma2(self) -> None:
        with pytest.raises(ValueError, match="sigma2 must be non-negative finite real"):
            SimplicialEngine(n_oscillators=4, dt=0.01, sigma2=-0.1)

    @pytest.mark.parametrize(
        "sigma2",
        [False, float("nan"), float("inf"), "0.1"],
    )
    def test_rejects_invalid_sigma2(self, sigma2: Any) -> None:
        with pytest.raises(ValueError, match="sigma2 must be non-negative"):
            SimplicialEngine(n_oscillators=4, dt=0.01, sigma2=sigma2)

    @pytest.mark.parametrize("sigma2", [False, -0.1, float("nan"), "0.1"])
    def test_sigma2_setter_rejects_invalid_values(self, sigma2: Any) -> None:
        engine = SimplicialEngine(n_oscillators=4, dt=0.01)

        with pytest.raises(ValueError, match="sigma2 must be non-negative"):
            engine.sigma2 = sigma2

    @pytest.mark.parametrize("n_steps", [False, 0, -1, 1.5, "10"])
    def test_run_rejects_invalid_step_count(self, n_steps: Any) -> None:
        engine = SimplicialEngine(n_oscillators=4, dt=0.01)
        phases = np.zeros(4, dtype=np.float64)
        omegas = np.ones(4, dtype=np.float64)
        knm = np.zeros((4, 4), dtype=np.float64)
        alpha = np.zeros((4, 4), dtype=np.float64)

        with pytest.raises(ValueError, match="n_steps must be >= 1"):
            engine.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=n_steps)

    def test_normalises_accepted_numpy_scalars(self) -> None:
        engine = SimplicialEngine(
            n_oscillators=np.int64(4),
            dt=np.float64(0.01),
            sigma2=np.float64(0.1),
        )

        assert engine._n == 4
        assert pytest.approx(0.01) == engine._dt
        assert pytest.approx(0.1) == engine.sigma2

    @pytest.mark.parametrize(
        ("field", "bad_value", "match"),
        [
            ("phases", np.zeros((5,), dtype=np.float64), "phases shape"),
            ("omegas", np.zeros((3,), dtype=np.float64), "omegas shape"),
            ("knm", np.zeros((4, 3), dtype=np.float64), "knm shape"),
            ("alpha", np.zeros((3, 4), dtype=np.float64), "alpha shape"),
        ],
    )
    def test_run_rejects_state_shape_mismatch(
        self,
        field: str,
        bad_value: np.ndarray,
        match: str,
    ) -> None:
        engine = SimplicialEngine(n_oscillators=4, dt=0.01)
        values = {
            "phases": np.zeros(4, dtype=np.float64),
            "omegas": np.ones(4, dtype=np.float64),
            "knm": np.zeros((4, 4), dtype=np.float64),
            "alpha": np.zeros((4, 4), dtype=np.float64),
        }
        values[field] = bad_value

        with pytest.raises(ValueError, match=match):
            engine.run(
                values["phases"],
                values["omegas"],
                values["knm"],
                0.0,
                0.0,
                values["alpha"],
                n_steps=1,
            )

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("phases", np.nan),
            ("omegas", np.inf),
            ("knm", np.nan),
            ("alpha", np.inf),
        ],
    )
    def test_run_rejects_non_finite_state_arrays(
        self,
        field: str,
        bad_value: float,
    ) -> None:
        engine = SimplicialEngine(n_oscillators=4, dt=0.01)
        phases = np.zeros(4, dtype=np.float64)
        omegas = np.ones(4, dtype=np.float64)
        knm = np.zeros((4, 4), dtype=np.float64)
        alpha = np.zeros((4, 4), dtype=np.float64)
        if field in {"knm", "alpha"}:
            locals()[field][0, 1] = bad_value
        else:
            locals()[field][0] = bad_value

        with pytest.raises(ValueError, match=field):
            engine.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=1)

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("zeta", False),
            ("zeta", np.nan),
            ("zeta", np.inf),
            ("zeta", "1.0"),
            ("psi", False),
            ("psi", np.nan),
            ("psi", np.inf),
            ("psi", "0.0"),
        ],
    )
    def test_run_rejects_invalid_scalar_inputs(
        self,
        field: str,
        bad_value: Any,
    ) -> None:
        engine = SimplicialEngine(n_oscillators=4, dt=0.01)
        kwargs = {"zeta": 0.0, "psi": 0.0}
        kwargs[field] = bad_value

        with pytest.raises(ValueError, match=field):
            engine.run(
                np.zeros(4, dtype=np.float64),
                np.ones(4, dtype=np.float64),
                np.zeros((4, 4), dtype=np.float64),
                kwargs["zeta"],
                kwargs["psi"],
                np.zeros((4, 4), dtype=np.float64),
                n_steps=1,
            )


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

    @pytest.mark.parametrize(
        ("phases", "match"),
        [
            (np.zeros((3, 1), dtype=np.float64), "phases shape"),
            (np.array([0.0, np.nan, 0.2], dtype=np.float64), "phases"),
        ],
    )
    def test_push_rejects_invalid_phase_snapshot(
        self,
        phases: np.ndarray,
        match: str,
    ) -> None:
        buffer = DelayBuffer(n_oscillators=3, max_delay_steps=5)

        with pytest.raises(ValueError, match=match):
            buffer.push(phases)


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

    @pytest.mark.parametrize("n_steps", [False, 0, -1, 1.5, "10"])
    def test_run_rejects_invalid_step_count(self, n_steps: Any) -> None:
        engine = DelayedEngine(n_oscillators=4, dt=0.01, delay_steps=2)

        with pytest.raises(ValueError, match="n_steps must be a positive integer"):
            engine.run(
                np.zeros(4, dtype=np.float64),
                np.ones(4, dtype=np.float64),
                np.zeros((4, 4), dtype=np.float64),
                n_steps=n_steps,
            )

    @pytest.mark.parametrize(
        ("field", "bad_value", "match"),
        [
            ("phases", np.zeros((5,), dtype=np.float64), "phases shape"),
            ("omegas", np.zeros((3,), dtype=np.float64), "omegas shape"),
            ("knm", np.zeros((4, 3), dtype=np.float64), "knm shape"),
            ("alpha", np.zeros((3, 4), dtype=np.float64), "alpha shape"),
        ],
    )
    def test_step_rejects_state_shape_mismatch(
        self,
        field: str,
        bad_value: np.ndarray,
        match: str,
    ) -> None:
        engine = DelayedEngine(n_oscillators=4, dt=0.01, delay_steps=2)
        values = {
            "phases": np.zeros(4, dtype=np.float64),
            "omegas": np.ones(4, dtype=np.float64),
            "knm": np.zeros((4, 4), dtype=np.float64),
            "alpha": np.zeros((4, 4), dtype=np.float64),
        }
        values[field] = bad_value

        with pytest.raises(ValueError, match=match):
            engine.step(
                values["phases"],
                values["omegas"],
                values["knm"],
                0.0,
                0.0,
                values["alpha"],
            )

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("phases", np.nan),
            ("omegas", np.inf),
            ("knm", np.nan),
            ("alpha", np.inf),
        ],
    )
    def test_step_rejects_non_finite_state_arrays(
        self,
        field: str,
        bad_value: float,
    ) -> None:
        engine = DelayedEngine(n_oscillators=4, dt=0.01, delay_steps=2)
        phases = np.zeros(4, dtype=np.float64)
        omegas = np.ones(4, dtype=np.float64)
        knm = np.zeros((4, 4), dtype=np.float64)
        alpha = np.zeros((4, 4), dtype=np.float64)
        if field in {"knm", "alpha"}:
            locals()[field][0, 1] = bad_value
        else:
            locals()[field][0] = bad_value

        with pytest.raises(ValueError, match=field):
            engine.step(phases, omegas, knm, 0.0, 0.0, alpha)

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("zeta", False),
            ("zeta", np.nan),
            ("zeta", np.inf),
            ("zeta", "1.0"),
            ("psi", False),
            ("psi", np.nan),
            ("psi", np.inf),
            ("psi", "0.0"),
        ],
    )
    def test_step_rejects_invalid_scalar_inputs(
        self,
        field: str,
        bad_value: Any,
    ) -> None:
        engine = DelayedEngine(n_oscillators=4, dt=0.01, delay_steps=2)
        kwargs = {"zeta": 0.0, "psi": 0.0}
        kwargs[field] = bad_value

        with pytest.raises(ValueError, match=field):
            engine.step(
                np.zeros(4, dtype=np.float64),
                np.ones(4, dtype=np.float64),
                np.zeros((4, 4), dtype=np.float64),
                kwargs["zeta"],
                kwargs["psi"],
                np.zeros((4, 4), dtype=np.float64),
            )


class TestInertialEngineValidation:
    def test_rejects_zero_n(self) -> None:
        with pytest.raises(ValueError, match="n must be >= 1"):
            InertialKuramotoEngine(n=0)

    def test_rejects_zero_dt(self) -> None:
        with pytest.raises(ValueError, match="dt must be positive"):
            InertialKuramotoEngine(n=4, dt=0.0)

    @pytest.mark.parametrize(
        ("field", "bad_value", "match"),
        [
            ("theta", np.zeros((5,), dtype=np.float64), "theta shape"),
            ("omega_dot", np.zeros((3,), dtype=np.float64), "omega_dot shape"),
            ("power", np.zeros((3,), dtype=np.float64), "power shape"),
            ("knm", np.zeros((4, 3), dtype=np.float64), "knm shape"),
            ("inertia", np.ones((3,), dtype=np.float64), "inertia shape"),
            ("damping", np.ones((5,), dtype=np.float64), "damping shape"),
        ],
    )
    def test_step_rejects_state_shape_mismatch(
        self,
        field: str,
        bad_value: np.ndarray,
        match: str,
    ) -> None:
        engine = InertialKuramotoEngine(n=4, dt=0.01)
        values = {
            "theta": np.zeros(4, dtype=np.float64),
            "omega_dot": np.zeros(4, dtype=np.float64),
            "power": np.zeros(4, dtype=np.float64),
            "knm": np.zeros((4, 4), dtype=np.float64),
            "inertia": np.ones(4, dtype=np.float64),
            "damping": np.zeros(4, dtype=np.float64),
        }
        values[field] = bad_value

        with pytest.raises(ValueError, match=match):
            engine.step(
                values["theta"],
                values["omega_dot"],
                values["power"],
                values["knm"],
                values["inertia"],
                values["damping"],
            )

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("theta", np.nan),
            ("omega_dot", np.inf),
            ("power", np.nan),
            ("knm", np.inf),
            ("inertia", np.nan),
            ("damping", np.inf),
        ],
    )
    def test_step_rejects_non_finite_state_arrays(
        self,
        field: str,
        bad_value: float,
    ) -> None:
        engine = InertialKuramotoEngine(n=4, dt=0.01)
        theta = np.zeros(4, dtype=np.float64)
        omega_dot = np.zeros(4, dtype=np.float64)
        power = np.zeros(4, dtype=np.float64)
        knm = np.zeros((4, 4), dtype=np.float64)
        inertia = np.ones(4, dtype=np.float64)
        damping = np.zeros(4, dtype=np.float64)
        if field == "knm":
            knm[0, 1] = bad_value
        else:
            locals()[field][0] = bad_value

        with pytest.raises(ValueError, match=field):
            engine.step(theta, omega_dot, power, knm, inertia, damping)

    @pytest.mark.parametrize("bad_inertia", [0.0, -1.0])
    def test_step_rejects_non_positive_inertia(self, bad_inertia: float) -> None:
        engine = InertialKuramotoEngine(n=4, dt=0.01)
        inertia = np.ones(4, dtype=np.float64)
        inertia[0] = bad_inertia

        with pytest.raises(ValueError, match="inertia"):
            engine.step(
                np.zeros(4, dtype=np.float64),
                np.zeros(4, dtype=np.float64),
                np.zeros(4, dtype=np.float64),
                np.zeros((4, 4), dtype=np.float64),
                inertia,
                np.zeros(4, dtype=np.float64),
            )

    def test_step_rejects_negative_damping(self) -> None:
        engine = InertialKuramotoEngine(n=4, dt=0.01)
        damping = np.zeros(4, dtype=np.float64)
        damping[0] = -0.1

        with pytest.raises(ValueError, match="damping"):
            engine.step(
                np.zeros(4, dtype=np.float64),
                np.zeros(4, dtype=np.float64),
                np.zeros(4, dtype=np.float64),
                np.zeros((4, 4), dtype=np.float64),
                np.ones(4, dtype=np.float64),
                damping,
            )

    @pytest.mark.parametrize("n_steps", [False, 0, -1, 1.5, "10"])
    def test_run_rejects_invalid_step_count(self, n_steps: Any) -> None:
        engine = InertialKuramotoEngine(n=4, dt=0.01)

        with pytest.raises(ValueError, match="n_steps must be >= 1"):
            engine.run(
                np.zeros(4, dtype=np.float64),
                np.zeros(4, dtype=np.float64),
                np.zeros(4, dtype=np.float64),
                np.zeros((4, 4), dtype=np.float64),
                np.ones(4, dtype=np.float64),
                np.zeros(4, dtype=np.float64),
                n_steps=n_steps,
            )


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

        with pytest.raises(ValueError, match="n"):
            JaxUPDEEngine(n=0)

    @pytest.mark.parametrize("n", [True, 1.5, "4"])
    def test_rejects_invalid_n_type(self, n: object) -> None:
        from scpn_phase_orchestrator.upde.jax_engine import JaxUPDEEngine

        with pytest.raises(ValueError, match="n"):
            JaxUPDEEngine(n=n)

    def test_rejects_zero_dt(self) -> None:
        from scpn_phase_orchestrator.upde.jax_engine import JaxUPDEEngine

        with pytest.raises(ValueError, match="dt"):
            JaxUPDEEngine(n=4, dt=0.0)

    @pytest.mark.parametrize("dt", [False, -0.01, float("nan"), float("inf"), "0.01"])
    def test_rejects_invalid_dt(self, dt: object) -> None:
        from scpn_phase_orchestrator.upde.jax_engine import JaxUPDEEngine

        with pytest.raises(ValueError, match="dt"):
            JaxUPDEEngine(n=4, dt=dt)

    def test_rejects_unknown_method(self) -> None:
        from scpn_phase_orchestrator.upde.jax_engine import JaxUPDEEngine

        with pytest.raises(ValueError, match="unsupported method"):
            JaxUPDEEngine(n=4, method="adams-bashforth")

    def test_rejects_non_string_method(self) -> None:
        from scpn_phase_orchestrator.upde.jax_engine import JaxUPDEEngine

        with pytest.raises(ValueError, match="unsupported method"):
            JaxUPDEEngine(n=4, method=True)

    @pytest.mark.parametrize("n", [False, 0, 1.5, "4"])
    def test_stuart_landau_rejects_invalid_n(self, n: object) -> None:
        from scpn_phase_orchestrator.upde.jax_engine import JaxStuartLandauEngine

        with pytest.raises(ValueError, match="n"):
            JaxStuartLandauEngine(n=n)

    @pytest.mark.parametrize(
        "dt", [False, 0.0, -0.01, float("nan"), float("inf"), "0.01"]
    )
    def test_stuart_landau_rejects_invalid_dt(self, dt: object) -> None:
        from scpn_phase_orchestrator.upde.jax_engine import JaxStuartLandauEngine

        with pytest.raises(ValueError, match="dt"):
            JaxStuartLandauEngine(n=4, dt=dt)


# Pipeline wiring: engine constructors guard the boundary between user
# configuration (YAML spec, CLI flag, API call) and numpy/rust state. A
# silent misconfiguration is a class of bug the supervisor cannot detect
# because the numbers still "compute" — these validators convert the
# silent failure into an early, traceable ValueError at load time.
