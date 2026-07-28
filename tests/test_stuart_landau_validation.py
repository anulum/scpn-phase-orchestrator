# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Stuart-Landau validation contracts

"""Input-validation and finite-output contracts for StuartLandauEngine."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.upde import stuart_landau as sl_module


class TestStuartLandauInputValidation:
    """Verify that StuartLandauEngine rejects every type of invalid input
    with the correct error message and field name."""

    @pytest.fixture()
    def engine(self):
        from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

        return StuartLandauEngine(2, dt=0.01)

    @pytest.mark.parametrize("n_oscillators", [False, 0, 1.5])
    def test_constructor_rejects_invalid_oscillator_count(
        self,
        n_oscillators: object,
    ) -> None:
        from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

        with pytest.raises(ValueError, match="n_oscillators"):
            StuartLandauEngine(n_oscillators, dt=0.01)

    @pytest.fixture()
    def valid_args(self):
        return {
            "state": np.array([0.1, 0.2, 0.7, 0.8]),
            "omegas": np.array([1.0, 1.0]),
            "mu": np.array([0.5, 0.5]),
            "knm": np.zeros((2, 2)),
            "knm_r": np.zeros((2, 2)),
            "zeta": 0.0,
            "psi": 0.0,
            "alpha": np.zeros((2, 2)),
        }

    def test_array_guard_rejects_failed_array_protocol(self) -> None:
        class FailedArray:
            def __array__(self, dtype=None, copy=None):
                raise TypeError("unavailable array payload")

        with pytest.raises(ValueError, match="probe must be a numeric array"):
            sl_module._as_real_numeric_array(FailedArray(), name="probe")

    def test_array_guard_rejects_float_conversion_overflow(self) -> None:
        huge_integer = np.array([10**1000], dtype=object)

        with pytest.raises(ValueError, match="probe must be a numeric array"):
            sl_module._as_real_numeric_array(huge_integer, name="probe")

    @pytest.mark.parametrize(
        "field",
        ["state", "omegas", "mu", "knm", "knm_r", "alpha"],
    )
    def test_numeric_string_arrays_fail_before_solver(
        self,
        engine,
        valid_args,
        field: str,
    ) -> None:
        args = dict(valid_args)
        args[field] = np.asarray(args[field]).astype(str)

        with pytest.raises(ValueError, match=f"{field} must be numeric"):
            engine.step(**args)

    @pytest.mark.parametrize(
        "field",
        ["state", "omegas", "mu", "knm", "knm_r", "alpha"],
    )
    def test_complex_arrays_fail_before_solver(
        self,
        engine,
        valid_args,
        field: str,
    ) -> None:
        args = dict(valid_args)
        value = np.asarray(args[field], dtype=object)
        value.flat[0] = complex(float(value.flat[0]), 1.0)
        args[field] = value

        with pytest.raises(ValueError, match=f"{field} must be real-valued"):
            engine.step(**args)

    def test_boolean_array_fails_before_solver(self, engine, valid_args) -> None:
        args = dict(valid_args)
        args["omegas"] = np.array([True, False])

        with pytest.raises(ValueError, match="omegas must be real-valued"):
            engine.step(**args)

    def test_real_numeric_object_arrays_remain_supported(
        self,
        engine,
        valid_args,
    ) -> None:
        args = {
            name: np.asarray(value, dtype=object)
            if isinstance(value, np.ndarray)
            else value
            for name, value in valid_args.items()
        }

        result = engine.step(**args)

        assert result.dtype == np.float64
        assert result.shape == (4,)

    @pytest.mark.parametrize(
        "method_name",
        ["compute_order_parameter", "compute_mean_amplitude"],
    )
    @pytest.mark.parametrize(
        ("state", "match"),
        [
            (np.array(["0.1", "0.2", "0.7", "0.8"]), "numeric"),
            (
                np.array([0.1 + 1.0j, 0.2, 0.7, 0.8]),
                "real-valued",
            ),
        ],
    )
    def test_observables_reject_coercive_state_aliases(
        self,
        engine,
        method_name: str,
        state: np.ndarray,
        match: str,
    ) -> None:
        method = getattr(engine, method_name)

        with pytest.raises(ValueError, match=match):
            method(state)

    @pytest.mark.parametrize(
        "field,bad_value,error_pattern",
        [
            ("state", np.array([0.1, 0.2, float("nan"), 0.8]), "state contains NaN"),
            ("omegas", np.array([float("nan"), 1.0]), "omegas contain NaN"),
            ("mu", np.array([float("inf"), 0.5]), "mu contains NaN"),
            ("knm", np.array([[0.0, float("nan")], [0.0, 0.0]]), "knm contains NaN"),
            (
                "knm_r",
                np.array([[0.0, float("inf")], [0.0, 0.0]]),
                "knm_r contains NaN",
            ),
            (
                "alpha",
                np.array([[0.0, float("nan")], [0.0, 0.0]]),
                "alpha contains NaN",
            ),
        ],
    )
    def test_nan_in_field_raises_valueerror(
        self, engine, valid_args, field, bad_value, error_pattern
    ):
        """Each numeric input field must be validated for NaN/Inf."""
        args = dict(valid_args)
        args[field] = bad_value
        with pytest.raises(ValueError, match=error_pattern):
            engine.step(**args)

    def test_epsilon_nan_raises(self, engine, valid_args):
        """Non-finite epsilon must be rejected."""
        with pytest.raises(ValueError, match="epsilon must be finite"):
            engine.step(**valid_args, epsilon=float("nan"))

    def test_epsilon_inf_raises(self, engine, valid_args):
        """Infinite epsilon must also be rejected."""
        with pytest.raises(ValueError, match="epsilon must be finite"):
            engine.step(**valid_args, epsilon=float("inf"))

    def test_boolean_epsilon_raises(self, engine, valid_args):
        with pytest.raises(ValueError, match="epsilon must be finite real"):
            engine.step(**valid_args, epsilon=False)

    @pytest.mark.parametrize("field", ["zeta", "psi"])
    def test_boolean_phase_drive_raises(self, engine, valid_args, field: str):
        args = dict(valid_args)
        args[field] = False

        with pytest.raises(ValueError, match="zeta and psi must be finite"):
            engine.step(**args)

    def test_valid_inputs_produce_finite_output(self, engine, valid_args):
        """Valid inputs → finite state vector of correct size."""
        result = engine.step(**valid_args)
        assert result.shape == (4,), f"SL state should be 2*N=4, got {result.shape}"
        assert np.all(np.isfinite(result)), (
            f"Valid inputs should give finite output: {result}"
        )

    @pytest.mark.parametrize(
        ("result", "match"),
        [
            (["0.1", "0.2", "0.7", "0.8"], "numeric"),
            (np.array([0.1 + 1.0j, 0.2, 0.7, 0.8]), "real-valued"),
            ([0.1, 0.2], "expected"),
            ([0.1, 0.2, np.nan, 0.8], "NaN or Inf"),
        ],
    )
    def test_rust_state_output_fails_closed(
        self,
        engine,
        valid_args,
        result: object,
        match: str,
    ) -> None:
        class FakeRust:
            last_dt = 0.01

            def step(self, *_args):
                return result

        engine._use_rust = True
        engine._rust = FakeRust()

        with pytest.raises(ValueError, match=match):
            engine.step(**valid_args)

    @pytest.mark.parametrize("last_dt", [False, -0.01, np.nan, "0.01"])
    def test_rust_adaptive_timestep_fails_closed(
        self,
        engine,
        valid_args,
        last_dt: object,
    ) -> None:
        class FakeRust:
            def __init__(self, timestep: object) -> None:
                self.last_dt = timestep

            def step(self, *_args):
                return np.array([0.1, 0.2, 0.7, 0.8])

        engine._use_rust = True
        engine._rust = FakeRust(last_dt)

        with pytest.raises(ValueError, match="Rust last_dt"):
            engine.step(**valid_args)

    def test_rk45_stiff_parameters_stays_finite_without_warning(self):
        """SL RK45 with stiff finite dynamics must return finite results."""
        from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

        eng = StuartLandauEngine(4, dt=0.5, method="rk45", atol=1e-15, rtol=1e-15)
        state = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 1.0, 1.0, 1.0, 1.0])
        omegas = np.array([10.0, 20.0, 30.0, 40.0])
        mu = np.full(4, 2.0)
        knm = np.full((4, 4), 25.0)
        np.fill_diagonal(knm, 0.0)
        knm_r = np.full((4, 4), 10.0)
        np.fill_diagonal(knm_r, 0.0)
        alpha = np.zeros((4, 4))

        result = eng.step(state, omegas, mu, knm, knm_r, 0.0, 0.0, alpha)
        assert result.shape == (8,)
        assert np.all(np.isfinite(result))
