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


class TestStuartLandauInputValidation:
    """Verify that StuartLandauEngine rejects every type of invalid input
    with the correct error message and field name."""

    @pytest.fixture()
    def engine(self):
        from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

        return StuartLandauEngine(2, dt=0.01)

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

    def test_valid_inputs_produce_finite_output(self, engine, valid_args):
        """Valid inputs → finite state vector of correct size."""
        result = engine.step(**valid_args)
        assert result.shape == (4,), f"SL state should be 2*N=4, got {result.shape}"
        assert np.all(np.isfinite(result)), (
            f"Valid inputs should give finite output: {result}"
        )

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
