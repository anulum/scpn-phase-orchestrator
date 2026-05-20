# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Basin stability tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.basin_stability import (
    BasinStabilityResult,
    basin_stability,
    multi_basin_stability,
)


class TestBasinStability:
    def test_identical_frequencies_high_stability(self):
        """Identical omegas + strong coupling → S_B ≈ 1."""
        N = 6
        omegas = np.zeros(N)
        knm = np.ones((N, N)) * 2.0
        np.fill_diagonal(knm, 0)
        result = basin_stability(
            omegas, knm, n_samples=20, n_transient=200, n_measure=50
        )
        assert isinstance(result, BasinStabilityResult)
        assert result.S_B > 0.5

    def test_zero_coupling_low_stability(self):
        """Zero coupling + spread frequencies → S_B ≈ 0."""
        N = 6
        rng = np.random.default_rng(42)
        omegas = rng.normal(0, 2.0, N)
        knm = np.zeros((N, N))
        result = basin_stability(
            omegas, knm, n_samples=20, n_transient=200, n_measure=50
        )
        assert result.S_B < 0.5

    def test_result_fields(self):
        N = 4
        omegas = np.zeros(N)
        knm = np.ones((N, N))
        np.fill_diagonal(knm, 0)
        result = basin_stability(
            omegas, knm, n_samples=10, n_transient=100, n_measure=50
        )
        assert result.n_samples == 10
        assert len(result.R_final) == 10
        assert 0 <= result.S_B <= 1.0
        assert result.R_threshold == 0.8

    def test_custom_threshold(self):
        N = 4
        omegas = np.zeros(N)
        knm = np.ones((N, N)) * 3.0
        np.fill_diagonal(knm, 0)
        result = basin_stability(
            omegas,
            knm,
            n_samples=10,
            n_transient=200,
            n_measure=50,
            R_threshold=0.5,
        )
        assert result.R_threshold == 0.5


class TestMultiBasinStability:
    def test_returns_dict(self):
        N = 4
        omegas = np.zeros(N)
        knm = np.ones((N, N)) * 2.0
        np.fill_diagonal(knm, 0)
        results = multi_basin_stability(
            omegas,
            knm,
            n_samples=10,
            n_transient=100,
            n_measure=50,
        )
        assert isinstance(results, dict)
        assert "R>=0.30" in results
        assert "R>=0.60" in results
        assert "R>=0.80" in results

    def test_monotonic_thresholds(self):
        """S_B at lower threshold >= S_B at higher threshold."""
        N = 6
        rng = np.random.default_rng(0)
        omegas = rng.normal(0, 0.5, N)
        knm = np.ones((N, N)) * 1.5
        np.fill_diagonal(knm, 0)
        results = multi_basin_stability(
            omegas,
            knm,
            n_samples=15,
            n_transient=200,
            n_measure=50,
        )
        assert results["R>=0.30"].S_B >= results["R>=0.80"].S_B


class TestBasinStabilityPipelineWiring:
    """Pipeline: basin_stability uses UPDEEngine internally."""

    def test_basin_stability_uses_engine(self):
        """basin_stability drives UPDEEngine for each random IC sample,
        proving the module is wired into the simulation core."""
        n = 4
        omegas = np.ones(n)
        knm = np.ones((n, n)) * 0.5
        np.fill_diagonal(knm, 0)
        result = basin_stability(
            omegas,
            knm,
            n_samples=10,
            n_transient=50,
            n_measure=20,
        )
        assert isinstance(result, BasinStabilityResult)
        assert 0.0 <= result.S_B <= 1.0
        assert result.n_samples == 10


class TestBasinStabilityValidation:
    def test_invalid_omegas_shape(self) -> None:
        N = 4
        knm = np.ones((N, N))
        np.fill_diagonal(knm, 0)
        with pytest.raises(
            ValueError,
            match="omegas shape \\(4, 1\\) must be one-dimensional",
        ):
            basin_stability(np.full((N, 1), 1.0), knm, n_samples=10)

    def test_invalid_coupling_shape(self) -> None:
        N = 4
        omegas = np.zeros(N)
        knm = np.ones((N - 1, N - 1))
        with pytest.raises(ValueError, match="shape"):
            basin_stability(omegas, knm, n_samples=10)

    def test_invalid_alpha_shape(self) -> None:
        N = 4
        omegas = np.zeros(N)
        knm = np.ones((N, N))
        alpha = np.zeros((N, N - 1))
        with pytest.raises(ValueError, match="shape"):
            basin_stability(
                omegas,
                knm,
                alpha=alpha,
                n_samples=10,
            )

    def test_nonfinite_inputs(self) -> None:
        N = 4
        omegas = np.ones(N)
        knm = np.ones((N, N))
        np.fill_diagonal(knm, 0)
        knm[0, 0] = np.nan
        with pytest.raises(ValueError, match="must contain only finite values"):
            basin_stability(omegas, knm, n_samples=10)

    @pytest.mark.parametrize(
        "value, param",
        [
            (0.0, "dt"),
            (-0.01, "dt"),
            (-1, "n_transient"),
            (-2, "n_measure"),
            (-3, "n_samples"),
            (-4, "seed"),
            (1.2, "R_threshold"),
            (-0.1, "R_threshold"),
        ],
    )
    def test_invalid_scalar_parameters(self, value: float | int, param: str) -> None:
        N = 4
        omegas = np.zeros(N)
        knm = np.ones((N, N))
        np.fill_diagonal(knm, 0)
        kwargs = {
            "omegas": omegas,
            "knm": knm,
            "n_samples": 10,
            "dt": 0.01,
            "n_transient": 10,
            "n_measure": 10,
            "R_threshold": 0.8,
            "seed": 7,
        }
        if param == "dt":
            kwargs["dt"] = value
        elif param == "n_transient":
            kwargs["n_transient"] = value
        elif param == "n_measure":
            kwargs["n_measure"] = value
        elif param == "n_samples":
            kwargs["n_samples"] = value
        elif param == "seed":
            kwargs["seed"] = value
        elif param == "R_threshold":
            kwargs["R_threshold"] = value
        with pytest.raises(ValueError, match=f"{param}"):
            basin_stability(**kwargs)

    def test_boolean_is_rejected_where_integer_is_required(self) -> None:
        N = 4
        omegas = np.zeros(N)
        knm = np.ones((N, N))
        np.fill_diagonal(knm, 0)
        with pytest.raises(ValueError, match="n_samples must be an integer >= 0"):
            basin_stability(omegas, knm, n_samples=True)


class TestBasinStabilityEdgeSemantics:
    def test_zero_samples_returns_empty_results(self) -> None:
        N = 4
        omegas = np.zeros(N)
        knm = np.ones((N, N))
        np.fill_diagonal(knm, 0)
        result = basin_stability(
            omegas,
            knm,
            n_samples=0,
            n_transient=10,
            n_measure=10,
            R_threshold=0.8,
        )
        assert result.n_samples == 0
        assert result.n_converged == 0
        assert result.S_B == 0.0
        assert result.R_final.shape == (0,)

    def test_zero_measurements_classify_with_zero_threshold(self) -> None:
        N = 4
        omegas = np.array([0.4, 0.5, 0.6, 0.7])
        knm = np.ones((N, N))
        np.fill_diagonal(knm, 0)
        result = basin_stability(
            omegas,
            knm,
            n_samples=12,
            n_transient=30,
            n_measure=0,
            R_threshold=0.0,
            seed=99,
        )
        assert result.n_samples == 12
        assert np.allclose(result.R_final, 0.0)
        assert result.n_converged == result.n_samples
        assert result.S_B == 1.0

    def test_steady_state_zero_measure_shortcuts_without_backend(self, monkeypatch):
        import scpn_phase_orchestrator.upde.basin_stability as basin_mod

        monkeypatch.setattr(
            basin_mod,
            "_dispatch",
            lambda: (_ for _ in ()).throw(
                RuntimeError("backend should not run"),
            ),
        )
        got = basin_mod.steady_state_r(
            np.array([0.1, 0.2]),
            np.array([1.0, -1.0]),
            np.ones((2, 2)),
            n_measure=0,
        )
        assert got == 0.0

    def test_multi_basin_rejects_empty_threshold_tuple(self) -> None:
        N = 4
        omegas = np.zeros(N)
        knm = np.ones((N, N))
        np.fill_diagonal(knm, 0)
        with pytest.raises(ValueError, match="at least one threshold"):
            multi_basin_stability(
                omegas,
                knm,
                n_samples=10,
                n_measure=10,
                R_thresholds=(),
            )
