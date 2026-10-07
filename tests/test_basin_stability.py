# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Basin stability tests

"""Public finite-window basin contracts and separate invalid-input controls."""

from __future__ import annotations

import cProfile
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _basin_stability_julia as basin_julia,
)
from scpn_phase_orchestrator.upde import (
    _basin_stability_validation as basin_validation,
)
from scpn_phase_orchestrator.upde import basin_stability as basin_mod
from scpn_phase_orchestrator.upde.basin_stability import (
    BasinStabilityResult,
    basin_stability,
    multi_basin_stability,
    steady_state_r,
)

FloatArray = NDArray[np.float64]
BasinBackend = Callable[
    [FloatArray, FloatArray, FloatArray, FloatArray, int, float, float, int, int], float
]


class MonteCarloRunner(Protocol):
    """Public sampling signature used by deliberately invalid output controls."""

    def __call__(
        self,
        omegas: FloatArray,
        knm: FloatArray,
        *,
        n_transient: int,
        n_measure: int,
        n_samples: int,
    ) -> object:
        """Run an actual public estimator with the selected fault-injected kernel."""
        ...


class _ArrayConversionFailure:
    def __array__(self, dtype: object | None = None) -> FloatArray:
        """Raise during NumPy coercion to exercise defensive validation paths."""
        raise TypeError("array conversion failed")


class _FakeJuliaMain:
    def __init__(self, module: object) -> None:
        """Create a minimal Julia ``Main`` stand-in for bridge-loader tests."""
        self.BasinStabilityJL = module
        self.included: list[str] = []

    def include(self, path: str) -> None:
        """Record the Julia side-file path requested by the bridge."""
        self.included.append(path)


class _FakeJuliaModule:
    def __init__(self, output: float = 0.75) -> None:
        """Create a minimal Julia backend module returning a fixed output."""
        self.output = output
        self.seen_n_measure: int | None = None

    def steady_state_r(
        self,
        phases_init: FloatArray,
        omegas: FloatArray,
        knm_flat: FloatArray,
        alpha_flat: FloatArray,
        n: int,
        k_scale: float,
        dt: float,
        n_transient: int,
        n_measure: int,
    ) -> float:
        """Record validated call inputs and return the configured output."""
        self.seen_n_measure = n_measure
        assert phases_init.flags.c_contiguous
        assert omegas.flags.c_contiguous
        assert knm_flat.flags.c_contiguous
        assert alpha_flat.flags.c_contiguous
        assert n >= 1
        assert np.isfinite(k_scale)
        assert dt > 0.0
        assert n_transient >= 0
        return self.output


class TestBasinStability:
    """Public finite-window sampling fractions, counts and reproducibility."""

    def test_identical_frequencies_high_stability(self) -> None:
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

    def test_zero_coupling_low_stability(self) -> None:
        """Zero coupling + spread frequencies → S_B ≈ 0."""
        N = 6
        rng = np.random.default_rng(42)
        omegas = rng.normal(0, 2.0, N)
        knm = np.zeros((N, N))
        result = basin_stability(
            omegas, knm, n_samples=20, n_transient=200, n_measure=50
        )
        assert result.S_B < 0.5

    def test_result_fields(self) -> None:
        """Returned counts and trial cardinality match the requested sample count."""
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

    def test_custom_threshold(self) -> None:
        """The actual estimator retains its requested inclusive threshold."""
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
    """Shared trial values classified at multiple inclusive thresholds."""

    def test_returns_dict(self) -> None:
        """The public multi-threshold result uses its established labels."""
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

    def test_monotonic_thresholds(self) -> None:
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
    """Public Monte Carlo results from the deterministic Euler trial kernel."""

    def test_public_monte_carlo_returns_finite_threshold_classification(self) -> None:
        """Exercise the original public Monte Carlo loop and its result fields."""
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
    """Reject malformed public arrays and scalar sampling controls."""

    def test_invalid_omegas_shape(self) -> None:
        """A frequency matrix cannot be admitted as a one-dimensional population."""
        N = 4
        knm = np.ones((N, N))
        np.fill_diagonal(knm, 0)
        with pytest.raises(
            ValueError,
            match="omegas shape \\(4, 1\\) must be one-dimensional",
        ):
            basin_stability(np.full((N, 1), 1.0), knm, n_samples=10)

    def test_invalid_coupling_shape(self) -> None:
        """Coupling dimensions must match the admitted oscillator population."""
        N = 4
        omegas = np.zeros(N)
        knm = np.ones((N - 1, N - 1))
        with pytest.raises(ValueError, match="shape"):
            basin_stability(omegas, knm, n_samples=10)

    def test_invalid_alpha_shape(self) -> None:
        """Lag dimensions must match the admitted oscillator population."""
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
        """Nonfinite graph entries are refused before numerical integration."""
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
        """Invalid step, sample, seed and threshold controls are refused."""
        N = 4
        omegas = np.zeros(N)
        knm = np.ones((N, N))
        np.fill_diagonal(knm, 0)
        dt = 0.01
        n_transient = 10
        n_measure = 10
        n_samples = 10
        r_threshold = 0.8
        seed = 7
        if param == "dt":
            dt = float(value)
        elif param == "n_transient":
            n_transient = int(value)
        elif param == "n_measure":
            n_measure = int(value)
        elif param == "n_samples":
            n_samples = int(value)
        elif param == "seed":
            seed = int(value)
        elif param == "R_threshold":
            r_threshold = float(value)
        with pytest.raises(ValueError, match=f"{param}"):
            basin_stability(
                omegas,
                knm,
                dt=dt,
                n_transient=n_transient,
                n_measure=n_measure,
                n_samples=n_samples,
                R_threshold=r_threshold,
                seed=seed,
            )

    def test_nonreal_public_scalar_is_rejected(self) -> None:
        """Public scalar validation rejects non-real objects."""
        with pytest.raises(ValueError, match="dt must be a finite real"):
            basin_stability(
                np.zeros(2, dtype=np.float64),
                np.array([[0.0, 0.4], [0.4, 0.0]], dtype=np.float64),
                dt=cast(float, object()),
                n_samples=1,
                n_transient=1,
                n_measure=1,
            )

    def test_nonfinite_public_scalar_is_rejected(self) -> None:
        """Public scalar validation rejects non-finite real values."""
        with pytest.raises(ValueError, match="dt must be a finite real"):
            basin_stability(
                np.zeros(2, dtype=np.float64),
                np.array([[0.0, 0.4], [0.4, 0.0]], dtype=np.float64),
                dt=float("nan"),
                n_samples=1,
                n_transient=1,
                n_measure=1,
            )

    def test_public_vectors_reject_boolean_aliases(self) -> None:
        """Public vector validation rejects boolean aliases before coercion."""
        with pytest.raises(ValueError, match="omegas must not contain boolean"):
            basin_stability(
                np.array([False, True], dtype=object),
                np.array([[0.0, 0.4], [0.4, 0.0]], dtype=np.float64),
                n_samples=1,
                n_transient=1,
                n_measure=1,
            )

    def test_public_matrix_rejects_boolean_aliases(self) -> None:
        """Public matrix validation rejects boolean aliases before coercion."""
        with pytest.raises(ValueError, match="knm must not contain boolean"):
            basin_stability(
                np.zeros(2, dtype=np.float64),
                np.array([[False, 0.4], [0.4, False]], dtype=object),
                n_samples=1,
                n_transient=1,
                n_measure=1,
            )

    def test_public_omegas_reject_nonfinite_values(self) -> None:
        """Public natural frequencies must be finite."""
        with pytest.raises(ValueError, match="omegas must contain only finite"):
            basin_stability(
                np.array([0.0, np.inf], dtype=np.float64),
                np.array([[0.0, 0.4], [0.4, 0.0]], dtype=np.float64),
                n_samples=1,
                n_transient=1,
                n_measure=1,
            )

    def test_boolean_is_rejected_where_integer_is_required(self) -> None:
        """Boolean counts cannot masquerade as integer sampling controls."""
        N = 4
        omegas = np.zeros(N)
        knm = np.ones((N, N))
        np.fill_diagonal(knm, 0)
        with pytest.raises(ValueError, match="n_samples must be an integer >= 0"):
            basin_stability(omegas, knm, n_samples=True)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("omegas", np.array(["0.0", "0.1"], dtype=object)),
            (
                "knm",
                np.array([["0.0", "0.4"], ["0.4", "0.0"]], dtype=object),
            ),
            (
                "alpha",
                np.array([["0.0", "0.1"], ["0.2", "0.0"]], dtype=object),
            ),
            ("dt", "0.01"),
            ("n_samples", "4"),
            ("R_threshold", "0.8"),
        ],
    )
    def test_numeric_string_aliases_are_rejected_before_public_coercion(
        self, field: str, value: object
    ) -> None:
        """Public basin-stability inputs must reject numeric strings."""
        kwargs: dict[str, object] = {
            "omegas": np.zeros(2, dtype=np.float64),
            "knm": np.array([[0.0, 0.4], [0.4, 0.0]], dtype=np.float64),
            "alpha": np.zeros((2, 2), dtype=np.float64),
            "dt": 0.01,
            "n_samples": 4,
            "R_threshold": 0.8,
            "n_transient": 1,
            "n_measure": 1,
        }
        kwargs[field] = value

        with pytest.raises(ValueError, match="numeric-string"):
            basin_stability(
                omegas=cast("FloatArray", kwargs["omegas"]),
                knm=cast("FloatArray", kwargs["knm"]),
                alpha=cast("FloatArray | None", kwargs.get("alpha")),
                dt=cast("float", kwargs.get("dt", 0.01)),
                n_transient=cast("int", kwargs.get("n_transient", 500)),
                n_measure=cast("int", kwargs.get("n_measure", 200)),
                n_samples=cast("int", kwargs.get("n_samples", 100)),
                R_threshold=cast("float", kwargs.get("R_threshold", 0.8)),
                seed=cast("int", kwargs.get("seed", 42)),
                backend=cast("str | None", kwargs.get("backend")),
            )

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("phases_init", np.array(["0.1", "0.2"], dtype=object)),
            ("omegas", np.array(["1.0", "1.1"], dtype=object)),
            (
                "knm",
                np.array([["0.0", "0.4"], ["0.4", "0.0"]], dtype=object),
            ),
            (
                "alpha",
                np.array([["0.0", "0.1"], ["0.2", "0.0"]], dtype=object),
            ),
            ("k_scale", "1.0"),
            ("dt", "0.01"),
            ("n_transient", "1"),
            ("n_measure", "1"),
        ],
    )
    def test_steady_state_numeric_string_aliases_are_rejected(
        self, field: str, value: object
    ) -> None:
        """Public one-trial inputs must reject numeric strings."""
        kwargs: dict[str, object] = {
            "phases_init": np.zeros(2, dtype=np.float64),
            "omegas": np.ones(2, dtype=np.float64),
            "knm": np.array([[0.0, 0.4], [0.4, 0.0]], dtype=np.float64),
            "alpha": np.zeros((2, 2), dtype=np.float64),
            "k_scale": 1.0,
            "dt": 0.01,
            "n_transient": 1,
            "n_measure": 1,
        }
        kwargs[field] = value

        with pytest.raises(ValueError, match="numeric-string"):
            steady_state_r(
                phases_init=cast("FloatArray", kwargs["phases_init"]),
                omegas=cast("FloatArray", kwargs["omegas"]),
                knm=cast("FloatArray", kwargs["knm"]),
                alpha=cast("FloatArray | None", kwargs.get("alpha")),
                k_scale=cast("float", kwargs.get("k_scale", 1.0)),
                dt=cast("float", kwargs.get("dt", 0.01)),
                n_transient=cast("int", kwargs.get("n_transient", 500)),
                n_measure=cast("int", kwargs.get("n_measure", 200)),
                backend=cast("str | None", kwargs.get("backend")),
            )


class TestBasinStabilityEdgeSemantics:
    """Empty sampling and measurement windows retain explicit conventions."""

    def test_zero_samples_returns_empty_results(self) -> None:
        """No sampled initial conditions produces empty values and a zero fraction."""
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
        """An empty window yields zero R, satisfying an inclusive zero threshold."""
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

    def test_result_rejects_converged_count_above_samples(self) -> None:
        """Result construction enforces converged/sample consistency."""
        with pytest.raises(ValueError, match="n_converged must be <= n_samples"):
            BasinStabilityResult(
                S_B=1.0,
                n_samples=1,
                n_converged=2,
                R_final=np.array([1.0], dtype=np.float64),
                R_threshold=0.8,
            )

    def test_result_rejects_out_of_range_final_order_parameter(self) -> None:
        """Result construction enforces unit-interval final R values."""
        with pytest.raises(ValueError, match="R_final values must lie in"):
            BasinStabilityResult(
                S_B=1.0,
                n_samples=1,
                n_converged=1,
                R_final=np.array([1.2], dtype=np.float64),
                R_threshold=0.8,
            )

    def test_multi_basin_accepts_phase_lag_matrix(self) -> None:
        """Multi-threshold estimation validates and uses non-null phase lag."""
        results = multi_basin_stability(
            np.zeros(2, dtype=np.float64),
            np.array([[0.0, 0.4], [0.4, 0.0]], dtype=np.float64),
            alpha=np.zeros((2, 2), dtype=np.float64),
            n_samples=1,
            n_transient=1,
            n_measure=1,
            R_thresholds=(0.5,),
        )

        assert list(results) == ["R>=0.50"]


class TestPublicBasinStabilityOutputContracts:
    """Deliberately invalid scalar returns are negative boundary controls."""

    def _problem(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        phases = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        omegas = np.array([1.0, 1.1, 1.2], dtype=np.float64)
        knm = np.ones((3, 3), dtype=np.float64)
        np.fill_diagonal(knm, 0.0)
        return phases, omegas, knm

    @pytest.mark.parametrize(
        ("output", "match"),
        [
            (True, "steady-state R"),
            ("0.5", "numeric-string"),
            (1.2, r"\[0, 1\]"),
            (float("nan"), "finite"),
        ],
    )
    def test_steady_state_rejects_invalid_optional_backend_output(
        self,
        monkeypatch: pytest.MonkeyPatch,
        output: object,
        match: str,
    ) -> None:
        """Deliberately invalid scalars cannot become published trial measurements."""

        def fake_backend(*_args: object) -> object:
            return output

        phases, omegas, knm = self._problem()
        monkeypatch.setattr(basin_mod, "_dispatch", lambda _owner=None: fake_backend)

        with pytest.raises((TypeError, ValueError), match=match):
            steady_state_r(
                phases,
                omegas,
                knm,
                n_transient=1,
                n_measure=1,
            )

    @pytest.mark.parametrize(
        "runner",
        [basin_stability, multi_basin_stability],
        ids=["basin", "multi"],
    )
    def test_public_monte_carlo_rejects_boolean_backend_output(
        self,
        monkeypatch: pytest.MonkeyPatch,
        runner: MonteCarloRunner,
    ) -> None:
        """Deliberate boolean returns cannot enter either public sample classifier."""

        def fake_backend(*_args: object) -> bool:
            return True

        _, omegas, knm = self._problem()
        monkeypatch.setattr(basin_mod, "_dispatch", lambda _owner=None: fake_backend)

        with pytest.raises(TypeError, match="steady-state R"):
            runner(
                omegas,
                knm,
                n_transient=1,
                n_measure=1,
                n_samples=3,
            )

    @pytest.mark.parametrize(
        "runner",
        [basin_stability, multi_basin_stability],
        ids=["basin", "multi"],
    )
    def test_public_monte_carlo_rejects_numeric_string_backend_output(
        self,
        monkeypatch: pytest.MonkeyPatch,
        runner: MonteCarloRunner,
    ) -> None:
        """Monte Carlo publication must reject stringified backend scalars."""

        def fake_backend(*_args: object) -> str:
            return "0.5"

        _, omegas, knm = self._problem()
        monkeypatch.setattr(basin_mod, "_dispatch", lambda _owner=None: fake_backend)

        with pytest.raises(ValueError, match="numeric-string"):
            runner(
                omegas,
                knm,
                n_transient=1,
                n_measure=1,
                n_samples=3,
            )

    def test_rust_loader_rejects_boolean_backend_output(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A deliberately invalid native return is rejected by its original wrapper."""

        def fake_steady_state(*_args: object) -> bool:
            return True

        fake_spo = types.ModuleType("spo_kernel")
        monkeypatch.setattr(
            fake_spo, "steady_state_r_rust", fake_steady_state, raising=False
        )
        monkeypatch.setitem(sys.modules, "spo_kernel", fake_spo)

        phases, omegas, knm = self._problem()
        wrapped = basin_mod._load_rust_fn()

        with pytest.raises(TypeError, match="steady-state R"):
            wrapped(
                phases,
                omegas,
                knm.ravel(),
                np.zeros(knm.size, dtype=np.float64),
                3,
                1.0,
                0.01,
                1,
                1,
            )


class TestBasinStabilityDefensiveContracts:
    """Reject malformed ingress and unavailable-owner negative controls."""

    def test_dispatch_returns_python_fallback_when_all_loaders_fail(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Injected loader failures leave the explicit Python fallback available."""

        def fail_backend(_name: str) -> BasinBackend:
            raise ImportError("backend unavailable")

        monkeypatch.setattr(basin_mod, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(basin_mod, "AVAILABLE_BACKENDS", ["rust"])
        monkeypatch.setattr(basin_mod, "_load_backend", fail_backend)

        assert basin_mod._dispatch() is None

    def test_boolean_alias_probe_treats_conversion_failure_as_not_boolean(
        self,
    ) -> None:
        """A failing conversion hook is not itself a boolean-alias observation."""
        assert basin_mod._contains_boolean_alias(_ArrayConversionFailure()) is False

    def test_numeric_string_probe_treats_conversion_failure_as_not_string(
        self,
    ) -> None:
        """A failing conversion hook is not itself a numeric-string observation."""
        assert (
            basin_mod._contains_numeric_string_alias(_ArrayConversionFailure()) is False
        )

    def test_steady_state_rejects_uncoercible_matrix_values(self) -> None:
        """Nonnumeric graph values cannot cross the real-valued public boundary."""
        bad_knm = cast(FloatArray, np.array([["not-float"]], dtype=object))

        with pytest.raises(ValueError, match="knm must be a finite float array"):
            steady_state_r(
                np.zeros(1, dtype=np.float64),
                np.ones(1, dtype=np.float64),
                bad_knm,
                n_transient=1,
                n_measure=1,
            )

    def test_steady_state_rejects_uncoercible_phase_vector(self) -> None:
        """Nonnumeric initial phases cannot cross the real-valued public boundary."""
        bad_phases = cast(FloatArray, np.array(["not-float"], dtype=object))

        with pytest.raises(
            ValueError,
            match="phases_init must be a finite one-dimensional array",
        ):
            steady_state_r(
                bad_phases,
                np.ones(1, dtype=np.float64),
                np.zeros((1, 1), dtype=np.float64),
                n_transient=1,
                n_measure=1,
            )

    def test_steady_state_rejects_empty_phase_vector(self) -> None:
        """A trial requires at least one oscillator after ingress validation."""
        with pytest.raises(
            ValueError,
            match="phases_init must contain at least one oscillator",
        ):
            steady_state_r(
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.zeros((0, 0), dtype=np.float64),
                n_transient=1,
                n_measure=1,
            )

    def test_public_python_zero_measurement_returns_zero(self) -> None:
        """The original public zero-window convention returns exactly zero."""
        assert (
            steady_state_r(
                np.zeros(1),
                np.ones(1),
                np.zeros((1, 1)),
                n_transient=1,
                n_measure=0,
                backend="python",
            )
            == 0.0
        )


class TestDirectBasinStabilityValidationContracts:
    """Direct accelerator ingress rejects aliases and invalid dimensions."""

    def test_rejects_nonnumeric_backend_vectors(self) -> None:
        """Direct flattened inputs refuse text rather than converting it to phases."""
        with pytest.raises(TypeError, match="phases_init must be numeric"):
            basin_validation.validate_basin_stability_inputs(
                cast(FloatArray, np.array(["not-float"], dtype=object)),
                np.ones(1, dtype=np.float64),
                np.zeros(1, dtype=np.float64),
                np.zeros(1, dtype=np.float64),
                1,
                1.0,
                0.01,
                0,
                1,
            )

    def test_rejects_empty_backend_vectors(self) -> None:
        """Direct accelerator ingress requires nonempty oscillator vectors."""
        with pytest.raises(
            ValueError,
            match="phases_init must contain at least one oscillator",
        ):
            basin_validation.validate_basin_stability_inputs(
                np.array([], dtype=np.float64),
                np.ones(1, dtype=np.float64),
                np.zeros(1, dtype=np.float64),
                np.zeros(1, dtype=np.float64),
                1,
                1.0,
                0.01,
                0,
                1,
            )

    def test_rejects_nonreal_backend_scalars(self) -> None:
        """Direct numerical controls reject non-real scalar objects."""
        with pytest.raises(TypeError, match="k_scale must be a real scalar"):
            basin_validation.validate_basin_stability_inputs(
                np.zeros(1, dtype=np.float64),
                np.ones(1, dtype=np.float64),
                np.zeros(1, dtype=np.float64),
                np.zeros(1, dtype=np.float64),
                1,
                cast(float, "not-real"),
                0.01,
                0,
                1,
            )

    def test_rejects_noninteger_backend_counts(self) -> None:
        """Direct oscillator dimensions must be integer-valued metadata."""
        with pytest.raises(TypeError, match="n must be an integer"):
            basin_validation.validate_basin_stability_inputs(
                np.zeros(1, dtype=np.float64),
                np.ones(1, dtype=np.float64),
                np.zeros(1, dtype=np.float64),
                np.zeros(1, dtype=np.float64),
                cast(int, "not-int"),
                1.0,
                0.01,
                0,
                1,
            )

    def test_rejects_numeric_string_backend_output(self) -> None:
        """Backend outputs must not be stringified order parameters."""
        with pytest.raises(ValueError, match="numeric-string"):
            basin_validation.validate_basin_stability_output("0.5")

    def test_direct_numeric_string_probe_covers_scalar_and_failure_paths(
        self,
    ) -> None:
        """Direct validator helper handles scalar aliases and bad array hooks."""
        assert basin_validation._contains_numeric_string_alias("0.5") is True
        assert (
            basin_validation._contains_numeric_string_alias(_ArrayConversionFailure())
            is False
        )


class TestBasinStabilityJuliaBridgeContracts:
    """Original Julia computations, cache reuse and explicit negative fault controls."""

    def test_named_julia_public_trial_executes_original_side_module(self) -> None:
        """A named original trial retains the analytically required tiny-edge effect."""
        phases = np.array([0.0, np.pi / 2])
        graph = np.array([[0.0, 5e-31], [5e-31, 0.0]])
        if "julia" not in basin_mod.AVAILABLE_BACKENDS:
            with pytest.raises(ImportError, match="requested basin backend 'julia'"):
                steady_state_r(
                    phases,
                    np.zeros(2),
                    graph,
                    dt=1e30,
                    n_transient=0,
                    n_measure=1,
                    backend="julia",
                )
            return
        value = steady_state_r(
            phases,
            np.zeros(2),
            graph,
            dt=1e30,
            n_transient=0,
            n_measure=1,
            backend="julia",
        )
        assert value == pytest.approx(np.cos((np.pi / 2 - 1) / 2), abs=2e-15)
        np.testing.assert_array_equal(phases, [0.0, np.pi / 2])

    def test_public_julia_calls_reuse_actual_loaded_module(self) -> None:
        """Repeated real computations retain the same original Julia module object."""
        phases = np.array([0.0, 1.0])
        graph = np.zeros((2, 2))
        if "julia" not in basin_mod.AVAILABLE_BACKENDS:
            with pytest.raises(ImportError, match="requested basin backend 'julia'"):
                steady_state_r(
                    phases,
                    np.zeros(2),
                    graph,
                    n_transient=0,
                    n_measure=1,
                    backend="julia",
                )
            return
        first = steady_state_r(
            phases, np.zeros(2), graph, n_transient=0, n_measure=1, backend="julia"
        )
        loaded = basin_julia._JULIA_MODULE
        assert loaded is not None
        second = steady_state_r(
            phases, np.zeros(2), graph, n_transient=0, n_measure=1, backend="julia"
        )
        assert basin_julia._JULIA_MODULE is loaded
        assert first == second == pytest.approx(np.cos(0.5), abs=2e-15)

    def test_original_julia_arithmetic_error_allows_a_healthy_following_trial(
        self,
    ) -> None:
        """An original native overflow propagates without poisoning a later trial."""
        phases = np.array([0.0, 1.0])
        graph = np.zeros((2, 2))
        if "julia" not in basin_mod.AVAILABLE_BACKENDS:
            with pytest.raises(ImportError, match="requested basin backend 'julia'"):
                steady_state_r(
                    phases,
                    np.zeros(2),
                    graph,
                    n_transient=0,
                    n_measure=1,
                    backend="julia",
                )
            return
        with pytest.raises(ValueError):
            steady_state_r(
                phases,
                np.full(2, 1e308),
                graph,
                dt=2.0,
                n_transient=0,
                n_measure=1,
                backend="julia",
            )
        recovered = steady_state_r(
            phases, np.zeros(2), graph, n_transient=0, n_measure=1, backend="julia"
        )
        assert recovered == pytest.approx(np.cos(0.5), abs=2e-15)

    def test_ensure_reports_missing_side_file(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """A missing Julia source is an explicit load error."""
        monkeypatch.setattr(basin_julia, "_JULIA_MODULE", None)
        monkeypatch.setattr(basin_julia, "_JULIA_FILE", tmp_path / "missing.jl")
        monkeypatch.setattr(
            basin_julia,
            "require_julia_main",
            lambda: _FakeJuliaMain(object()),
        )

        with pytest.raises(ImportError, match="julia side-file not found"):
            basin_julia._ensure()

    def test_steady_state_bridge_rejects_numeric_string_output(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Julia backend return values must fail before string coercion."""
        fake_module = _FakeJuliaModule(output=cast(float, "0.75"))
        monkeypatch.setattr(basin_julia, "_ensure", lambda: fake_module)

        with pytest.raises(ValueError, match="numeric-string"):
            basin_julia.steady_state_r_julia(
                np.zeros(2, dtype=np.float64),
                np.ones(2, dtype=np.float64),
                np.zeros(4, dtype=np.float64),
                np.zeros(4, dtype=np.float64),
                2,
                1.0,
                0.01,
                0,
                1,
            )


def test_basin_stability_docs_record_numeric_string_contract() -> None:
    """Public docs must record the basin numeric-string boundary."""
    doc = Path("docs/reference/api/upde_basin_stability.md").read_text(encoding="utf-8")

    assert "numeric-string aliases are rejected before float coercion" in doc


class TestDispatchFallbackChain:
    """Observe public computation after an injected first-loader failure."""

    def test_repeated_public_trials_reuse_the_original_named_owner(self) -> None:
        """Actual repeated values retain the cached original callable identity."""
        owner = next(
            (
                name
                for name in ("rust", "go", "julia", "mojo")
                if name in basin_mod.AVAILABLE_BACKENDS
            ),
            "python",
        )
        phases = np.array([0.0, 1.0])
        graph = np.zeros((2, 2))
        first = steady_state_r(
            phases, np.zeros(2), graph, n_transient=0, n_measure=1, backend=owner
        )
        cached = basin_mod._BACKEND_CACHE.get(owner)
        second = steady_state_r(
            phases, np.zeros(2), graph, n_transient=0, n_measure=1, backend=owner
        )
        assert first == second == pytest.approx(np.cos(0.5), abs=2e-15)
        if owner != "python":
            assert cached is not None and basin_mod._BACKEND_CACHE[owner] is cached

    def test_dispatch_falls_back_to_next_backend_when_active_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An injected first-loader failure leaves subsequent owners original."""
        if basin_mod.ACTIVE_BACKEND == "python":
            assert steady_state_r(
                np.array([0.0, 1.0]),
                np.zeros(2),
                np.zeros((2, 2)),
                n_transient=0,
                n_measure=1,
            ) == pytest.approx(np.cos(0.5))
            return
        active = basin_mod.ACTIVE_BACKEND

        def unavailable() -> BasinBackend:
            """Negative control: fail the actual first preference at load time."""
            raise ImportError("first owner unavailable")

        monkeypatch.setattr(basin_mod, "_BACKEND_CACHE", {})
        monkeypatch.setitem(basin_mod._LOADERS, active, unavailable)
        with cProfile.Profile() as profile:
            result = steady_state_r(
                np.array([0.0, 1.0]),
                np.zeros(2),
                np.zeros((2, 2)),
                n_transient=0,
                n_measure=1,
            )
        assert result == pytest.approx(np.cos(0.5), abs=2e-15)
        profile.create_stats()
        assert any("steady_state_r" in key[2] for key in profile.stats)

    def test_public_empty_window_does_not_integrate_unused_transient(self) -> None:
        """An unused transient cannot overflow an empty public measurement window."""
        assert (
            steady_state_r(
                np.array([0.1, 0.2]),
                np.array([1e308, -1e308]),
                np.zeros((2, 2)),
                dt=2.0,
                n_transient=10,
                n_measure=0,
                backend="python",
            )
            == 0.0
        )

    def test_multi_basin_rejects_empty_threshold_tuple(self) -> None:
        """Multi-threshold classification requires at least one threshold."""
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


class TestBasinDispatch:
    """Distinguish automatic fallback from strict named-owner refusal."""

    def test_dispatch_falls_back_to_python_when_loader_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An injected unavailable loader leaves the Python automatic fallback."""
        import scpn_phase_orchestrator.upde.basin_stability as basin_mod

        previous_backend = basin_mod.ACTIVE_BACKEND
        previous_available = list(basin_mod.AVAILABLE_BACKENDS)
        previous_loader = basin_mod._LOADERS["go"]
        previous_cache = dict(basin_mod._BACKEND_CACHE)
        basin_mod.ACTIVE_BACKEND = "go"
        basin_mod.AVAILABLE_BACKENDS = ["go", "python"]
        basin_mod._BACKEND_CACHE.clear()
        monkeypatch.setitem(
            basin_mod._LOADERS,
            "go",
            lambda: (_ for _ in ()).throw(ImportError("go backend unavailable")),
        )
        try:
            backend = basin_mod._dispatch()
        finally:
            basin_mod.ACTIVE_BACKEND = previous_backend
            basin_mod.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(basin_mod._LOADERS, "go", previous_loader)
            basin_mod._BACKEND_CACHE = previous_cache

        assert backend is None

    def test_dispatch_uses_next_available_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A named first-owner load failure cannot return an automatic fallback."""
        if basin_mod.ACTIVE_BACKEND == "python":
            assert steady_state_r(
                np.array([0.0, 1.0]),
                np.zeros(2),
                np.zeros((2, 2)),
                n_transient=0,
                n_measure=1,
            ) == pytest.approx(np.cos(0.5))
            return
        active = basin_mod.ACTIVE_BACKEND

        def unavailable() -> BasinBackend:
            """Negative control: fail the actual first preference at load time."""
            raise ImportError("first owner unavailable")

        monkeypatch.setattr(basin_mod, "_BACKEND_CACHE", {})
        monkeypatch.setitem(basin_mod._LOADERS, active, unavailable)
        with pytest.raises(ImportError, match="requested basin backend"):
            steady_state_r(
                np.array([0.0, 1.0]),
                np.zeros(2),
                np.zeros((2, 2)),
                n_transient=0,
                n_measure=1,
                backend=active,
            )
