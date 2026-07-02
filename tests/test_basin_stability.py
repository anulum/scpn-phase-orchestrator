# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Basin stability tests

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _basin_stability_julia as basin_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.upde import (
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
BasinBackend = Callable[..., float]


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
    def test_returns_dict(self) -> None:
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
    """Pipeline: basin_stability uses UPDEEngine internally."""

    def test_basin_stability_uses_engine(self) -> None:
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


class TestPublicBasinStabilityOutputContracts:
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
        def fake_backend(*_args: object) -> object:
            return output

        phases, omegas, knm = self._problem()
        monkeypatch.setattr(basin_mod, "_dispatch", lambda: fake_backend)

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
        runner: Callable[..., object],
    ) -> None:
        def fake_backend(*_args: object) -> bool:
            return True

        _, omegas, knm = self._problem()
        monkeypatch.setattr(basin_mod, "_dispatch", lambda: fake_backend)

        with pytest.raises(TypeError, match="steady-state R"):
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
        def fake_steady_state(*_args: object) -> bool:
            return True

        fake_spo = types.ModuleType("spo_kernel")
        fake_spo_dynamic = cast(Any, fake_spo)
        fake_spo_dynamic.steady_state_r_rust = fake_steady_state
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
    def test_dispatch_returns_python_fallback_when_all_loaders_fail(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fail_backend(_name: str) -> BasinBackend:
            raise ImportError("backend unavailable")

        monkeypatch.setattr(basin_mod, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(basin_mod, "AVAILABLE_BACKENDS", ["rust"])
        monkeypatch.setattr(basin_mod, "_load_backend", fail_backend)

        assert basin_mod._dispatch() is None

    def test_boolean_alias_probe_treats_conversion_failure_as_not_boolean(
        self,
    ) -> None:
        assert basin_mod._contains_boolean_alias(_ArrayConversionFailure()) is False

    def test_steady_state_rejects_uncoercible_matrix_values(self) -> None:
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

    def test_python_reference_zero_measurement_returns_zero(self) -> None:
        assert (
            basin_mod._python_steady_state_r(
                np.zeros(1, dtype=np.float64),
                np.ones(1, dtype=np.float64),
                np.zeros(1, dtype=np.float64),
                np.zeros(1, dtype=np.float64),
                1,
                1.0,
                0.01,
                1,
                0,
            )
            == 0.0
        )


class TestDirectBasinStabilityValidationContracts:
    def test_rejects_nonnumeric_backend_vectors(self) -> None:
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
        with pytest.raises(TypeError, match="k_scale must be a real scalar"):
            basin_validation.validate_basin_stability_inputs(
                np.zeros(1, dtype=np.float64),
                np.ones(1, dtype=np.float64),
                np.zeros(1, dtype=np.float64),
                np.zeros(1, dtype=np.float64),
                1,
                cast(float, "1.0"),
                0.01,
                0,
                1,
            )

    def test_rejects_noninteger_backend_counts(self) -> None:
        with pytest.raises(TypeError, match="n must be an integer"):
            basin_validation.validate_basin_stability_inputs(
                np.zeros(1, dtype=np.float64),
                np.ones(1, dtype=np.float64),
                np.zeros(1, dtype=np.float64),
                np.zeros(1, dtype=np.float64),
                cast(int, "1"),
                1.0,
                0.01,
                0,
                1,
            )


class TestBasinStabilityJuliaBridgeContracts:
    def test_ensure_loads_side_file_and_caches_module(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        module = object()
        fake_main = _FakeJuliaMain(module)
        side_file = tmp_path / "basin_stability.jl"
        side_file.write_text("module BasinStabilityJL\nend\n", encoding="utf-8")

        monkeypatch.setattr(basin_julia, "_JULIA_MODULE", None)
        monkeypatch.setattr(basin_julia, "_JULIA_FILE", side_file)
        monkeypatch.setattr(basin_julia, "require_julia_main", lambda: fake_main)

        assert basin_julia._ensure() is module
        assert basin_julia._ensure() is module
        assert fake_main.included == [str(side_file)]

    def test_ensure_reports_missing_side_file(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(basin_julia, "_JULIA_MODULE", None)
        monkeypatch.setattr(basin_julia, "_JULIA_FILE", tmp_path / "missing.jl")
        monkeypatch.setattr(
            basin_julia,
            "require_julia_main",
            lambda: _FakeJuliaMain(object()),
        )

        with pytest.raises(ImportError, match="julia side-file not found"):
            basin_julia._ensure()

    def test_steady_state_bridge_validates_and_returns_backend_output(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_module = _FakeJuliaModule(output=0.75)
        monkeypatch.setattr(basin_julia, "_ensure", lambda: fake_module)

        result = basin_julia.steady_state_r_julia(
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

        assert result == 0.75
        assert fake_module.seen_n_measure == 1


class TestDispatchFallbackChain:
    def test_dispatch_falls_back_to_next_backend_when_active_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scpn_phase_orchestrator.upde.basin_stability as basin_mod

        calls: dict[str, int] = {"rust": 0, "go": 0}

        def _fail_rust() -> BasinBackend:
            calls["rust"] += 1
            raise ImportError("rust unavailable")

        def _ok_go() -> BasinBackend:
            calls["go"] += 1

            def backend(*_args: object, **_kwargs: object) -> float:
                return 0.5

            return backend

        monkeypatch.setattr(basin_mod, "_BACKEND_CACHE", {})
        monkeypatch.setattr(basin_mod, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(basin_mod, "AVAILABLE_BACKENDS", ["rust", "go", "python"])
        monkeypatch.setattr(basin_mod, "_LOADERS", {"rust": _fail_rust, "go": _ok_go})

        fn = basin_mod._dispatch()
        assert fn is not None
        assert float(fn()) == 0.5
        assert calls == {"rust": 1, "go": 1}

    def test_dispatch_uses_cached_loader_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scpn_phase_orchestrator.upde.basin_stability as basin_mod

        calls: dict[str, int] = {"go": 0}

        def _ok_go() -> BasinBackend:
            calls["go"] += 1

            def backend(*_args: object, **_kwargs: object) -> float:
                return 0.25

            return backend

        monkeypatch.setattr(basin_mod, "_BACKEND_CACHE", {})
        monkeypatch.setattr(basin_mod, "ACTIVE_BACKEND", "go")
        monkeypatch.setattr(basin_mod, "AVAILABLE_BACKENDS", ["go", "python"])
        monkeypatch.setattr(basin_mod, "_LOADERS", {"go": _ok_go})

        basin_mod._dispatch()
        basin_mod._dispatch()

        assert calls["go"] == 1

    def test_steady_state_zero_measure_shortcuts_without_backend(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
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


class TestBasinDispatch:
    def test_dispatch_falls_back_to_python_when_loader_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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
        import scpn_phase_orchestrator.upde.basin_stability as basin_mod

        previous_backend = basin_mod.ACTIVE_BACKEND
        previous_available = list(basin_mod.AVAILABLE_BACKENDS)
        previous_go = basin_mod._LOADERS["go"]
        previous_rust = basin_mod._LOADERS["rust"]
        previous_cache = dict(basin_mod._BACKEND_CACHE)
        basin_mod.ACTIVE_BACKEND = "go"
        basin_mod.AVAILABLE_BACKENDS = ["go", "rust", "python"]
        basin_mod._BACKEND_CACHE.clear()

        def fake_backend(*_args: object) -> float:
            return 0.0

        def fail_go() -> BasinBackend:
            raise ImportError("go backend unavailable")

        monkeypatch.setitem(
            basin_mod._LOADERS,
            "go",
            fail_go,
        )
        monkeypatch.setitem(basin_mod._LOADERS, "rust", lambda: fake_backend)
        try:
            backend = basin_mod._dispatch()
        finally:
            basin_mod.ACTIVE_BACKEND = previous_backend
            basin_mod.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(basin_mod._LOADERS, "go", previous_go)
            monkeypatch.setitem(basin_mod._LOADERS, "rust", previous_rust)
            basin_mod._BACKEND_CACHE = previous_cache

        assert backend is fake_backend
