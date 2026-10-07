# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Bifurcation continuation tests

"""Public finite-horizon sweep, search and generic record contracts."""

from __future__ import annotations

import cProfile
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from benchmarks.kuramoto_trial_reference import scalar_trial
from scpn_phase_orchestrator.upde import bifurcation as bif
from scpn_phase_orchestrator.upde.bifurcation import (
    BifurcationDiagram,
    BifurcationPoint,
    find_critical_coupling,
    trace_sync_transition,
)

FloatArray = NDArray[np.float64]


class TestBifurcationPoint:
    """Finite sample records preserve their compatibility marker and values."""

    def test_fields(self) -> None:
        """A valid sample record preserves its supplied numerical fields."""
        p = BifurcationPoint(K=1.0, R=0.5, stable=True)
        assert p.K == 1.0
        assert p.R == 0.5

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"K": -0.1}, "K must be non-negative"),
            ({"K": np.nan}, "K must be a finite real"),
            ({"K": False}, "K must be a finite real"),
            ({"R": -0.1}, "R must be in \\[0, 1\\]"),
            ({"R": 1.1}, "R must be in \\[0, 1\\]"),
            ({"R": np.inf}, "R must be a finite real"),
            ({"R": False}, "R must be a finite real"),
            ({"stable": 1}, "stable must be a boolean flag"),
        ],
    )
    def test_rejects_invalid_physical_sample_values(
        self, kwargs: dict[str, object], match: str
    ) -> None:
        """Malformed finite-sample fields cannot enter a public diagram."""
        base: dict[str, object] = {"K": 1.0, "R": 0.5, "stable": True}
        base.update(kwargs)
        with pytest.raises(ValueError, match=match):
            BifurcationPoint(
                K=cast("float", base["K"]),
                R=cast("float", base["R"]),
                stable=cast("bool", base["stable"]),
            )


class TestBifurcationDiagram:
    """Generic sampled diagram records and their ordered array views."""

    def test_empty(self) -> None:
        """An empty generic diagram has no samples or critical estimate."""
        d = BifurcationDiagram()
        assert len(d.points) == 0
        assert d.K_critical is None

    def test_properties(self) -> None:
        """Ordered record fields produce matching K and R array views."""
        d = BifurcationDiagram(
            points=[
                BifurcationPoint(K=0.0, R=0.01, stable=True),
                BifurcationPoint(K=2.0, R=0.8, stable=True),
            ]
        )
        np.testing.assert_array_equal(d.K_values, [0.0, 2.0])
        np.testing.assert_array_equal(d.R_values, [0.01, 0.8])

    def test_valid_critical_coupling_is_normalised_to_float(self) -> None:
        """A finite real critical value retains its canonical Python float form."""
        diagram = BifurcationDiagram(K_critical=cast(float, np.float32(1.25)))

        assert diagram.K_critical == 1.25
        assert isinstance(diagram.K_critical, float)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"points": (BifurcationPoint(K=1.0, R=0.5, stable=True),)}, "points"),
            ({"points": [object()]}, "points\\[0\\]"),
            ({"K_critical": -1.0}, "K_critical must be non-negative"),
            ({"K_critical": np.nan}, "K_critical must be a finite real"),
            ({"K_critical": False}, "K_critical must be a finite real"),
        ],
    )
    def test_rejects_invalid_diagram_record_values(
        self, kwargs: dict[str, object], match: str
    ) -> None:
        """Malformed diagram lists and critical-value aliases are refused."""
        base: dict[str, object] = {"points": [], "K_critical": None}
        base.update(kwargs)
        with pytest.raises(ValueError, match=match):
            BifurcationDiagram(
                points=cast("list[BifurcationPoint]", base.get("points", [])),
                K_critical=cast("float | None", base.get("K_critical")),
            )


class TestTraceSyncTransition:
    """Original public independent coupling sweeps and threshold crossings."""

    def test_returns_diagram(self) -> None:
        """The original public sweep returns the requested sampled diagram."""
        N = 8
        rng = np.random.default_rng(42)
        omegas = rng.normal(0, 0.5, N)
        diag = trace_sync_transition(
            omegas,
            K_range=(0.0, 3.0),
            n_points=10,
            n_transient=200,
            n_measure=100,
        )
        assert isinstance(diag, BifurcationDiagram)
        assert len(diag.points) == 10

    def test_R_increases_with_K(self) -> None:
        """The specified network has a larger response at the upper grid endpoint."""
        N = 8
        rng = np.random.default_rng(0)
        omegas = rng.normal(0, 0.3, N)
        diag = trace_sync_transition(
            omegas,
            K_range=(0.0, 5.0),
            n_points=8,
            n_transient=200,
            n_measure=100,
        )
        R_first = diag.R_values[:3].mean()
        R_last = diag.R_values[-3:].mean()
        assert R_last > R_first

    def test_finds_K_critical(self) -> None:
        """The sampled response returns a bounded crossing when one is detected."""
        N = 8
        rng = np.random.default_rng(42)
        omegas = rng.standard_cauchy(N) * 0.5
        omegas = np.clip(omegas, -5, 5)
        diag = trace_sync_transition(
            omegas,
            K_range=(0.0, 4.0),
            n_points=8,
            n_transient=200,
            n_measure=100,
        )
        if diag.K_critical is not None:
            assert 0.1 < diag.K_critical < 4.0


class TestTraceSyncTransitionCoverage:
    """Validate K_critical interpolation when the response crosses threshold."""

    def test_k_critical_interpolation(self) -> None:
        """Force a clear R threshold crossing to exercise interpolation."""
        omegas = np.linspace(-1, 1, 10)
        diag = trace_sync_transition(
            omegas,
            K_range=(0.0, 8.0),
            n_points=20,
            n_transient=500,
            n_measure=200,
        )
        if diag.K_critical is not None:
            assert 0.0 < diag.K_critical < 8.0

    def test_no_crossing_k_critical_none(self) -> None:
        """Very low K range + wide ω → R stays below threshold → K_critical None."""
        omegas = np.linspace(-50, 50, 6)
        diag = trace_sync_transition(
            omegas,
            K_range=(0.0, 0.01),
            n_points=5,
            n_transient=100,
            n_measure=50,
        )
        assert diag.K_critical is None

    def test_custom_knm_template(self) -> None:
        """An admitted custom graph retains the requested grid cardinality."""
        n = 4
        omegas = np.linspace(-0.5, 0.5, n)
        knm = np.ones((n, n)) * 0.5
        np.fill_diagonal(knm, 0.0)
        diag = trace_sync_transition(
            omegas,
            knm_template=knm,
            K_range=(0.0, 5.0),
            n_points=5,
            n_transient=100,
            n_measure=50,
        )
        assert len(diag.points) == 5

    def test_trace_rejects_self_coupling_diagonal(self) -> None:
        """K_ii is not a physical pair interaction in the Kuramoto graph."""
        n = 4
        omegas = np.linspace(-0.5, 0.5, n)
        knm = np.ones((n, n)) * 0.5
        knm[1, 1] = 0.25

        with pytest.raises(ValueError, match="diagonal must be zero"):
            trace_sync_transition(
                omegas,
                knm_template=knm,
                K_range=(0.0, 2.0),
                n_points=3,
                n_transient=0,
                n_measure=1,
            )

    def test_r_values_bounded(self) -> None:
        """Original public sweep measurements remain within the unit interval."""
        omegas = np.linspace(-1, 1, 6)
        diag = trace_sync_transition(
            omegas,
            K_range=(0.0, 5.0),
            n_points=8,
            n_transient=200,
            n_measure=100,
        )
        assert np.all(diag.R_values >= 0)
        assert np.all(diag.R_values <= 1.0 + 1e-6)

    def test_python_fallback_interpolates_threshold_crossing(self) -> None:
        """Two-oscillator algebra determines the first interpolated upcrossing."""
        phases = np.random.default_rng(92).uniform(0, 2 * np.pi, 2)
        delta = float(phases[1] - phases[0])
        grid = np.linspace(0.0, 1.0, 3)
        expected = np.array(
            [abs(np.cos((delta - 2 * scale * np.sin(delta)) / 2)) for scale in grid]
        )
        diagram = trace_sync_transition(
            np.zeros(2),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            K_range=(0.0, 1.0),
            n_points=3,
            dt=1.0,
            n_transient=0,
            n_measure=1,
            seed=92,
            backend="python",
        )
        np.testing.assert_allclose(diagram.R_values, expected, atol=2e-15)
        assert expected[0] < 0.1 < expected[1]
        crossing = 0.5 * (0.1 - expected[0]) / (expected[1] - expected[0])
        assert diagram.K_critical == pytest.approx(crossing, abs=2e-15)


class TestInputValidation:
    """Public numerical domains reject malformed shapes and coercive aliases."""

    def test_array_guard_rejects_failed_array_protocol(self) -> None:
        """A deliberately failed conversion hook cannot become numerical evidence."""

        class FailedArray:
            def __array__(
                self, dtype: object = None, copy: object = None
            ) -> FloatArray:
                raise TypeError("unavailable array payload")

        with pytest.raises(ValueError, match="probe must be a numeric array"):
            bif._as_real_numeric_array(FailedArray(), name="probe")

    def test_array_guard_rejects_float_conversion_overflow(self) -> None:
        """An unrepresentable integer cannot become a finite numerical array."""
        huge_integer = np.array([10**1000], dtype=object)

        with pytest.raises(ValueError, match="probe must be a numeric array"):
            bif._as_real_numeric_array(huge_integer, name="probe")

    def test_array_guard_preserves_real_numeric_object_arrays(self) -> None:
        """The original public graph accepts finite real object-valued arrays."""
        diagram = trace_sync_transition(
            np.array([np.float32(0.25), 2, -0.5], dtype=object),
            n_points=2,
            n_transient=0,
            n_measure=1,
            backend="python",
        )
        assert len(diagram.points) == 2
        assert np.all(np.isfinite(diagram.R_values))

    def test_trace_rejects_empty_frequency_sample(self) -> None:
        """A public coupling grid requires at least one oscillator."""
        with pytest.raises(ValueError, match="at least one oscillator"):
            trace_sync_transition(np.array([], dtype=np.float64))

    def test_trace_rejects_non_finite_matrix(self) -> None:
        """Nonfinite custom coupling entries are rejected before trial execution."""
        knm = np.zeros((2, 2), dtype=np.float64)
        knm[0, 1] = np.inf

        with pytest.raises(ValueError, match="only finite values"):
            trace_sync_transition(np.zeros(2), knm_template=knm)

    def test_trace_rejects_non_tuple_k_range(self) -> None:
        """The public grid range retains its two-value tuple contract."""
        with pytest.raises(ValueError, match="exactly two finite values"):
            trace_sync_transition(
                np.zeros(2), K_range=cast("tuple[float, float]", [0.0, 1.0])
            )

    @pytest.mark.parametrize(
        ("field", "bad_value", "match"),
        [
            ("omegas", np.zeros((3, 1), dtype=np.float64), "omegas shape"),
            ("omegas", np.array([0.0, np.nan], dtype=np.float64), "omegas"),
            ("omegas", np.array([True, False, True]), "omegas"),
            ("knm_template", np.zeros((3, 2), dtype=np.float64), "knm_template"),
            ("knm_template", np.eye(3, dtype=bool), "knm_template"),
            ("alpha", np.zeros((2, 3), dtype=np.float64), "alpha"),
            ("alpha", np.eye(3, dtype=bool), "alpha"),
        ],
    )
    def test_trace_rejects_invalid_arrays(
        self,
        field: str,
        bad_value: FloatArray,
        match: str,
    ) -> None:
        """Malformed graph, lag and frequency shapes fail public ingress."""
        kwargs = {
            "omegas": np.ones(3, dtype=np.float64),
            "knm_template": np.zeros((3, 3), dtype=np.float64),
            "alpha": np.zeros((3, 3), dtype=np.float64),
            "n_points": 2,
            "n_transient": 1,
            "n_measure": 1,
        }
        kwargs[field] = bad_value

        with pytest.raises(ValueError, match=match):
            trace_sync_transition(
                omegas=cast("FloatArray", kwargs["omegas"]),
                knm_template=cast("FloatArray | None", kwargs.get("knm_template")),
                alpha=cast("FloatArray | None", kwargs.get("alpha")),
                K_range=cast("tuple[float,float]", kwargs.get("K_range", (0.0, 5.0))),
                n_points=cast("int", kwargs.get("n_points", 50)),
                dt=cast("float", kwargs.get("dt", 0.01)),
                n_transient=cast("int", kwargs.get("n_transient", 2000)),
                n_measure=cast("int", kwargs.get("n_measure", 500)),
                seed=cast("int", kwargs.get("seed", 42)),
                backend=cast("str | None", kwargs.get("backend")),
            )

    @pytest.mark.parametrize(
        ("field", "bad_value", "match"),
        [
            ("omegas", np.array(["-0.2", "0.0", "0.2"]), "numeric"),
            (
                "omegas",
                np.array([-0.2 + 1.0j, 0.0, 0.2 - 2.0j]),
                "real-valued",
            ),
            (
                "knm_template",
                np.array([["0", "1", "1"], ["1", "0", "1"], ["1", "1", "0"]]),
                "numeric",
            ),
            (
                "knm_template",
                np.array(
                    [[0.0, 1.0j, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
                    dtype=object,
                ),
                "real-valued",
            ),
            (
                "alpha",
                np.array([["0", "0", "0"], ["0", "0", "0"], ["0", "0", "0"]]),
                "numeric",
            ),
            (
                "alpha",
                np.array(
                    [[0.0, 1.0j, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    dtype=object,
                ),
                "real-valued",
            ),
        ],
    )
    def test_trace_rejects_coercive_array_aliases_before_dispatch(
        self,
        monkeypatch: pytest.MonkeyPatch,
        field: str,
        bad_value: FloatArray,
        match: str,
    ) -> None:
        """Boolean, text and complex payloads cannot reach the trial dispatcher."""
        monkeypatch.setattr(bif, "_HAS_COMPOSITE_RUST", False)
        monkeypatch.setattr(
            bif,
            "_steady_state_R_dispatch",
            lambda *_args, **_kwargs: pytest.fail("invalid input reached dispatch"),
        )
        kwargs: dict[str, object] = {
            "omegas": np.array([-0.2, 0.0, 0.2]),
            "knm_template": np.zeros((3, 3), dtype=np.float64),
            "alpha": np.zeros((3, 3), dtype=np.float64),
            "n_points": 2,
            "n_transient": 0,
            "n_measure": 1,
        }
        kwargs[field] = bad_value

        with pytest.raises(ValueError, match=match):
            trace_sync_transition(
                omegas=cast("FloatArray", kwargs["omegas"]),
                knm_template=cast("FloatArray | None", kwargs.get("knm_template")),
                alpha=cast("FloatArray | None", kwargs.get("alpha")),
                K_range=cast("tuple[float,float]", kwargs.get("K_range", (0.0, 5.0))),
                n_points=cast("int", kwargs.get("n_points", 50)),
                dt=cast("float", kwargs.get("dt", 0.01)),
                n_transient=cast("int", kwargs.get("n_transient", 2000)),
                n_measure=cast("int", kwargs.get("n_measure", 500)),
                seed=cast("int", kwargs.get("seed", 42)),
                backend=cast("str | None", kwargs.get("backend")),
            )

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("K_range", (-0.1, 1.0)),
            ("K_range", (1.0, 1.0)),
            ("K_range", (0.0, np.nan)),
            ("n_points", False),
            ("n_points", 1),
            ("n_points", 2.5),
            ("dt", 0.0),
            ("dt", np.inf),
            ("n_transient", -1),
            ("n_measure", -1),
            ("seed", False),
            ("seed", 1.5),
        ],
    )
    def test_trace_rejects_invalid_runtime_parameters(
        self,
        field: str,
        bad_value: object,
    ) -> None:
        """Invalid grid bounds, counts, timestep and seed aliases are refused."""
        kwargs = {
            "K_range": (0.0, 1.0),
            "n_points": 2,
            "dt": 0.01,
            "n_transient": 1,
            "n_measure": 1,
            "seed": 1,
        }
        kwargs[field] = bad_value

        with pytest.raises(ValueError, match=field):
            trace_sync_transition(
                np.ones(3, dtype=np.float64),
                knm_template=cast("FloatArray | None", kwargs.get("knm_template")),
                alpha=cast("FloatArray | None", kwargs.get("alpha")),
                K_range=cast("tuple[float,float]", kwargs.get("K_range", (0.0, 5.0))),
                n_points=cast("int", kwargs.get("n_points", 50)),
                dt=cast("float", kwargs.get("dt", 0.01)),
                n_transient=cast("int", kwargs.get("n_transient", 2000)),
                n_measure=cast("int", kwargs.get("n_measure", 500)),
                seed=cast("int", kwargs.get("seed", 42)),
                backend=cast("str | None", kwargs.get("backend")),
            )

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("omegas", np.zeros((3, 1), dtype=np.float64)),
            ("omegas", np.array([True, False, True])),
            ("knm_template", np.zeros((3, 2), dtype=np.float64)),
            ("knm_template", np.eye(3, dtype=bool)),
            ("dt", 0.0),
            ("n_transient", -1),
            ("n_measure", -1),
            ("tol", 0.0),
            ("seed", False),
        ],
    )
    def test_find_rejects_invalid_contract(
        self,
        field: str,
        bad_value: object,
    ) -> None:
        """Invalid graph and search controls fail before numerical bisection."""
        kwargs = {
            "omegas": np.ones(3, dtype=np.float64),
            "knm_template": np.zeros((3, 3), dtype=np.float64),
            "dt": 0.01,
            "n_transient": 1,
            "n_measure": 1,
            "tol": 0.1,
            "seed": 1,
        }
        kwargs[field] = bad_value
        omegas = cast(FloatArray, kwargs.pop("omegas"))

        with pytest.raises(ValueError, match=field):
            find_critical_coupling(
                omegas,
                knm_template=cast("FloatArray | None", kwargs.get("knm_template")),
                dt=cast("float", kwargs.get("dt", 0.01)),
                n_transient=cast("int", kwargs.get("n_transient", 3000)),
                n_measure=cast("int", kwargs.get("n_measure", 1000)),
                tol=cast("float", kwargs.get("tol", 0.05)),
                seed=cast("int", kwargs.get("seed", 42)),
                backend=cast("str | None", kwargs.get("backend")),
            )

    @pytest.mark.parametrize(
        ("field", "bad_value", "match"),
        [
            ("omegas", np.array(["-0.2", "0.0", "0.2"]), "numeric"),
            (
                "omegas",
                np.array([-0.2, complex(0.0, 1.0), 0.2], dtype=object),
                "real-valued",
            ),
            (
                "knm_template",
                np.array([["0", "1", "1"], ["1", "0", "1"], ["1", "1", "0"]]),
                "numeric",
            ),
            (
                "knm_template",
                np.array(
                    [[0.0, 1.0j, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
                    dtype=object,
                ),
                "real-valued",
            ),
        ],
    )
    def test_find_rejects_coercive_array_aliases_before_dispatch(
        self,
        monkeypatch: pytest.MonkeyPatch,
        field: str,
        bad_value: FloatArray,
        match: str,
    ) -> None:
        """Coercive numerical aliases cannot reach the threshold-search trials."""
        monkeypatch.setattr(bif, "_HAS_COMPOSITE_RUST", False)
        monkeypatch.setattr(
            bif,
            "_steady_state_R_dispatch",
            lambda *_args, **_kwargs: pytest.fail("invalid input reached dispatch"),
        )
        kwargs: dict[str, object] = {
            "omegas": np.array([-0.2, 0.0, 0.2]),
            "knm_template": np.zeros((3, 3), dtype=np.float64),
            "n_transient": 0,
            "n_measure": 1,
        }
        kwargs[field] = bad_value
        omegas = cast(FloatArray, kwargs.pop("omegas"))

        with pytest.raises(ValueError, match=match):
            find_critical_coupling(
                omegas,
                knm_template=cast("FloatArray | None", kwargs.get("knm_template")),
                dt=cast("float", kwargs.get("dt", 0.01)),
                n_transient=cast("int", kwargs.get("n_transient", 3000)),
                n_measure=cast("int", kwargs.get("n_measure", 1000)),
                tol=cast("float", kwargs.get("tol", 0.05)),
                seed=cast("int", kwargs.get("seed", 42)),
                backend=cast("str | None", kwargs.get("backend")),
            )

    def test_find_rejects_self_coupling_diagonal(self) -> None:
        """Binary-search K_c uses the same zero-self-coupling graph contract."""
        knm = np.zeros((3, 3), dtype=np.float64)
        knm[0, 0] = 0.1

        with pytest.raises(ValueError, match="diagonal must be zero"):
            find_critical_coupling(
                np.array([-0.2, 0.0, 0.2]),
                knm_template=knm,
                n_transient=0,
                n_measure=1,
            )


class TestFindCriticalCoupling:
    """Original threshold searches, finite windows and bounded iterations."""

    def test_returns_finite(self) -> None:
        """The specified finite frequency sample produces a finite search estimate."""
        N = 8
        rng = np.random.default_rng(42)
        omegas = rng.normal(0, 0.3, N)
        Kc = find_critical_coupling(omegas, n_transient=200, n_measure=100, tol=0.15)
        assert np.isfinite(Kc)
        assert Kc > 0

    def test_no_transition(self) -> None:
        """Identical frequencies → R=1 at any K>0, K_c ≈ 0."""
        N = 8
        omegas = np.zeros(N)
        Kc = find_critical_coupling(
            cast(FloatArray, omegas), n_transient=100, n_measure=50, tol=0.15
        )
        assert np.isfinite(Kc)
        assert Kc < 1.0

    def test_wide_spread_returns_float(self) -> None:
        """Wide ω spread → K_c either NaN or large positive."""
        n = 8
        omegas = np.linspace(-100, 100, n)
        Kc = find_critical_coupling(
            omegas,
            n_transient=100,
            n_measure=50,
            tol=1.0,
        )
        assert isinstance(Kc, float)
        if not np.isnan(Kc):
            assert Kc >= 0

    def test_binary_search_converges(self) -> None:
        """Moderate ω spread → binary search finds K_c in range."""
        omegas = np.linspace(-2, 2, 8)
        Kc = find_critical_coupling(
            omegas,
            n_transient=500,
            n_measure=200,
            tol=0.5,
        )
        if not np.isnan(Kc):
            assert 0.0 < Kc < 20.0

    def test_measurement_window_zero_returns_nan(self) -> None:
        """A real empty window has R=0 at the upper endpoint and no crossing."""
        assert np.isnan(
            find_critical_coupling(
                np.array([0.1, -0.2, 0.3]), n_transient=0, n_measure=0, backend="python"
            )
        )

    def test_default_knm(self) -> None:
        """An omitted graph uses the original default coupling template."""
        omegas = np.array([1.0, 2.0, 3.0, 4.0])
        Kc = find_critical_coupling(
            omegas,
            knm_template=None,
            n_transient=200,
            n_measure=100,
            tol=1.0,
        )
        assert isinstance(Kc, float)

    def test_returns_nan_when_upper_bound_remains_subcritical(self) -> None:
        """A fixed disconnected antiphase sample remains below threshold."""
        with cProfile.Profile() as profile:
            critical = find_critical_coupling(
                np.zeros(2),
                np.zeros((2, 2)),
                n_transient=0,
                n_measure=1,
                seed=92,
                backend="python",
            )
        profile.create_stats()
        calls = [
            item[1]
            for key, item in profile.stats.items()
            if key[2] == "_python_steady_state_r"
        ]
        assert np.isnan(critical)
        assert calls == [1]

    def test_binary_search_moves_lower_bound_after_subcritical_midpoint(self) -> None:
        """Actual one-step algebra places K=10 below and K=15 above R=0.1."""
        p = np.random.default_rng(92).uniform(0, 2 * np.pi, 2)
        delta = float(p[1] - p[0])
        assert abs(np.cos((delta - 0.2 * np.sin(delta)) / 2)) < 0.1
        assert abs(np.cos((delta - 0.3 * np.sin(delta)) / 2)) > 0.1
        critical = find_critical_coupling(
            np.zeros(2),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            dt=0.01,
            n_transient=0,
            n_measure=1,
            tol=6.0,
            seed=92,
            backend="python",
        )
        assert critical == 12.5

    def test_binary_search_honours_the_iteration_cap(self) -> None:
        """A real singleton R=1 stops at 30 bisections despite tiny tolerance."""
        with cProfile.Profile() as profile:
            critical = find_critical_coupling(
                np.zeros(1), n_transient=0, n_measure=1, tol=1e-20, backend="python"
            )
        profile.create_stats()
        calls = [
            item[1]
            for key, item in profile.stats.items()
            if key[2] == "_python_steady_state_r"
        ]
        assert calls == [31]
        assert critical == 20.0 / (2**31)


class TestBifurcationDispatchSurface:
    """Original delegated inputs agree with independent Euler measurements."""

    def test_python_path_forwards_kernel_inputs(self) -> None:
        """Signed directed lagged public inputs match independent scalar trials."""
        omegas = np.array([-0.2, 0.1, 0.4])
        knm = np.array([[0.0, 0.2, 0.3], [0.4, 0.0, 0.5], [0.2, 0.4, 0.0]])
        alpha = np.full((3, 3), 0.05)
        phases = np.random.default_rng(7).uniform(0, 2 * np.pi, 3)
        diagram = trace_sync_transition(
            omegas,
            knm,
            alpha,
            K_range=(0.0, 4.0),
            n_points=4,
            dt=0.03,
            n_transient=10,
            n_measure=5,
            seed=7,
            backend="python",
        )
        expected = [
            scalar_trial(
                phases.tolist(),
                omegas.tolist(),
                knm.tolist(),
                alpha.tolist(),
                scale=float(scale),
                dt=0.03,
                transient=10,
                measure=5,
            )
            for scale in np.linspace(0.0, 4.0, 4)
        ]
        np.testing.assert_allclose(diagram.R_values, expected, atol=2e-14, rtol=2e-14)


class TestBifurcationPublicMeasurements:
    """Pipeline: bifurcation analysis uses UPDEEngine internally."""

    def test_trace_sync_returns_real_grid_measurements(self) -> None:
        """The original public sweep returns finite measurements at every grid point."""
        omegas = np.array([1.0, 1.5, 2.0, 0.5])
        diag = trace_sync_transition(
            omegas,
            K_range=(0.1, 2.0),
            n_points=5,
        )
        assert isinstance(diag, BifurcationDiagram)
        assert len(diag.points) == 5
        for pt in diag.points:
            assert isinstance(pt, BifurcationPoint)
            assert 0.0 <= pt.R <= 1.0
            assert pt.K >= 0.0
