# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Algorithmic tests for basin stability

"""Algorithmic properties of ``upde.basin_stability``.

Covered: physics limits (high K → high S_B, zero K → low S_B,
lower R_threshold yields higher S_B); bounds 0 ≤ S_B ≤ 1 and
0 ≤ R_final ≤ 1; threshold classification consistency;
``multi_basin_stability`` keys and monotonicity; deterministic
result for identical seed; order-parameter helper via
``steady_state_r`` with locked initial phases;
Hypothesis-driven invariants.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde import basin_stability as b_mod
from scpn_phase_orchestrator.upde.basin_stability import (
    BasinStabilityResult,
    basin_stability,
    multi_basin_stability,
    steady_state_r,
)

FloatArray = NDArray[np.float64]

TWO_PI = 2.0 * math.pi


def test_named_rust_preserves_noncontiguous_target_source_inputs() -> None:
    """The original Rust boundary preserves logical order when copying strided views."""
    phases = np.array([0.0, 99.0, 1.2, 99.0])[::2]
    omega = np.array([0.2, 99.0, -0.1, 99.0])[::2]
    graph = np.array([[0.0, 0.0], [-0.7, 0.0]]).T
    expected = abs(math.cos((1.2 - 0.01 * (0.3 - 0.7 * math.sin(1.2))) / 2))
    if "rust" not in b_mod.AVAILABLE_BACKENDS:
        with pytest.raises(ImportError, match="requested basin backend 'rust'"):
            steady_state_r(
                phases, omega, graph, n_transient=0, n_measure=1, backend="rust"
            )
        return
    actual = steady_state_r(
        phases, omega, graph, n_transient=0, n_measure=1, backend="rust"
    )
    assert actual == pytest.approx(expected, abs=2e-15)
    np.testing.assert_array_equal(phases, [0.0, 1.2])


def test_named_optional_owners_execute_or_report_actual_absence() -> None:
    """Each optional owner supplies its original numerical trial or actual refusal."""
    phases = np.array([0.0, math.pi / 2])
    graph = np.array([[0.0, 5e-31], [5e-31, 0.0]])
    expected = math.cos((math.pi / 2 - 1) / 2)
    for owner in ("rust", "go", "julia", "mojo"):
        if owner not in b_mod.AVAILABLE_BACKENDS:
            with pytest.raises(ImportError, match="requested basin backend"):
                steady_state_r(
                    phases,
                    np.zeros(2),
                    graph,
                    dt=1e30,
                    n_transient=0,
                    n_measure=1,
                    backend=owner,
                )
        else:
            actual = steady_state_r(
                phases,
                np.zeros(2),
                graph,
                dt=1e30,
                n_transient=0,
                n_measure=1,
                backend=owner,
            )
            assert actual == pytest.approx(expected, abs=2e-15)


def _all_to_all(n: int, strength: float = 1.0) -> FloatArray:
    k = np.ones((n, n)) * strength / n
    np.fill_diagonal(k, 0.0)
    return k


class TestSteadyStateRKernel:
    """Original Python trial windows under specified finite network inputs."""

    def test_locked_initial_gives_r_near_one_at_strong_coupling(self) -> None:
        """The specified initially locked population retains high finite-window R."""
        n = 8
        omegas = np.ones(n) * 0.5
        knm = _all_to_all(n, strength=10.0)
        phases = np.full(n, 1.3)
        r = steady_state_r(
            phases,
            omegas,
            knm,
            dt=0.01,
            n_transient=200,
            n_measure=100,
            backend="python",
        )
        assert r > 0.99

    def test_zero_coupling_desynchronises(self) -> None:
        """The specified uncoupled frequencies have a low measured window average."""
        n = 6
        omegas = np.array([1.0, 1.7, 2.3, 3.1, 4.2, 5.5])
        knm = np.zeros((n, n))
        phases = np.linspace(0.0, TWO_PI, n, endpoint=False)
        r = steady_state_r(
            phases,
            omegas,
            knm,
            dt=0.01,
            n_transient=500,
            n_measure=500,
            backend="python",
        )
        assert r < 0.9

    def test_zero_measure_returns_zero(self) -> None:
        """The explicit Python owner retains its zero-window convention."""
        n = 3
        omegas = np.ones(n)
        knm = _all_to_all(n)
        phases = np.zeros(n)
        r = steady_state_r(
            phases,
            omegas,
            knm,
            dt=0.01,
            n_transient=10,
            n_measure=0,
            backend="python",
        )
        assert r == 0.0

    def test_r_in_bounds(self) -> None:
        """Actual finite trial measurements remain within the unit interval."""
        n = 5
        omegas = np.linspace(0.5, 2.0, n)
        knm = _all_to_all(n, strength=2.0)
        rng = np.random.default_rng(1)
        phases = rng.uniform(0, TWO_PI, n)
        r = steady_state_r(
            phases,
            omegas,
            knm,
            dt=0.01,
            n_transient=200,
            n_measure=100,
            backend="python",
        )
        assert 0.0 <= r <= 1.0 + 1e-12


class TestBasinStability:
    """Public finite-window sampling fractions, counts and reproducibility."""

    def test_strong_coupling_high_sb(self) -> None:
        """The specified strongly coupled population passes most sampled trials."""
        n = 5
        omegas = np.ones(n)
        knm = _all_to_all(n, strength=10.0)
        result = basin_stability(
            omegas,
            knm,
            dt=0.01,
            n_transient=300,
            n_measure=100,
            n_samples=20,
            R_threshold=0.8,
            seed=42,
            backend="python",
        )
        assert isinstance(result, BasinStabilityResult)
        assert result.S_B > 0.5
        assert result.n_samples == 20
        assert 0 <= result.n_converged <= 20

    def test_zero_coupling_low_sb(self) -> None:
        """The specified uncoupled population passes few sampled trials."""
        n = 6
        omegas = np.linspace(0.5, 3.5, n)
        knm = np.zeros((n, n))
        result = basin_stability(
            omegas,
            knm,
            dt=0.01,
            n_transient=300,
            n_measure=100,
            n_samples=15,
            R_threshold=0.9,
            seed=7,
            backend="python",
        )
        assert result.S_B <= 0.2

    def test_bounds(self) -> None:
        """Actual sampled fractions, counts and trial values remain bounded."""
        n = 4
        omegas = np.ones(n)
        knm = _all_to_all(n, strength=2.0)
        result = basin_stability(
            omegas,
            knm,
            dt=0.01,
            n_transient=200,
            n_measure=100,
            n_samples=10,
            R_threshold=0.5,
            seed=3,
            backend="python",
        )
        assert 0.0 <= result.S_B <= 1.0
        assert result.R_final.shape == (10,)
        assert np.all(result.R_final >= 0.0)
        assert np.all(result.R_final <= 1.0 + 1e-12)

    def test_deterministic_same_seed(self) -> None:
        """Repeated public sampling with the same seed produces identical trials."""
        n = 4
        omegas = np.array([1.0, 1.5, 2.0, 2.5])
        knm = _all_to_all(n, strength=3.0)
        r1 = basin_stability(
            omegas,
            knm,
            dt=0.01,
            n_transient=150,
            n_measure=80,
            n_samples=10,
            R_threshold=0.5,
            seed=123,
            backend="python",
        )
        r2 = basin_stability(
            omegas,
            knm,
            dt=0.01,
            n_transient=150,
            n_measure=80,
            n_samples=10,
            R_threshold=0.5,
            seed=123,
            backend="python",
        )
        assert r1.S_B == r2.S_B
        assert r1.n_converged == r2.n_converged
        np.testing.assert_array_equal(r1.R_final, r2.R_final)

    def test_lower_threshold_higher_sb(self) -> None:
        """Lowering the inclusive threshold cannot remove passing trials."""
        n = 4
        omegas = np.array([1.0, 2.0, 3.0, 4.0])
        knm = _all_to_all(n, strength=3.0)
        r_low = basin_stability(
            omegas,
            knm,
            dt=0.01,
            n_transient=200,
            n_measure=100,
            n_samples=15,
            R_threshold=0.3,
            seed=42,
            backend="python",
        )
        r_high = basin_stability(
            omegas,
            knm,
            dt=0.01,
            n_transient=200,
            n_measure=100,
            n_samples=15,
            R_threshold=0.9,
            seed=42,
            backend="python",
        )
        assert r_low.S_B >= r_high.S_B


class TestMultiBasinStability:
    """Shared trial values classified at multiple inclusive thresholds."""

    def test_keys_and_monotonicity(self) -> None:
        """Established labels preserve monotone threshold-classification fractions."""
        n = 4
        omegas = np.ones(n)
        knm = _all_to_all(n, strength=4.0)
        results = multi_basin_stability(
            omegas,
            knm,
            dt=0.01,
            n_transient=150,
            n_measure=80,
            n_samples=12,
            R_thresholds=(0.3, 0.6, 0.9),
            seed=5,
            backend="python",
        )
        assert set(results.keys()) == {"R>=0.30", "R>=0.60", "R>=0.90"}
        sb_vals = [results[k].S_B for k in ("R>=0.30", "R>=0.60", "R>=0.90")]
        assert sb_vals[0] >= sb_vals[1] >= sb_vals[2]

    def test_shared_r_finals(self) -> None:
        """All threshold entries must share the same R_final array."""
        n = 4
        omegas = np.ones(n)
        knm = _all_to_all(n, strength=3.0)
        results = multi_basin_stability(
            omegas,
            knm,
            dt=0.01,
            n_transient=120,
            n_measure=60,
            n_samples=8,
            R_thresholds=(0.3, 0.8),
            seed=11,
            backend="python",
        )
        r1 = results["R>=0.30"].R_final
        r2 = results["R>=0.80"].R_final
        np.testing.assert_array_equal(r1, r2)


class TestHypothesis:
    """Sampled invariant checks on the original public estimator."""

    @given(
        n=st.integers(min_value=2, max_value=6),
        strength=st.floats(min_value=0.5, max_value=5.0),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=8,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_sb_in_unit_interval(self, n: int, strength: float, seed: int) -> None:
        """Actual sampled fractions remain within the unit interval."""
        omegas = np.ones(n)
        knm = _all_to_all(n, strength=strength)
        result = basin_stability(
            omegas,
            knm,
            dt=0.01,
            n_transient=80,
            n_measure=40,
            n_samples=6,
            R_threshold=0.5,
            seed=seed,
            backend="python",
        )
        assert 0.0 <= result.S_B <= 1.0
        assert np.all(np.isfinite(result.R_final))


class TestDispatcherSurface:
    """Declared preference retains the Python owner as its final option."""

    def test_available_non_empty(self) -> None:
        """Discovery retains at least the always available Python owner."""
        assert b_mod.AVAILABLE_BACKENDS
        assert "python" in b_mod.AVAILABLE_BACKENDS

    def test_active_is_first(self) -> None:
        """The automatic owner agrees with the declared discovery preference."""
        assert b_mod.AVAILABLE_BACKENDS[0] == b_mod.ACTIVE_BACKEND


class TestInputShapes:
    """Default lag and empty-sample public input conventions."""

    def test_default_alpha_zeros(self) -> None:
        """Omitting lag produces the same trials as an explicit zero lag matrix."""
        n = 3
        omegas = np.ones(n)
        knm = _all_to_all(n)
        r_no_alpha = basin_stability(
            omegas,
            knm,
            dt=0.01,
            n_transient=60,
            n_measure=30,
            n_samples=4,
            R_threshold=0.5,
            seed=1,
            backend="python",
        )
        r_zero_alpha = basin_stability(
            omegas,
            knm,
            alpha=np.zeros((n, n)),
            dt=0.01,
            n_transient=60,
            n_measure=30,
            n_samples=4,
            R_threshold=0.5,
            seed=1,
            backend="python",
        )
        np.testing.assert_allclose(
            r_no_alpha.R_final,
            r_zero_alpha.R_final,
            atol=1e-15,
        )

    def test_zero_samples_returns_empty(self) -> None:
        """The original empty-sample convention returns no trial measurements."""
        n = 3
        omegas = np.ones(n)
        knm = _all_to_all(n)
        result = basin_stability(
            omegas,
            knm,
            dt=0.01,
            n_transient=20,
            n_measure=10,
            n_samples=0,
            R_threshold=0.5,
            seed=1,
            backend="python",
        )
        assert result.S_B == 0.0
        assert result.n_samples == 0
        assert result.n_converged == 0
        assert result.R_final.shape == (0,)


class TestInputValidation:
    """Public numerical domains reject malformed shapes and coercive aliases."""

    @pytest.mark.parametrize(
        ("field", "bad_value", "match"),
        [
            ("phases_init", np.zeros((3, 1), dtype=np.float64), "phases_init"),
            ("omegas", np.zeros((4,), dtype=np.float64), "omegas shape"),
            ("knm", np.zeros((3, 2), dtype=np.float64), "knm shape"),
            ("alpha", np.zeros((2, 3), dtype=np.float64), "alpha shape"),
        ],
    )
    def test_steady_state_rejects_shape_mismatch(
        self,
        field: str,
        bad_value: FloatArray,
        match: str,
    ) -> None:
        """Public trial dimensions are checked before selecting a numerical owner."""
        values = {
            "phases_init": np.zeros(3, dtype=np.float64),
            "omegas": np.ones(3, dtype=np.float64),
            "knm": np.zeros((3, 3), dtype=np.float64),
            "alpha": np.zeros((3, 3), dtype=np.float64),
        }
        values[field] = bad_value

        with pytest.raises(ValueError, match=match):
            steady_state_r(
                values["phases_init"],
                values["omegas"],
                values["knm"],
                alpha=values["alpha"],
                n_transient=1,
                n_measure=1,
            )

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("phases_init", np.nan),
            ("omegas", np.inf),
            ("knm", np.nan),
            ("alpha", np.inf),
        ],
    )
    def test_steady_state_rejects_non_finite_arrays(
        self,
        field: str,
        bad_value: float,
    ) -> None:
        """Nonfinite public phase, frequency, graph and lag arrays are refused."""
        phases_init = np.zeros(3, dtype=np.float64)
        omegas = np.ones(3, dtype=np.float64)
        knm = np.zeros((3, 3), dtype=np.float64)
        alpha = np.zeros((3, 3), dtype=np.float64)
        if field in {"knm", "alpha"}:
            locals()[field][0, 1] = bad_value
        else:
            locals()[field][0] = bad_value

        with pytest.raises(ValueError, match=field):
            steady_state_r(
                phases_init,
                omegas,
                knm,
                alpha=alpha,
                n_transient=1,
                n_measure=1,
            )

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("k_scale", False),
            ("k_scale", np.nan),
            ("k_scale", np.inf),
            ("k_scale", "1.0"),
            ("dt", False),
            ("dt", 0.0),
            ("dt", np.nan),
            ("n_transient", False),
            ("n_transient", -1),
            ("n_transient", 1.5),
            ("n_measure", False),
            ("n_measure", -1),
            ("n_measure", "1"),
        ],
    )
    def test_steady_state_rejects_invalid_runtime_parameters(
        self,
        field: str,
        bad_value: object,
    ) -> None:
        """Invalid trial scales, steps and count aliases are refused."""
        kwargs: dict[str, object] = {
            "k_scale": 1.0,
            "dt": 0.01,
            "n_transient": 1,
            "n_measure": 1,
        }
        kwargs[field] = bad_value

        with pytest.raises(ValueError, match=field):
            steady_state_r(
                np.zeros(3, dtype=np.float64),
                np.ones(3, dtype=np.float64),
                np.zeros((3, 3), dtype=np.float64),
                alpha=cast("FloatArray | None", kwargs.get("alpha")),
                k_scale=cast("float", kwargs.get("k_scale", 1.0)),
                dt=cast("float", kwargs.get("dt", 0.01)),
                n_transient=cast("int", kwargs.get("n_transient", 500)),
                n_measure=cast("int", kwargs.get("n_measure", 200)),
                backend=cast("str | None", kwargs.get("backend")),
            )

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("dt", 0.0),
            ("n_transient", -1),
            ("n_measure", -1),
            ("n_samples", -1),
            ("R_threshold", np.nan),
            ("R_threshold", 1.5),
            ("seed", False),
            ("seed", 1.5),
        ],
    )
    def test_basin_stability_rejects_invalid_runtime_parameters(
        self,
        field: str,
        bad_value: object,
    ) -> None:
        """Invalid sampling controls cannot enter the Monte Carlo loop."""
        kwargs: dict[str, object] = {
            "dt": 0.01,
            "n_transient": 1,
            "n_measure": 1,
            "n_samples": 1,
            "R_threshold": 0.5,
            "seed": 1,
        }
        kwargs[field] = bad_value

        with pytest.raises(ValueError, match=field):
            basin_stability(
                np.ones(3, dtype=np.float64),
                np.zeros((3, 3), dtype=np.float64),
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
        "R_thresholds",
        [(), (0.3, np.nan), (-0.1, 0.5), (0.7, 1.2), (False, 0.5)],
    )
    def test_multi_basin_stability_rejects_invalid_thresholds(
        self,
        R_thresholds: tuple[object, ...],
    ) -> None:
        """Thresholds must be finite real values in the unit interval."""
        with pytest.raises(ValueError, match="R_thresholds"):
            multi_basin_stability(
                np.ones(3, dtype=np.float64),
                np.zeros((3, 3), dtype=np.float64),
                n_transient=1,
                n_measure=1,
                n_samples=1,
                R_thresholds=cast("tuple[float, ...]", R_thresholds),
            )

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("phases_init", np.array([True, False, True])),
            ("omegas", np.array([True, False, True])),
            ("knm", np.eye(3, dtype=bool)),
            ("alpha", np.eye(3, dtype=bool)),
        ],
    )
    def test_steady_state_rejects_boolean_alias_arrays(
        self, field: str, bad_value: FloatArray
    ) -> None:
        """Boolean arrays cannot be converted into public real-valued trial inputs."""
        values = {
            "phases_init": np.zeros(3, dtype=np.float64),
            "omegas": np.ones(3, dtype=np.float64),
            "knm": np.zeros((3, 3), dtype=np.float64),
            "alpha": np.zeros((3, 3), dtype=np.float64),
        }
        values[field] = bad_value

        with pytest.raises(
            ValueError, match=f"{field} must not contain boolean values"
        ):
            steady_state_r(
                values["phases_init"],
                values["omegas"],
                values["knm"],
                alpha=values["alpha"],
                n_transient=1,
                n_measure=1,
            )

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("omegas", np.array([True, False, True])),
            ("knm", np.eye(3, dtype=bool)),
            ("alpha", np.eye(3, dtype=bool)),
        ],
    )
    def test_basin_stability_rejects_boolean_alias_arrays(
        self, field: str, bad_value: FloatArray
    ) -> None:
        """Boolean arrays cannot be converted into real-valued sampling inputs."""
        kwargs = {
            "omegas": np.ones(3, dtype=np.float64),
            "knm": np.zeros((3, 3), dtype=np.float64),
            "alpha": np.zeros((3, 3), dtype=np.float64),
        }
        kwargs[field] = bad_value

        with pytest.raises(
            ValueError, match=f"{field} must not contain boolean values"
        ):
            basin_stability(
                kwargs["omegas"],
                kwargs["knm"],
                alpha=kwargs["alpha"],
                n_transient=1,
                n_measure=1,
                n_samples=1,
                seed=1,
            )


class TestBasinStabilityResultValidation:
    """Result fields retain numerical types and consistent sample meaning."""

    def test_normalizes_public_record_values(self) -> None:
        """Consistent record fields retain canonical finite representations."""
        result = BasinStabilityResult(
            S_B=0.5,
            n_samples=2,
            n_converged=1,
            R_final=cast(FloatArray, [0.25, 0.75]),
            R_threshold=0.5,
        )

        assert result.S_B == 0.5
        assert result.n_samples == 2
        assert result.n_converged == 1
        assert result.R_final.dtype == np.float64
        np.testing.assert_allclose(result.R_final, [0.25, 0.75])
        assert result.R_threshold == 0.5

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"S_B": 1.2}, "S_B must be in \\[0, 1\\]"),
            ({"n_samples": -1}, "n_samples must be an integer >= 0"),
            ({"n_converged": 3}, "n_converged must be <= n_samples"),
            ({"R_final": [0.25]}, "R_final shape"),
            ({"R_final": [0.25, np.nan]}, "R_final must contain only finite values"),
            ({"R_final": [0.25, 1.25]}, "R_final values must lie in"),
            ({"R_final": [True, False]}, "R_final must not contain boolean values"),
            ({"R_threshold": -0.1}, "R_threshold must be in \\[0, 1\\]"),
        ],
    )
    def test_rejects_invalid_public_record_values(
        self, kwargs: dict[str, object], match: str
    ) -> None:
        """Malformed record fields cannot become finite-window sampling evidence."""
        base: dict[str, object] = {
            "S_B": 0.5,
            "n_samples": 2,
            "n_converged": 1,
            "R_final": [0.25, 0.75],
            "R_threshold": 0.5,
        }
        base.update(kwargs)

        with pytest.raises(ValueError, match=match):
            BasinStabilityResult(
                S_B=cast("float", base["S_B"]),
                n_samples=cast("int", base["n_samples"]),
                n_converged=cast("int", base["n_converged"]),
                R_final=cast("FloatArray", base["R_final"]),
                R_threshold=cast("float", base["R_threshold"]),
            )
