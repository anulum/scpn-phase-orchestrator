# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — OPT-entropy tests

from __future__ import annotations

import itertools
import math
from typing import Any, get_type_hints

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor import opt_entropy as oe_module
from scpn_phase_orchestrator.monitor.opt_entropy import (
    ordinal_pattern_sequence,
    transition_entropy,
)
from tests.typing_contracts import assert_precise_ndarray_hint


class TestOrdinalPatternSequence:
    def test_public_array_contracts_are_parameterised(self) -> None:
        hints = (
            get_type_hints(ordinal_pattern_sequence)["series"],
            get_type_hints(ordinal_pattern_sequence)["return"],
            get_type_hints(transition_entropy)["series"],
        )
        for hint in hints:
            assert_precise_ndarray_hint(hint)
        assert "float64" in str(get_type_hints(ordinal_pattern_sequence)["series"])
        assert "int64" in str(get_type_hints(ordinal_pattern_sequence)["return"])

    def test_strictly_increasing_is_identity_pattern(self) -> None:
        codes = ordinal_pattern_sequence(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 3, 1)
        np.testing.assert_array_equal(codes, np.zeros(3, dtype=np.int64))

    def test_strictly_decreasing_is_max_pattern(self) -> None:
        codes = ordinal_pattern_sequence(np.array([5.0, 4.0, 3.0, 2.0, 1.0]), 3, 1)
        np.testing.assert_array_equal(codes, np.full(3, 5, dtype=np.int64))

    def test_all_orderings_form_a_bijection_to_factorial_range(self) -> None:
        # The six strict orderings of three distinct values must map to the
        # six distinct Lehmer codes 0..5 exactly once each.
        seen = {
            int(ordinal_pattern_sequence(np.array(perm, dtype=np.float64), 3, 1)[0])
            for perm in itertools.permutations((0.0, 1.0, 2.0))
        }
        assert seen == set(range(6))

    def test_window_count_matches_formula(self) -> None:
        series = np.arange(20, dtype=np.float64)
        for dimension in (2, 3, 4):
            for delay in (1, 2, 3):
                codes = ordinal_pattern_sequence(series, dimension, delay)
                assert codes.shape[0] == 20 - (dimension - 1) * delay

    def test_ties_break_by_index(self) -> None:
        # Equal samples keep ascending sample order, so a flat window is the
        # identity pattern (code 0) under the Bandt–Pompe convention.
        codes = ordinal_pattern_sequence(np.array([7.0, 7.0, 7.0, 7.0]), 3, 1)
        np.testing.assert_array_equal(codes, np.zeros(2, dtype=np.int64))

    def test_codes_within_factorial_range(self) -> None:
        rng = np.random.default_rng(11)
        for dimension in (2, 3, 4, 5):
            codes = ordinal_pattern_sequence(rng.standard_normal(200), dimension, 1)
            assert int(codes.min()) >= 0
            assert int(codes.max()) < math.factorial(dimension)

    def test_short_series_returns_empty(self) -> None:
        assert ordinal_pattern_sequence(np.array([1.0, 2.0]), 3, 1).shape == (0,)
        assert ordinal_pattern_sequence(np.array([]), 3, 1).shape == (0,)

    def test_accepts_array_like(self) -> None:
        codes = ordinal_pattern_sequence([3.0, 1.0, 2.0, 4.0], 3, 1)
        assert codes.dtype == np.int64
        assert codes.shape == (2,)


class TestTransitionEntropy:
    def test_constant_series_is_zero(self) -> None:
        assert transition_entropy(np.ones(128), 3, 1) == 0.0

    def test_monotone_series_is_zero(self) -> None:
        # A strictly increasing ramp has one ordinal pattern, hence a single
        # self-transition: no transition diversity.
        assert transition_entropy(np.arange(128, dtype=np.float64), 3, 1) == 0.0

    def test_short_series_is_zero(self) -> None:
        assert transition_entropy(np.array([1.0, 2.0]), 3, 1) == 0.0
        assert transition_entropy(np.array([1.0, 2.0, 3.0]), 3, 1) == 0.0

    def test_bounded_unit_interval(self) -> None:
        rng = np.random.default_rng(3)
        for _ in range(10):
            series = rng.standard_normal(int(rng.integers(50, 500)))
            value = transition_entropy(series, 3, 1)
            assert 0.0 <= value <= 1.0

    def test_noise_exceeds_regular_dynamics(self) -> None:
        rng = np.random.default_rng(0)
        noise = transition_entropy(rng.standard_normal(4000), 3, 1)
        t = np.linspace(0.0, 200.0 * np.pi, 4000)
        regular = transition_entropy(np.sin(t), 3, 1)
        assert regular < noise

    @pytest.mark.parametrize("dimension", [2, 3, 4, 5])
    def test_dimension_variations_bounded(self, dimension: int) -> None:
        rng = np.random.default_rng(dimension)
        value = transition_entropy(rng.standard_normal(2000), dimension, 1)
        assert 0.0 <= value <= 1.0

    def test_accepts_array_like(self) -> None:
        value = transition_entropy([0.0, 1.0, 0.5, 2.0, 1.5, 3.0, 2.5], 3, 1)
        assert isinstance(value, float)


class TestInputValidation:
    @pytest.mark.parametrize(
        ("series", "match"),
        [
            (np.zeros((3, 2), dtype=np.float64), "one-dimensional"),
            (np.array([0.0, np.nan]), "finite"),
            (np.array([0.0, np.inf]), "finite"),
            (np.array([True, False]), "boolean"),
            (np.array([0.0, np.bool_(True)], dtype=object), "boolean"),
            (np.array([0.0 + 1.0j, 1.0 + 0.0j]), "real-valued"),
            (np.array([0.0, 1.0j], dtype=object), "real-valued"),
            (np.array(["a", "b", "c"], dtype=object), "one-dimensional float array"),
        ],
    )
    def test_rejects_invalid_series(self, series: np.ndarray, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            transition_entropy(series, 3, 1)
        with pytest.raises(ValueError, match=match):
            ordinal_pattern_sequence(series, 3, 1)

    @pytest.mark.parametrize("dimension", [1, 8, 0, -1, True, 3.0, "3"])
    def test_rejects_invalid_dimension(self, dimension: Any) -> None:
        with pytest.raises(ValueError, match="dimension"):
            transition_entropy(np.arange(20, dtype=np.float64), dimension, 1)

    @pytest.mark.parametrize("delay", [0, -1, True, 2.0, "2"])
    def test_rejects_invalid_delay(self, delay: Any) -> None:
        with pytest.raises(ValueError, match="delay"):
            transition_entropy(np.arange(20, dtype=np.float64), 3, delay)


class TestDispatch:
    def test_transition_entropy_uses_backend_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        series = np.array([0.0, 1.0, 0.5, 2.0, 1.5, 3.0], dtype=np.float64)
        expected = oe_module._transition_entropy_reference(series, 3, 1)
        calls: list[tuple[np.ndarray, int, int]] = []

        def _fake(s: np.ndarray, d: int, tau: int) -> float:
            calls.append((s, d, tau))
            return expected

        monkeypatch.setattr(
            oe_module,
            "_dispatch",
            lambda fn_name: _fake if fn_name == "transition_entropy" else None,
        )
        assert transition_entropy(series, 3, 1) == pytest.approx(expected, abs=1e-12)
        assert len(calls) == 1

    def test_transition_entropy_falls_back_when_backend_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raising(_s: np.ndarray, _d: int, _t: int) -> float:
            raise RuntimeError("boom")

        monkeypatch.setattr(
            oe_module,
            "_dispatch",
            lambda fn_name: _raising if fn_name == "transition_entropy" else None,
        )
        series = np.sin(np.linspace(0.0, 50.0, 500))
        value = transition_entropy(series, 3, 1)
        assert 0.0 <= value <= 1.0

    def test_ordinal_sequence_rejects_wrong_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        series = np.array([3.0, 1.0, 2.0, 4.0, 0.0], dtype=np.float64)

        def _wrong(s: np.ndarray, d: int, tau: int) -> np.ndarray:
            return np.zeros(s.size - (d - 1) * tau, dtype=np.int64) + 99

        monkeypatch.setattr(
            oe_module,
            "_dispatch",
            lambda fn_name: _wrong if fn_name == "ordinal_pattern_sequence" else None,
        )
        with pytest.raises(ValueError, match="ordinal pattern"):
            ordinal_pattern_sequence(series, 3, 1)

    def test_transition_entropy_rejects_wrong_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        series = np.array([0.0, 1.0, 0.5, 2.0, 1.5, 3.0], dtype=np.float64)
        reference = oe_module._transition_entropy_reference(series, 3, 1)

        def _wrong(_s: np.ndarray, _d: int, _t: int) -> float:
            return 0.0 if reference > 0.5 else 0.99

        monkeypatch.setattr(
            oe_module,
            "_dispatch",
            lambda fn_name: _wrong if fn_name == "transition_entropy" else None,
        )
        with pytest.raises(ValueError, match="exact reference"):
            transition_entropy(series, 3, 1)

    def test_dispatch_falls_back_to_python_when_loader_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        previous_backend = oe_module.ACTIVE_BACKEND
        previous_available = list(oe_module.AVAILABLE_BACKENDS)
        oe_module.ACTIVE_BACKEND = "go"
        oe_module.AVAILABLE_BACKENDS = ["go", "python"]
        oe_module._BACKEND_CACHE.clear()
        monkeypatch.setitem(
            oe_module._LOADERS,
            "go",
            lambda: (_ for _ in ()).throw(ImportError("go backend unavailable")),
        )
        try:
            fn = oe_module._dispatch("transition_entropy")
        finally:
            oe_module.ACTIVE_BACKEND = previous_backend
            oe_module.AVAILABLE_BACKENDS = previous_available
            oe_module._BACKEND_CACHE.clear()
        assert fn is None


class TestPythonReferencePath:
    def test_both_primitives_return_reference_without_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # _dispatch returning None is the pure-Python path: the public
        # functions return the NumPy reference directly.
        monkeypatch.setattr(oe_module, "_dispatch", lambda _name: None)
        series = np.array([3.0, 1.0, 2.0, 4.0, 0.0, 5.0], dtype=np.float64)
        np.testing.assert_array_equal(
            ordinal_pattern_sequence(series, 3, 1),
            oe_module._ordinal_codes_reference(series, 3, 1),
        )
        assert transition_entropy(series, 3, 1) == oe_module._transition_entropy_reference(
            series, 3, 1
        )


class TestPipelineWiring:
    def test_engine_order_parameter_series_yields_bounded_entropy(self) -> None:
        """Engine trajectory → order-parameter series → transition entropy.

        Proves the monitor consumes a real integrator observable end-to-end.
        """
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        n = 12
        eng = UPDEEngine(n, dt=0.02)
        rng = np.random.default_rng(5)
        phases = rng.uniform(0.0, 2.0 * np.pi, n)
        omegas = rng.normal(0.0, 0.5, n)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        series = np.empty(400, dtype=np.float64)
        for step in range(400):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
            r, _psi = compute_order_parameter(phases)
            series[step] = r
        value = transition_entropy(series, 3, 1)
        assert 0.0 <= value <= 1.0


class TestBackendBoundaryHardening:
    @pytest.mark.parametrize("backend_value", [-0.1, 1.1, np.nan, np.inf, True])
    def test_invalid_backend_scalar_fails_closed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        backend_value: Any,
    ) -> None:
        def _bad(_s: np.ndarray, _d: int, _t: int) -> Any:
            return backend_value

        monkeypatch.setattr(
            oe_module,
            "_dispatch",
            lambda fn_name: _bad if fn_name == "transition_entropy" else None,
        )
        with pytest.raises(ValueError):
            transition_entropy(np.sin(np.linspace(0.0, 40.0, 300)), 3, 1)

    @pytest.mark.parametrize(
        "backend_codes",
        [
            np.array([0.0, 1.5], dtype=np.float64),  # non-integer
            np.array([0, 999], dtype=np.int64),  # out of factorial range
            np.array([[0, 1], [2, 3]], dtype=np.int64),  # wrong rank
            np.array([True, False]),  # boolean alias
            np.array([0.0 + 1.0j, 1.0 + 0.0j]),  # complex
            np.array([np.inf, 0.0]),  # non-finite
            np.array([0, 0], dtype=np.int64),  # in-range integer but != reference
        ],
    )
    def test_invalid_backend_codes_fail_closed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        backend_codes: np.ndarray,
    ) -> None:
        def _bad(_s: np.ndarray, _d: int, _t: int) -> np.ndarray:
            return backend_codes

        monkeypatch.setattr(
            oe_module,
            "_dispatch",
            lambda fn_name: _bad if fn_name == "ordinal_pattern_sequence" else None,
        )
        with pytest.raises(ValueError):
            ordinal_pattern_sequence(
                np.array([3.0, 1.0, 2.0, 4.0], dtype=np.float64), 3, 1
            )


class TestDispatchInternals:
    def test_resolve_backends_skips_failing_loader(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(
            oe_module._LOADERS,
            "go",
            lambda: (_ for _ in ()).throw(ImportError("go unavailable")),
        )
        oe_module._BACKEND_CACHE.clear()
        try:
            active, available = oe_module._resolve_backends()
        finally:
            oe_module._BACKEND_CACHE.clear()
        assert "go" not in available
        assert available[-1] == "python"
        assert active == available[0]

    def test_dispatch_skips_loader_missing_function(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        previous_backend = oe_module.ACTIVE_BACKEND
        previous_available = list(oe_module.AVAILABLE_BACKENDS)
        oe_module.ACTIVE_BACKEND = "go"
        oe_module.AVAILABLE_BACKENDS = ["go", "python"]
        oe_module._BACKEND_CACHE.clear()
        monkeypatch.setitem(
            oe_module._LOADERS,
            "go",
            lambda: {"transition_entropy": lambda *a: 0.0},
        )
        try:
            fn = oe_module._dispatch("ordinal_pattern_sequence")
        finally:
            oe_module.ACTIVE_BACKEND = previous_backend
            oe_module.AVAILABLE_BACKENDS = previous_available
            oe_module._BACKEND_CACHE.clear()
        assert fn is None

    def test_ordinal_sequence_falls_back_when_backend_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raising(_s: np.ndarray, _d: int, _t: int) -> np.ndarray:
            raise RuntimeError("boom")

        monkeypatch.setattr(
            oe_module,
            "_dispatch",
            lambda fn_name: _raising if fn_name == "ordinal_pattern_sequence" else None,
        )
        series = np.array([3.0, 1.0, 2.0, 4.0, 0.0], dtype=np.float64)
        codes = ordinal_pattern_sequence(series, 3, 1)
        np.testing.assert_array_equal(
            codes, oe_module._ordinal_codes_reference(series, 3, 1)
        )

    def test_dispatch_returns_none_when_chain_exhausted_without_python(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        previous_backend = oe_module.ACTIVE_BACKEND
        previous_available = list(oe_module.AVAILABLE_BACKENDS)
        oe_module.ACTIVE_BACKEND = "go"
        oe_module.AVAILABLE_BACKENDS = ["go"]  # no python floor
        oe_module._BACKEND_CACHE.clear()
        monkeypatch.setitem(
            oe_module._LOADERS,
            "go",
            lambda: {"transition_entropy": lambda *a: 0.0},
        )
        try:
            fn = oe_module._dispatch("ordinal_pattern_sequence")
        finally:
            oe_module.ACTIVE_BACKEND = previous_backend
            oe_module.AVAILABLE_BACKENDS = previous_available
            oe_module._BACKEND_CACHE.clear()
        assert fn is None
