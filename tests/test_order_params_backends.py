# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Per-backend parity tests for order parameters

"""Per-backend parity tests for ``upde/order_params.py``.

Exercises every non-Python backend individually against the NumPy
reference for all three compute kernels.

Tolerance budgets match the AttnRes reference:

* Rust / Julia / Go — bit-exact (≤ 1e-12)
* Mojo — ≤ 1e-13 (text round-trip)
"""

from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _order_params_go as order_params_go_mod,
)
from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _order_params_julia as order_params_julia_mod,
)
from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _order_params_mojo as order_params_mojo_mod,
)
from scpn_phase_orchestrator.experimental.accelerators.upde._order_params_go import (
    layer_coherence_go,
    order_parameter_go,
    plv_go,
)
from scpn_phase_orchestrator.experimental.accelerators.upde._order_params_julia import (
    layer_coherence_julia,
    order_parameter_julia,
    plv_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.upde._order_params_mojo import (
    layer_coherence_mojo,
    order_parameter_mojo,
    plv_mojo,
)
from scpn_phase_orchestrator.upde import (
    _order_params_validation as order_params_validation,
)
from scpn_phase_orchestrator.upde import order_params as op_mod
from scpn_phase_orchestrator.upde.order_params import (
    AVAILABLE_BACKENDS,
    compute_layer_coherence,
    compute_order_parameter,
    compute_plv,
)

TWO_PI = 2.0 * np.pi


def test__order_params_validation_helper_is_directly_linked_to_backend_tests() -> None:
    phases = np.array([0.0, np.pi], dtype=np.float64)
    indices = np.array([0, 1], dtype=np.int64)

    validated_phases = order_params_validation.validate_order_parameter_inputs(phases)
    r, psi = order_params_validation.validate_order_parameter_output(0.25, 1.0)
    phases_a, phases_b = order_params_validation.validate_plv_inputs(phases, phases)
    unit_value = order_params_validation.validate_unit_interval_output(
        0.5,
        name="plv",
    )
    layer_phases, layer_indices = (
        order_params_validation.validate_layer_coherence_inputs(
            phases,
            indices,
        )
    )

    np.testing.assert_array_equal(validated_phases, phases)
    assert (r, psi) == (0.25, 1.0)
    np.testing.assert_array_equal(phases_a, phases_b)
    assert unit_value == 0.5
    np.testing.assert_array_equal(layer_phases, phases)
    np.testing.assert_array_equal(layer_indices, indices)


class _FakeGoOrderParamsLib:
    def __init__(
        self,
        *,
        order: tuple[float, float] = (0.5, 1.0),
        plv: float = 0.25,
        layer: float = 0.75,
        rc: int = 0,
    ) -> None:
        self.order = order
        self.plv = plv
        self.layer = layer
        self.rc = rc

    def OrderParameter(self, *_args: object) -> int:
        ctypes.cast(_args[-2], ctypes.POINTER(ctypes.c_double))[0] = self.order[0]
        ctypes.cast(_args[-1], ctypes.POINTER(ctypes.c_double))[0] = self.order[1]
        return self.rc

    def PLV(self, *_args: object) -> int:
        ctypes.cast(_args[-1], ctypes.POINTER(ctypes.c_double))[0] = self.plv
        return self.rc

    def LayerCoherence(self, *_args: object) -> int:
        ctypes.cast(_args[-1], ctypes.POINTER(ctypes.c_double))[0] = self.layer
        return self.rc


class _FakeJuliaOrderParams:
    def __init__(
        self,
        *,
        order: tuple[float, float] = (0.5, 1.0),
        plv: float = 0.25,
        layer: float = 0.75,
    ) -> None:
        self.order = order
        self._plv = plv
        self.layer = layer

    def order_parameter(self, _phases: np.ndarray) -> tuple[float, float]:
        return self.order

    def plv(self, _phases_a: np.ndarray, _phases_b: np.ndarray) -> float:
        return self._plv

    def layer_coherence(self, _phases: np.ndarray, _indices: np.ndarray) -> float:
        return self.layer


def _force(backend: str) -> str:
    """Monkey-patch ``ACTIVE_BACKEND`` and return the previous value."""
    prev = op_mod.ACTIVE_BACKEND
    op_mod.ACTIVE_BACKEND = backend
    return prev


def _reset(prev: str) -> None:
    op_mod.ACTIVE_BACKEND = prev


def _reference(
    phases_a: np.ndarray,
    phases_b: np.ndarray,
    indices: np.ndarray,
) -> tuple[tuple[float, float], float, float]:
    prev = _force("python")
    try:
        r_psi = compute_order_parameter(phases_a)
        plv_val = compute_plv(phases_a, phases_b)
        lc = compute_layer_coherence(phases_a, indices)
    finally:
        _reset(prev)
    return r_psi, plv_val, lc


class TestDirectBackendBoundaryContracts:
    @pytest.mark.parametrize(
        ("fn", "loader_attr"),
        [
            (order_parameter_go, "_load_lib"),
            (order_parameter_julia, "_ensure_julia_loaded"),
            (order_parameter_mojo, "_run"),
        ],
    )
    @pytest.mark.parametrize(
        "phases",
        [
            np.array([True, False]),
            np.array([0.0 + 0.0j, 1.0 + 0.0j]),
            np.array([0.0, np.nan]),
            np.array([[0.0, 1.0]]),
        ],
    )
    def test_order_parameter_rejects_invalid_inputs_before_runtime_load(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fn,
        loader_attr: str,
        phases: np.ndarray,
    ) -> None:
        module = fn.__globals__["__name__"]
        mod = {
            order_params_go_mod.__name__: order_params_go_mod,
            order_params_julia_mod.__name__: order_params_julia_mod,
            order_params_mojo_mod.__name__: order_params_mojo_mod,
        }[module]
        monkeypatch.setattr(
            mod,
            loader_attr,
            lambda *_args, **_kwargs: pytest.fail("runtime must not load"),
        )

        with pytest.raises(ValueError, match="phases"):
            fn(phases)

    @pytest.mark.parametrize("fn", [plv_go, plv_julia, plv_mojo])
    @pytest.mark.parametrize(
        ("phases_a", "phases_b", "match"),
        [
            (
                np.array([True, False]),
                np.zeros(2, dtype=np.float64),
                "phases_a",
            ),
            (
                np.zeros(2, dtype=np.float64),
                np.array([0.0 + 0.0j, 1.0 + 0.0j]),
                "phases_b",
            ),
            (
                np.zeros(2, dtype=np.float64),
                np.array([0.0, np.nan]),
                "phases_b",
            ),
            (
                np.zeros((1, 2), dtype=np.float64),
                np.zeros(2, dtype=np.float64),
                "phases_a",
            ),
        ],
    )
    def test_plv_rejects_invalid_inputs_before_runtime_load(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fn,
        phases_a: np.ndarray,
        phases_b: np.ndarray,
        match: str,
    ) -> None:
        module = fn.__globals__["__name__"]
        mod = {
            order_params_go_mod.__name__: order_params_go_mod,
            order_params_julia_mod.__name__: order_params_julia_mod,
            order_params_mojo_mod.__name__: order_params_mojo_mod,
        }[module]
        loader_attr = (
            "_run"
            if mod is order_params_mojo_mod
            else ("_load_lib" if mod is order_params_go_mod else "_ensure_julia_loaded")
        )
        monkeypatch.setattr(
            mod,
            loader_attr,
            lambda *_args, **_kwargs: pytest.fail("runtime must not load"),
        )

        with pytest.raises(ValueError, match=match):
            fn(phases_a, phases_b)

    @pytest.mark.parametrize("fn", [plv_go, plv_julia, plv_mojo])
    def test_plv_rejects_empty_non_empty_mismatch_before_runtime_load(self, fn) -> None:
        with pytest.raises(ValueError, match="equal-length"):
            fn(np.array([], dtype=np.float64), np.zeros(1, dtype=np.float64))

    @pytest.mark.parametrize(
        "fn",
        [layer_coherence_go, layer_coherence_julia, layer_coherence_mojo],
    )
    @pytest.mark.parametrize(
        ("phases", "indices", "match"),
        [
            (np.array([True, False]), np.array([0], dtype=np.int64), "phases"),
            (
                np.array([0.0 + 0.0j, 1.0 + 0.0j]),
                np.array([0], dtype=np.int64),
                "phases",
            ),
            (np.array([0.0, np.nan]), np.array([0], dtype=np.int64), "phases"),
            (np.zeros(2, dtype=np.float64), np.array([True]), "indices"),
            (np.zeros(2, dtype=np.float64), np.array([0.5]), "indices"),
            (np.zeros(2, dtype=np.float64), np.array([-1]), "indices"),
            (np.zeros(2, dtype=np.float64), np.array([2]), "indices"),
            (np.zeros(2, dtype=np.float64), np.array([0, 0]), "indices"),
            (np.zeros((1, 2), dtype=np.float64), np.array([0]), "phases"),
        ],
    )
    def test_layer_coherence_rejects_invalid_inputs_before_runtime_load(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fn,
        phases: np.ndarray,
        indices: np.ndarray,
        match: str,
    ) -> None:
        module = fn.__globals__["__name__"]
        mod = {
            order_params_go_mod.__name__: order_params_go_mod,
            order_params_julia_mod.__name__: order_params_julia_mod,
            order_params_mojo_mod.__name__: order_params_mojo_mod,
        }[module]
        loader_attr = (
            "_run"
            if mod is order_params_mojo_mod
            else ("_load_lib" if mod is order_params_go_mod else "_ensure_julia_loaded")
        )
        monkeypatch.setattr(
            mod,
            loader_attr,
            lambda *_args, **_kwargs: pytest.fail("runtime must not load"),
        )

        with pytest.raises(ValueError, match=match):
            fn(phases, indices)

    @pytest.mark.parametrize(
        ("order_fn", "plv_fn", "layer_fn"),
        [
            (order_parameter_go, plv_go, layer_coherence_go),
            (order_parameter_julia, plv_julia, layer_coherence_julia),
            (order_parameter_mojo, plv_mojo, layer_coherence_mojo),
        ],
    )
    def test_empty_measure_cases_skip_runtime_loading(
        self,
        monkeypatch: pytest.MonkeyPatch,
        order_fn,
        plv_fn,
        layer_fn,
    ) -> None:
        for fn in (order_fn, plv_fn, layer_fn):
            module = fn.__globals__["__name__"]
            mod = {
                order_params_go_mod.__name__: order_params_go_mod,
                order_params_julia_mod.__name__: order_params_julia_mod,
                order_params_mojo_mod.__name__: order_params_mojo_mod,
            }[module]
            loader_attr = (
                "_run"
                if mod is order_params_mojo_mod
                else (
                    "_load_lib"
                    if mod is order_params_go_mod
                    else "_ensure_julia_loaded"
                )
            )
            monkeypatch.setattr(
                mod,
                loader_attr,
                lambda *_args, **_kwargs: pytest.fail("runtime must not load"),
            )

        phases = np.array([], dtype=np.float64)
        assert order_fn(phases) == (0.0, 0.0)
        assert plv_fn(phases, phases) == 0.0
        assert layer_fn(phases, np.array([], dtype=np.int64)) == 0.0

    @pytest.mark.parametrize(
        ("module", "order_fn", "plv_fn", "layer_fn", "patcher"),
        [
            (
                order_params_go_mod,
                order_parameter_go,
                plv_go,
                layer_coherence_go,
                lambda monkeypatch, **kw: monkeypatch.setattr(
                    order_params_go_mod,
                    "_load_lib",
                    lambda: _FakeGoOrderParamsLib(**kw),
                ),
            ),
            (
                order_params_julia_mod,
                order_parameter_julia,
                plv_julia,
                layer_coherence_julia,
                lambda monkeypatch, **kw: monkeypatch.setattr(
                    order_params_julia_mod,
                    "_ensure_julia_loaded",
                    lambda: _FakeJuliaOrderParams(**kw),
                ),
            ),
            (
                order_params_mojo_mod,
                order_parameter_mojo,
                plv_mojo,
                layer_coherence_mojo,
                None,
            ),
        ],
    )
    def test_direct_backend_outputs_are_physical_and_canonical(
        self,
        monkeypatch: pytest.MonkeyPatch,
        module,
        order_fn,
        plv_fn,
        layer_fn,
        patcher,
    ) -> None:
        phases = np.array([0.0, 1.0], dtype=np.float64)
        indices = np.array([0], dtype=np.int64)

        if module is order_params_mojo_mod:
            monkeypatch.setattr(module, "_run", lambda *_args, **_kwargs: [0.5, -0.25])
        else:
            patcher(monkeypatch, order=(0.5, -0.25), plv=0.25, layer=0.75)
        assert order_fn(phases) == pytest.approx((0.5, (-0.25) % TWO_PI))

        if module is order_params_mojo_mod:
            monkeypatch.setattr(module, "_run", lambda *_args, **_kwargs: [0.25])
        assert plv_fn(phases, phases) == pytest.approx(0.25)

        if module is order_params_mojo_mod:
            monkeypatch.setattr(module, "_run", lambda *_args, **_kwargs: [0.75])
        assert layer_fn(phases, indices) == pytest.approx(0.75)

    @pytest.mark.parametrize(
        ("module", "order_fn", "plv_fn", "layer_fn", "patcher"),
        [
            (
                order_params_go_mod,
                order_parameter_go,
                plv_go,
                layer_coherence_go,
                lambda monkeypatch, **kw: monkeypatch.setattr(
                    order_params_go_mod,
                    "_load_lib",
                    lambda: _FakeGoOrderParamsLib(**kw),
                ),
            ),
            (
                order_params_julia_mod,
                order_parameter_julia,
                plv_julia,
                layer_coherence_julia,
                lambda monkeypatch, **kw: monkeypatch.setattr(
                    order_params_julia_mod,
                    "_ensure_julia_loaded",
                    lambda: _FakeJuliaOrderParams(**kw),
                ),
            ),
            (
                order_params_mojo_mod,
                order_parameter_mojo,
                plv_mojo,
                layer_coherence_mojo,
                None,
            ),
        ],
    )
    def test_direct_backend_outputs_reject_non_physical_scalars(
        self,
        monkeypatch: pytest.MonkeyPatch,
        module,
        order_fn,
        plv_fn,
        layer_fn,
        patcher,
    ) -> None:
        phases = np.array([0.0, 1.0], dtype=np.float64)
        indices = np.array([0], dtype=np.int64)

        if module is order_params_mojo_mod:
            monkeypatch.setattr(module, "_run", lambda *_args, **_kwargs: [1.25, 0.0])
        else:
            patcher(monkeypatch, order=(1.25, 0.0), plv=0.25, layer=0.75)
        with pytest.raises(ValueError, match="R"):
            order_fn(phases)

        if module is order_params_mojo_mod:
            monkeypatch.setattr(module, "_run", lambda *_args, **_kwargs: [0.5, np.nan])
        else:
            patcher(monkeypatch, order=(0.5, np.nan), plv=0.25, layer=0.75)
        with pytest.raises(ValueError, match="mean phase"):
            order_fn(phases)

        if module is order_params_mojo_mod:
            monkeypatch.setattr(module, "_run", lambda *_args, **_kwargs: [-0.1])
        else:
            patcher(monkeypatch, order=(0.5, 0.0), plv=-0.1, layer=0.75)
        with pytest.raises(ValueError, match="PLV"):
            plv_fn(phases, phases)

        if module is order_params_mojo_mod:
            monkeypatch.setattr(module, "_run", lambda *_args, **_kwargs: [1.1])
        else:
            patcher(monkeypatch, order=(0.5, 0.0), plv=0.25, layer=1.1)
        with pytest.raises(ValueError, match="layer coherence"):
            layer_fn(phases, indices)

    @pytest.mark.parametrize(
        ("stdout", "expected_count", "label", "match"),
        [
            ("", 2, "R", "Mojo order_params R returned 0 lines, expected 2"),
            (
                "0.5\n1.0\n7.0\n",
                2,
                "R",
                "Mojo order_params R returned 3 lines, expected 2",
            ),
            (
                "0.5\n\n",
                1,
                "PLV",
                "Mojo order_params PLV returned 2 lines, expected 1",
            ),
            ("not-a-number\n", 1, "LC", "finite real values"),
            ("nan\n", 1, "LC", "finite real values"),
        ],
    )
    def test_mojo_runner_rejects_malformed_raw_stdout(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stdout: str,
        expected_count: int,
        label: str,
        match: str,
    ) -> None:
        monkeypatch.setattr(order_params_mojo_mod, "_ensure_exe", lambda: "order")
        monkeypatch.setattr(
            order_params_mojo_mod.subprocess,
            "run",
            lambda *_args, **_kwargs: SimpleNamespace(
                returncode=0,
                stdout=stdout,
                stderr="",
            ),
        )

        with pytest.raises(ValueError, match=match):
            order_params_mojo_mod._run(
                "R 1 0.0\n",
                expected_count=expected_count,
                label=label,
            )


# ---------------------------------------------------------------------
# Rust parity
# ---------------------------------------------------------------------


class TestRustParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "rust" not in AVAILABLE_BACKENDS:
            pytest.skip("Rust backend not built on this host")

    @given(
        n=st.integers(min_value=2, max_value=256),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=20, deadline=None)
    def test_bit_exact_order_parameter(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0.0, TWO_PI, size=n)
        ref_op = _reference(phases, phases, np.arange(n, dtype=np.int64))[0]
        prev = _force("rust")
        try:
            r, psi = compute_order_parameter(phases)
        finally:
            _reset(prev)
        assert abs(r - ref_op[0]) < 1e-12
        assert abs(psi - ref_op[1]) < 1e-12

    @given(
        n=st.integers(min_value=2, max_value=256),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=20, deadline=None)
    def test_bit_exact_plv(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        a = rng.uniform(0.0, TWO_PI, size=n)
        b = rng.uniform(0.0, TWO_PI, size=n)
        ref = _reference(a, b, np.arange(n, dtype=np.int64))[1]
        prev = _force("rust")
        try:
            result = compute_plv(a, b)
        finally:
            _reset(prev)
        assert abs(result - ref) < 1e-12

    def test_layer_coherence_parity(self) -> None:
        rng = np.random.default_rng(0)
        phases = rng.uniform(0.0, TWO_PI, size=20)
        indices = np.array([0, 3, 5, 7, 11, 13], dtype=np.int64)
        ref = _reference(phases, phases, indices)[2]
        prev = _force("rust")
        try:
            result = compute_layer_coherence(phases, indices)
        finally:
            _reset(prev)
        assert abs(result - ref) < 1e-12


# ---------------------------------------------------------------------
# Julia parity
# ---------------------------------------------------------------------


class TestJuliaParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "julia" not in AVAILABLE_BACKENDS:
            pytest.skip("Julia backend not available")

    @pytest.mark.parametrize("n", [8, 32, 128])
    def test_order_parameter(self, n: int) -> None:
        rng = np.random.default_rng(7 + n)
        phases = rng.uniform(0.0, TWO_PI, size=n)
        ref = _reference(phases, phases, np.arange(n, dtype=np.int64))[0]
        prev = _force("julia")
        try:
            r, psi = compute_order_parameter(phases)
        finally:
            _reset(prev)
        assert abs(r - ref[0]) < 1e-12
        assert abs(psi - ref[1]) < 1e-12

    def test_plv_and_layer_coherence(self) -> None:
        rng = np.random.default_rng(13)
        a = rng.uniform(0.0, TWO_PI, size=60)
        b = rng.uniform(0.0, TWO_PI, size=60)
        indices = np.array([2, 5, 10, 20, 30], dtype=np.int64)
        ref_plv = _reference(a, b, indices)[1]
        ref_lc = _reference(a, b, indices)[2]
        prev = _force("julia")
        try:
            plv_val = compute_plv(a, b)
            lc = compute_layer_coherence(a, indices)
        finally:
            _reset(prev)
        assert abs(plv_val - ref_plv) < 1e-12
        assert abs(lc - ref_lc) < 1e-12


# ---------------------------------------------------------------------
# Go parity
# ---------------------------------------------------------------------


class TestGoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "go" not in AVAILABLE_BACKENDS:
            pytest.skip("Go backend not built")

    @given(
        n=st.integers(min_value=2, max_value=128),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=15,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_order_parameter(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0.0, TWO_PI, size=n)
        ref = _reference(phases, phases, np.arange(n, dtype=np.int64))[0]
        prev = _force("go")
        try:
            r, psi = compute_order_parameter(phases)
        finally:
            _reset(prev)
        assert abs(r - ref[0]) < 1e-12
        assert abs(psi - ref[1]) < 1e-12


# ---------------------------------------------------------------------
# Mojo parity
# ---------------------------------------------------------------------


class TestMojoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "mojo" not in AVAILABLE_BACKENDS:
            pytest.skip("Mojo backend not built")

    @pytest.mark.parametrize("n", [8, 24, 64])
    def test_order_parameter(self, n: int) -> None:
        rng = np.random.default_rng(17 + n)
        phases = rng.uniform(0.0, TWO_PI, size=n)
        ref = _reference(phases, phases, np.arange(n, dtype=np.int64))[0]
        prev = _force("mojo")
        try:
            r, psi = compute_order_parameter(phases)
        finally:
            _reset(prev)
        assert abs(r - ref[0]) < 1e-13
        assert abs(psi - ref[1]) < 1e-13

    def test_plv_and_layer_coherence(self) -> None:
        rng = np.random.default_rng(23)
        a = rng.uniform(0.0, TWO_PI, size=40)
        b = rng.uniform(0.0, TWO_PI, size=40)
        indices = np.array([3, 6, 9, 12], dtype=np.int64)
        ref_plv = _reference(a, b, indices)[1]
        ref_lc = _reference(a, b, indices)[2]
        prev = _force("mojo")
        try:
            plv_val = compute_plv(a, b)
            lc = compute_layer_coherence(a, indices)
        finally:
            _reset(prev)
        assert abs(plv_val - ref_plv) < 1e-13
        assert abs(lc - ref_lc) < 1e-13


# ---------------------------------------------------------------------
# Cross-backend consistency
# ---------------------------------------------------------------------


class TestCrossBackendConsistency:
    @pytest.mark.skipif(
        len(AVAILABLE_BACKENDS) < 2,
        reason="Only the Python fallback is available",
    )
    def test_all_backends_agree(self) -> None:
        rng = np.random.default_rng(2026)
        n = 32
        phases_a = rng.uniform(0.0, TWO_PI, size=n)
        phases_b = rng.uniform(0.0, TWO_PI, size=n)
        indices = np.arange(0, n, 3, dtype=np.int64)
        ref_op, ref_plv, ref_lc = _reference(phases_a, phases_b, indices)

        tolerances = {
            "rust": 1e-12,
            "julia": 1e-12,
            "go": 1e-12,
            "mojo": 1e-13,
            "python": 0.0,
        }

        for backend in AVAILABLE_BACKENDS:
            atol = tolerances[backend]
            prev = _force(backend)
            try:
                r, psi = compute_order_parameter(phases_a)
                plv_val = compute_plv(phases_a, phases_b)
                lc = compute_layer_coherence(phases_a, indices)
            finally:
                _reset(prev)
            assert abs(r - ref_op[0]) <= atol, (
                f"{backend} R diff {abs(r - ref_op[0]):.2e} exceeds {atol}"
            )
            assert abs(psi - ref_op[1]) <= atol
            assert abs(plv_val - ref_plv) <= atol
            assert abs(lc - ref_lc) <= atol
