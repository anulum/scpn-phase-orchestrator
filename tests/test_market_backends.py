# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Cross-backend parity for market kernels

"""Cross-backend parity for ``market_order_parameter`` and
``market_plv``.

The native backends all use the sincos expansion
``sin(θ_j − θ_i) = s_j·c_i − c_j·s_i`` inside the inner loop,
while the Python reference builds the complex order parameter
directly via ``np.mean(np.exp(1j·θ))``. The two forms are
mathematically identical but accumulate different floating-point
rounding, so parity is tight (~1e-15) but not always 0.0.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _market_go as market_go_mod,
)
from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _market_julia as market_julia_mod,
)
from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _market_mojo as market_mojo_mod,
)
from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _market_validation as market_validation,
)
from scpn_phase_orchestrator.upde import market as m_mod
from scpn_phase_orchestrator.upde.market import (
    market_order_parameter,
    market_plv,
)

TOL = 1e-12
OrderBackend = Callable[..., np.ndarray]
PLVBackend = Callable[..., np.ndarray]
DIRECT_ORDER_BACKENDS: tuple[OrderBackend, ...] = (
    market_go_mod.market_order_parameter_go,
    market_julia_mod.market_order_parameter_julia,
    market_mojo_mod.market_order_parameter_mojo,
)
DIRECT_PLV_BACKENDS: tuple[PLVBackend, ...] = (
    market_go_mod.market_plv_go,
    market_julia_mod.market_plv_julia,
    market_mojo_mod.market_plv_mojo,
)


def test__market_validation_helper_is_directly_linked_to_backend_tests() -> None:
    assert callable(market_validation.validate_market_order_inputs)
    assert callable(market_validation.validate_market_order_output)
    assert callable(market_validation.validate_market_plv_inputs)
    assert callable(market_validation.validate_market_plv_output)


@contextlib.contextmanager
def _force_backend(name: str):
    prev = m_mod.ACTIVE_BACKEND
    m_mod.ACTIVE_BACKEND = name
    try:
        yield
    finally:
        m_mod.ACTIVE_BACKEND = prev


def _problem(seed: int, T: int = 40, N: int = 5):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 2 * np.pi, (T, N))


def _direct_payload(t: int = 6, n: int = 3) -> np.ndarray:
    return _problem(11, T=t, N=n).ravel()


def _forbid_runtime_load(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail_loader() -> object:
        raise AssertionError("optional backend loaded before validation")

    monkeypatch.setattr(market_go_mod, "_load_lib", _fail_loader)
    monkeypatch.setattr(market_julia_mod, "_ensure", _fail_loader)
    monkeypatch.setattr(market_mojo_mod, "_ensure_exe", _fail_loader)


def _op_backend(backend: str, seed: int, T: int = 40, N: int = 5):
    if backend not in m_mod.AVAILABLE_BACKENDS:
        pytest.skip(f"backend {backend!r} unavailable")
    phases = _problem(seed, T, N)
    with _force_backend(backend):
        return market_order_parameter(phases)


def _plv_backend(backend: str, seed: int, T: int = 40, N: int = 5, W: int = 10):
    if backend not in m_mod.AVAILABLE_BACKENDS:
        pytest.skip(f"backend {backend!r} unavailable")
    phases = _problem(seed, T, N)
    with _force_backend(backend):
        return market_plv(phases, window=W)


class TestDirectMojoBoundaryContracts:
    @pytest.mark.parametrize(
        ("stdout", "expected_lines", "label", "match"),
        [
            ("", 2, "ORDER", "Mojo market ORDER returned 0 lines, expected 2"),
            (
                "0.1\n0.2\n0.3\n",
                2,
                "ORDER",
                "Mojo market ORDER returned 3 lines, expected 2",
            ),
            (
                "0.1\n\n0.2\n",
                2,
                "ORDER",
                "Mojo market ORDER returned 3 lines, expected 2",
            ),
            ("0.1\nnot-a-number\n", 2, "PLV", "finite real values"),
            ("0.1\nnan\n", 2, "PLV", "finite real values"),
        ],
    )
    def test_mojo_runner_rejects_malformed_raw_stdout(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stdout: str,
        expected_lines: int,
        label: str,
        match: str,
    ) -> None:
        monkeypatch.setattr(market_mojo_mod, "_ensure_exe", lambda: "market")
        monkeypatch.setattr(
            market_mojo_mod.subprocess,
            "run",
            lambda *_args, **_kwargs: SimpleNamespace(
                returncode=0,
                stdout=stdout,
                stderr="",
            ),
        )

        with pytest.raises(ValueError, match=match):
            market_mojo_mod._run_mojo(
                ["ORDER", "1", "1", "0.0"],
                expected_lines=expected_lines,
                label=label,
            )


class TestDirectBackendBoundaryContracts:
    @pytest.mark.parametrize("backend", DIRECT_ORDER_BACKENDS)
    @pytest.mark.parametrize(
        ("phases", "t", "n"),
        [
            (_direct_payload().reshape(2, -1), 6, 3),
            (_direct_payload().astype(bool), 6, 3),
            (_direct_payload().astype(np.complex128) + 1j, 6, 3),
            (np.array([np.nan, *_direct_payload()[1:]]), 6, 3),
            (_direct_payload()[:-1], 6, 3),
            (_direct_payload(), True, 3),
            (_direct_payload(), 0, 3),
            (_direct_payload(), 6, True),
            (_direct_payload(), 6, 0),
        ],
    )
    def test_order_backends_reject_invalid_inputs_before_runtime_loading(
        self,
        monkeypatch: pytest.MonkeyPatch,
        backend: OrderBackend,
        phases: object,
        t: object,
        n: object,
    ) -> None:
        _forbid_runtime_load(monkeypatch)

        with pytest.raises((TypeError, ValueError)):
            backend(phases, t, n)

    @pytest.mark.parametrize("backend", DIRECT_PLV_BACKENDS)
    @pytest.mark.parametrize(
        ("phases", "t", "n", "window"),
        [
            (_direct_payload().reshape(2, -1), 6, 3, 2),
            (_direct_payload().astype(bool), 6, 3, 2),
            (_direct_payload().astype(np.complex128) + 1j, 6, 3, 2),
            (np.array([np.inf, *_direct_payload()[1:]]), 6, 3, 2),
            (_direct_payload()[:-1], 6, 3, 2),
            (_direct_payload(), True, 3, 2),
            (_direct_payload(), 6, True, 2),
            (_direct_payload(), 6, 3, True),
            (_direct_payload(), 6, 3, 0),
            (_direct_payload(), 6, 3, 7),
        ],
    )
    def test_plv_backends_reject_invalid_inputs_before_runtime_loading(
        self,
        monkeypatch: pytest.MonkeyPatch,
        backend: PLVBackend,
        phases: object,
        t: object,
        n: object,
        window: object,
    ) -> None:
        _forbid_runtime_load(monkeypatch)

        with pytest.raises((TypeError, ValueError)):
            backend(phases, t, n, window)

    @pytest.mark.parametrize(
        "output",
        [
            np.array([0.1, np.nan]),
            np.array([0.1, 1.2]),
            np.array([0.1], dtype=np.float64),
            np.array([True, False]),
            np.array([0.1 + 0.0j, 0.2 + 0.0j]),
        ],
    )
    def test_order_output_contract_rejects_non_physical_backend_payloads(
        self, output: np.ndarray
    ) -> None:
        with pytest.raises((TypeError, ValueError)):
            market_validation.validate_market_order_output(output, t=2)

    @pytest.mark.parametrize(
        "output",
        [
            np.array([0.1, np.nan, 0.2, 1.0]),
            np.array([0.1, 1.2, 0.2, 1.0]),
            np.array([0.1, 0.2, 1.0]),
            np.array([True, False, True, False]),
            np.array([0.1 + 0.0j, 0.2 + 0.0j, 0.2 + 0.0j, 1.0 + 0.0j]),
            np.array([0.2, 0.2, 0.2, 0.2]),
            np.array([1.0, 0.1, 0.2, 1.0]),
        ],
    )
    def test_plv_output_contract_rejects_non_physical_backend_payloads(
        self, output: np.ndarray
    ) -> None:
        with pytest.raises((TypeError, ValueError)):
            market_validation.validate_market_plv_output(output, t=2, n=2, window=2)


class TestOrderParameterParity:
    def test_rust(self):
        ref = _op_backend("python", 0)
        got = _op_backend("rust", 0)
        assert np.max(np.abs(got - ref)) < TOL

    def test_julia(self):
        ref = _op_backend("python", 1)
        got = _op_backend("julia", 1)
        assert np.max(np.abs(got - ref)) < TOL

    def test_go(self):
        ref = _op_backend("python", 2)
        got = _op_backend("go", 2)
        assert np.max(np.abs(got - ref)) < TOL

    def test_mojo(self):
        ref = _op_backend("python", 3, T=20)
        got = _op_backend("mojo", 3, T=20)
        assert np.max(np.abs(got - ref)) < 1e-10


class TestPLVParity:
    def test_rust(self):
        ref = _plv_backend("python", 4)
        got = _plv_backend("rust", 4)
        assert np.max(np.abs(got - ref)) < TOL

    def test_julia(self):
        ref = _plv_backend("python", 5)
        got = _plv_backend("julia", 5)
        assert np.max(np.abs(got - ref)) < TOL

    def test_go(self):
        ref = _plv_backend("python", 6)
        got = _plv_backend("go", 6)
        assert np.max(np.abs(got - ref)) < TOL

    def test_mojo(self):
        ref = _plv_backend("python", 7, T=20, N=3, W=5)
        got = _plv_backend("mojo", 7, T=20, N=3, W=5)
        assert np.max(np.abs(got - ref)) < 1e-10


class TestHypothesisParity:
    @given(
        T=st.integers(min_value=10, max_value=40),
        N=st.integers(min_value=2, max_value=6),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=6,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_op_rust_hypothesis(self, T, N, seed):
        if "rust" not in m_mod.AVAILABLE_BACKENDS:
            pytest.skip("rust unavailable")
        ref = _op_backend("python", seed, T=T, N=N)
        got = _op_backend("rust", seed, T=T, N=N)
        assert np.max(np.abs(got - ref)) < TOL

    @given(
        T=st.integers(min_value=10, max_value=30),
        N=st.integers(min_value=2, max_value=5),
        W=st.integers(min_value=3, max_value=8),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=5,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_plv_go_hypothesis(self, T, N, W, seed):
        if "go" not in m_mod.AVAILABLE_BACKENDS or W >= T:
            pytest.skip("go unavailable or window too wide")
        ref = _plv_backend("python", seed, T=T, N=N, W=W)
        got = _plv_backend("go", seed, T=T, N=N, W=W)
        assert np.max(np.abs(got - ref)) < TOL


class TestDispatchFallbackChain:
    def test_dispatch_falls_back_to_next_backend_when_loader_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict[str, int] = {"rust": 0, "go": 0}

        def _fail_rust():
            calls["rust"] += 1
            raise ImportError("missing rust backend")

        def _ok_go():
            calls["go"] += 1
            return (
                lambda phases_flat, t, n: np.full(t, 0.5, dtype=np.float64),
                lambda phases_flat, t, n, window: np.zeros(
                    ((t - window + 1), n, n), dtype=np.float64
                ),
            )

        monkeypatch.setattr(m_mod, "_BACKEND_CACHE", {})
        monkeypatch.setattr(m_mod, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(m_mod, "AVAILABLE_BACKENDS", ["rust", "go", "python"])
        monkeypatch.setattr(m_mod, "_LOADERS", {"rust": _fail_rust, "go": _ok_go})

        phases = np.zeros((6, 2), dtype=np.float64)
        op = m_mod.market_order_parameter(phases)
        assert op.shape == (6,)
        assert np.allclose(op, 0.5)
        assert calls == {"rust": 1, "go": 1}

    def test_dispatch_uses_cached_loader_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict[str, int] = {"go": 0}

        def _ok_go():
            calls["go"] += 1
            return (
                lambda phases_flat, t, n: np.full(t, 0.25, dtype=np.float64),
                lambda phases_flat, t, n, window: np.zeros(
                    ((t - window + 1), n, n), dtype=np.float64
                ),
            )

        monkeypatch.setattr(m_mod, "_BACKEND_CACHE", {})
        monkeypatch.setattr(m_mod, "ACTIVE_BACKEND", "go")
        monkeypatch.setattr(m_mod, "AVAILABLE_BACKENDS", ["go", "python"])
        monkeypatch.setattr(m_mod, "_LOADERS", {"go": _ok_go})

        phases = np.zeros((7, 3), dtype=np.float64)
        first = m_mod.market_order_parameter(phases)
        second = m_mod.market_order_parameter(phases)

        assert np.allclose(first, 0.25)
        assert np.allclose(second, 0.25)
        assert calls["go"] == 1
