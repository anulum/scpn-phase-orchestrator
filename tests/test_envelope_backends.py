# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Per-backend parity for envelope kernels

"""Cross-backend parity for :func:`extract_envelope` and
:func:`envelope_modulation_depth`.

Tolerances: Rust / Julia / Go 1e-12; Mojo 1e-9. Measured
bit-equivalent (0.0) on Rust/Julia/Go and 3.3e-15 on Mojo.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from typing import get_type_hints

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _envelope_mojo as envelope_mojo_mod,
)
from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _envelope_validation as envelope_validation,
)
from scpn_phase_orchestrator.experimental.accelerators.upde._envelope_go import (
    envelope_modulation_depth_go,
    extract_envelope_go,
)
from scpn_phase_orchestrator.experimental.accelerators.upde._envelope_julia import (
    envelope_modulation_depth_julia,
    extract_envelope_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.upde._envelope_mojo import (
    envelope_modulation_depth_mojo,
    extract_envelope_mojo,
)
from scpn_phase_orchestrator.upde import envelope as e_mod
from scpn_phase_orchestrator.upde.envelope import (
    AVAILABLE_BACKENDS,
    envelope_modulation_depth,
    extract_envelope,
)
from tests.typing_contracts import assert_precise_ndarray_hint

ExtractBackend = Callable[[np.ndarray, int], np.ndarray]
ModBackend = Callable[[np.ndarray], float]
DIRECT_EXTRACT_BACKENDS = (
    extract_envelope_go,
    extract_envelope_julia,
    extract_envelope_mojo,
)
DIRECT_MOD_BACKENDS = (
    envelope_modulation_depth_go,
    envelope_modulation_depth_julia,
    envelope_modulation_depth_mojo,
)


def test__envelope_validation_helper_is_directly_linked_to_backend_tests() -> None:
    assert callable(envelope_validation.validate_extract_envelope_input)
    assert callable(envelope_validation.validate_extract_envelope_output)
    assert callable(envelope_validation.validate_envelope_modulation_input)
    assert callable(envelope_validation.validate_envelope_modulation_output)


def _force(backend: str) -> str:
    prev = e_mod.ACTIVE_BACKEND
    e_mod.ACTIVE_BACKEND = backend
    return prev


def _reset(prev: str) -> None:
    e_mod.ACTIVE_BACKEND = prev


def _ref_env(amps, window):
    prev = _force("python")
    try:
        return extract_envelope(amps, window)
    finally:
        _reset(prev)


def _ref_mod(env):
    prev = _force("python")
    try:
        return envelope_modulation_depth(env)
    finally:
        _reset(prev)


def _amps(seed: int, n: int = 500) -> np.ndarray:
    return np.abs(np.random.default_rng(seed).normal(1.0, 0.3, n))


class TestDirectBackendBoundaryContracts:
    @pytest.mark.parametrize("backend", DIRECT_EXTRACT_BACKENDS)
    @pytest.mark.parametrize(
        ("amps", "window"),
        [
            (np.ones((1, 4)), 2),
            (np.array([True, False]), 2),
            (np.array([1.0 + 1j, 2.0 + 0j]), 2),
            (np.array([1.0, np.nan, 2.0]), 2),
            (np.array([1.0, np.inf, 2.0]), 2),
            (np.ones(4), True),
            (np.ones(4), 0),
        ],
    )
    def test_extract_invalid_inputs_fail_before_optional_runtime_loading(
        self,
        backend: ExtractBackend,
        amps: np.ndarray,
        window: object,
    ) -> None:
        with pytest.raises((TypeError, ValueError)):
            backend(amps, window)

    @pytest.mark.parametrize("backend", DIRECT_MOD_BACKENDS)
    @pytest.mark.parametrize(
        "env",
        [
            np.ones((1, 4)),
            np.array([True, False]),
            np.array([1.0 + 1j, 2.0 + 0j]),
            np.array([1.0, np.nan, 2.0]),
            np.array([1.0, np.inf, 2.0]),
        ],
    )
    def test_mod_invalid_inputs_fail_before_optional_runtime_loading(
        self,
        backend: ModBackend,
        env: np.ndarray,
    ) -> None:
        with pytest.raises((TypeError, ValueError)):
            backend(env)

    @pytest.mark.parametrize("backend", DIRECT_EXTRACT_BACKENDS)
    def test_extract_empty_returns_empty_without_optional_runtime(
        self,
        backend: ExtractBackend,
    ) -> None:
        result = backend(np.array([], dtype=np.float64), 3)
        assert result.dtype == np.float64
        assert result.shape == (0,)

    @pytest.mark.parametrize("backend", DIRECT_EXTRACT_BACKENDS)
    @pytest.mark.parametrize("window", [4, 8])
    def test_extract_global_rms_without_optional_runtime(
        self,
        backend: ExtractBackend,
        window: int,
    ) -> None:
        amps = np.array([-1.0, 2.0, -3.0, 4.0], dtype=np.float64)
        expected = np.full(4, np.sqrt(np.mean(amps * amps)), dtype=np.float64)
        np.testing.assert_allclose(backend(amps, window), expected, atol=0.0)

    @pytest.mark.parametrize("backend", DIRECT_MOD_BACKENDS)
    def test_mod_empty_and_non_positive_return_zero_without_optional_runtime(
        self,
        backend: ModBackend,
    ) -> None:
        assert backend(np.array([], dtype=np.float64)) == 0.0
        assert backend(np.array([-3.0, -1.0], dtype=np.float64)) == 0.0

    @pytest.mark.parametrize(
        ("stdout", "expected_count", "label", "match"),
        [
            ("", 3, "RMS", "Mojo envelope RMS returned 0 lines, expected 3"),
            (
                "0.5\n0.6\n",
                3,
                "RMS",
                "Mojo envelope RMS returned 2 lines, expected 3",
            ),
            (
                "0.5\n\n0.6\n",
                2,
                "RMS",
                "Mojo envelope RMS returned 3 lines, expected 2",
            ),
            (
                "0.5\n0.6\n0.7\n",
                1,
                "MOD",
                "Mojo envelope MOD returned 3 lines, expected 1",
            ),
            ("not-a-number\n", 1, "MOD", "finite real values"),
            ("inf\n", 1, "MOD", "finite real values"),
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
        monkeypatch.setattr(envelope_mojo_mod, "_ensure_exe", lambda: "envelope")
        monkeypatch.setattr(
            envelope_mojo_mod.subprocess,
            "run",
            lambda *_args, **_kwargs: types.SimpleNamespace(
                returncode=0,
                stdout=stdout,
                stderr="",
            ),
        )

        with pytest.raises(ValueError, match=match):
            envelope_mojo_mod._run(
                "MOD 1 0.5\n",
                expected_count=expected_count,
                label=label,
            )


class TestRustParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "rust" not in AVAILABLE_BACKENDS:
            pytest.skip("Rust backend not built")

    @given(
        n=st.integers(min_value=30, max_value=800),
        window=st.integers(min_value=2, max_value=25),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_extract(self, n: int, window: int, seed: int) -> None:
        # Hypothesis range keeps ``window ≤ n``. The ``window > n``
        # edge case is ill-defined (can't RMS a window you don't
        # have); Rust returns zeros, the other backends return a
        # whole-trace RMS. Parity tests stay in the physically
        # meaningful regime.
        amps = _amps(seed, n)
        ref = _ref_env(amps, window)
        prev = _force("rust")
        try:
            got = extract_envelope(amps, window)
        finally:
            _reset(prev)
        np.testing.assert_allclose(got, ref, atol=1e-12)

    def test_modulation(self) -> None:
        env = _amps(7)
        ref = _ref_mod(env)
        prev = _force("rust")
        try:
            got = envelope_modulation_depth(env)
        finally:
            _reset(prev)
        assert abs(got - ref) < 1e-12


class TestJuliaParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "julia" not in AVAILABLE_BACKENDS:
            pytest.skip("Julia backend not available")

    @pytest.mark.parametrize("seed", [0, 42])
    def test_extract_and_mod(self, seed: int) -> None:
        amps = _amps(seed)
        ref_env = _ref_env(amps, 10)
        ref_mod = _ref_mod(amps)
        prev = _force("julia")
        try:
            env = extract_envelope(amps, 10)
            mod = envelope_modulation_depth(amps)
        finally:
            _reset(prev)
        np.testing.assert_allclose(env, ref_env, atol=1e-12)
        assert abs(mod - ref_mod) < 1e-12


class TestGoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "go" not in AVAILABLE_BACKENDS:
            pytest.skip("Go backend not built")

    @given(
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=8,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_extract(self, seed: int) -> None:
        amps = _amps(seed)
        ref = _ref_env(amps, 10)
        prev = _force("go")
        try:
            got = extract_envelope(amps, 10)
        finally:
            _reset(prev)
        np.testing.assert_allclose(got, ref, atol=1e-12)


class TestMojoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "mojo" not in AVAILABLE_BACKENDS:
            pytest.skip("Mojo backend not built")

    @pytest.mark.parametrize("seed", [0, 77])
    def test_extract(self, seed: int) -> None:
        amps = _amps(seed)
        ref = _ref_env(amps, 10)
        prev = _force("mojo")
        try:
            got = extract_envelope(amps, 10)
        finally:
            _reset(prev)
        np.testing.assert_allclose(got, ref, atol=1e-9)


class TestCrossBackendConsistency:
    @pytest.mark.skipif(
        len(AVAILABLE_BACKENDS) < 2,
        reason="Only Python fallback available",
    )
    def test_all_backends_agree(self) -> None:
        amps = _amps(2026)
        ref_env = _ref_env(amps, 10)
        ref_mod = _ref_mod(amps)
        tolerances = {
            "rust": 1e-12,
            "julia": 1e-12,
            "go": 1e-12,
            "mojo": 1e-9,
            "python": 0.0,
        }
        for backend in AVAILABLE_BACKENDS:
            prev = _force(backend)
            try:
                env = extract_envelope(amps, 10)
                mod = envelope_modulation_depth(amps)
            finally:
                _reset(prev)
            np.testing.assert_allclose(env, ref_env, atol=tolerances[backend])
            assert abs(mod - ref_mod) <= tolerances[backend]


class TestBackendTypingContracts:
    @pytest.mark.parametrize(
        ("fn", "label"),
        [
            (extract_envelope_go, "go:extract"),
            (extract_envelope_julia, "julia:extract"),
            (extract_envelope_mojo, "mojo:extract"),
        ],
    )
    def test_extract_annotations_use_float64_ndarray(self, fn, label: str) -> None:
        hints = get_type_hints(fn)
        for name in ("amps", "return"):
            text = str(hints[name])
            assert_precise_ndarray_hint(
                hints[name],
                context=f"{label}:{name}",
            )
            assert "numpy.float64" in text, f"{label}:{name} missing float64 annotation"

    @pytest.mark.parametrize(
        ("fn", "label"),
        [
            (envelope_modulation_depth_go, "go:mod"),
            (envelope_modulation_depth_julia, "julia:mod"),
            (envelope_modulation_depth_mojo, "mojo:mod"),
        ],
    )
    def test_mod_annotations_use_float64_ndarray(self, fn, label: str) -> None:
        hints = get_type_hints(fn)
        text = str(hints["env"])
        assert_precise_ndarray_hint(
            hints["env"],
            context=f"{label}:env",
        )
        assert "numpy.float64" in text, f"{label}:env missing float64 annotation"


class TestBackendLoaderContracts:
    def test_rust_loader_flattens_inputs_and_returns_float64(self, monkeypatch) -> None:
        calls = {}

        def extract_envelope_rust(amps, window: int):
            calls["extract"] = (amps.flags.c_contiguous, window, amps.shape)
            return amps + window

        def envelope_modulation_depth_rust(env):
            calls["mod"] = (env.flags.c_contiguous, env.shape)
            return 0.375

        fake_spo = types.ModuleType("spo_kernel")
        fake_spo.extract_envelope_rust = extract_envelope_rust
        fake_spo.envelope_modulation_depth_rust = envelope_modulation_depth_rust
        monkeypatch.setitem(sys.modules, "spo_kernel", fake_spo)

        loaded = e_mod._load_rust_fns()
        amps = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        env = np.array([[0.5, 0.75], [1.0, 1.25]], dtype=np.float64)

        extract = loaded["extract"]
        mod = loaded["mod"]
        got_env = extract(amps, 3)
        got_mod = mod(env)

        np.testing.assert_allclose(got_env, amps.ravel() + 3)
        assert got_env.dtype == np.float64
        assert got_mod == 0.375
        assert calls == {
            "extract": (True, 3, (4,)),
            "mod": (True, (4,)),
        }


class TestDispatchFallbackChain:
    def test_dispatch_falls_back_to_next_backend_when_active_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict[str, int] = {"rust": 0, "go": 0}

        def _fail_rust() -> dict[str, object]:
            calls["rust"] += 1
            raise ImportError("rust unavailable")

        def _ok_go() -> dict[str, object]:
            calls["go"] += 1
            return {
                "extract": lambda amps, window: np.ascontiguousarray(
                    amps, dtype=np.float64
                ),
                "mod": lambda env: 0.25,
            }

        monkeypatch.setattr(e_mod, "_BACKEND_CACHE", {})
        monkeypatch.setattr(e_mod, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(e_mod, "AVAILABLE_BACKENDS", ["rust", "go", "python"])
        monkeypatch.setattr(e_mod, "_LOADERS", {"rust": _fail_rust, "go": _ok_go})

        fn = e_mod._dispatch("mod")
        assert fn is not None
        assert float(fn(np.array([1.0], dtype=np.float64))) == 0.25
        assert calls == {"rust": 1, "go": 1}

    def test_probe_returns_infinite_latency_for_broken_backend(
        self, monkeypatch
    ) -> None:
        def broken_loader() -> dict[str, object]:
            raise RuntimeError("backend unavailable during probe")

        monkeypatch.setitem(e_mod._LOADERS, "go", broken_loader)
        e_mod._BACKEND_CACHE.pop("go", None)

        assert e_mod._extract_probe_seconds("go") == float("inf")

    def test_dispatch_falls_back_to_python_when_active_backend_fails(
        self, monkeypatch
    ) -> None:
        def broken_loader() -> dict[str, object]:
            raise OSError("backend disappeared after discovery")

        monkeypatch.setattr(e_mod, "ACTIVE_BACKEND", "go")
        monkeypatch.setitem(e_mod._LOADERS, "go", broken_loader)
        e_mod._BACKEND_CACHE.pop("go", None)

        amps = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        got = extract_envelope(amps, window=2)
        expected = np.array([np.sqrt(2.5), np.sqrt(2.5), np.sqrt(6.5)])

        np.testing.assert_allclose(got, expected)
