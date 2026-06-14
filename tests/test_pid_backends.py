# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Per-backend parity for the PID decomposition

"""Cross-backend parity for :func:`redundancy` / :func:`synergy`.

Every accelerated backend must reproduce the NumPy reference redundancy and
synergy within 1e-10 (Rust / Julia / Go) or 1e-6 (Mojo, subprocess text
round-trip). Direct backend adapters validate the phase history, group indices,
and bin count before any runtime loads.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _pid_validation as pid_validation,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._pid_go import (
    pid_decomposition_go,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._pid_julia import (
    pid_decomposition_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._pid_mojo import (
    pid_decomposition_mojo,
)
from scpn_phase_orchestrator.monitor import pid as pid_module
from scpn_phase_orchestrator.monitor.pid import (
    AVAILABLE_BACKENDS,
    redundancy,
    synergy,
)

TWO_PI = 2.0 * np.pi


def _force(backend: str) -> str:
    prev = pid_module.ACTIVE_BACKEND
    pid_module.ACTIVE_BACKEND = backend
    return prev


def _reset(prev: str) -> None:
    pid_module.ACTIVE_BACKEND = prev


def _problem(seed: int, t: int = 1500, n: int = 8):
    rng = np.random.default_rng(seed)
    history = rng.uniform(0, TWO_PI, (t, n))
    na = n // 2
    return history, list(range(na)), list(range(na, n))


def _reference(history, ga, gb, n_bins):
    prev = _force("python")
    try:
        return redundancy(history, ga, gb, n_bins), synergy(history, ga, gb, n_bins)
    finally:
        _reset(prev)


def _assert_parity(backend: str, seed: int, atol: float) -> None:
    history, ga, gb = _problem(seed)
    ref_r, ref_s = _reference(history, ga, gb, 16)
    prev = _force(backend)
    try:
        got_r = redundancy(history, ga, gb, 16)
        got_s = synergy(history, ga, gb, 16)
    finally:
        _reset(prev)
    assert abs(got_r - ref_r) <= atol, f"{backend} redundancy {got_r} vs {ref_r}"
    assert abs(got_s - ref_s) <= atol, f"{backend} synergy {got_s} vs {ref_s}"


def test_validation_helper_is_linked() -> None:
    assert callable(pid_validation.validate_pid_backend_inputs)


class TestRustParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "rust" not in AVAILABLE_BACKENDS:
            pytest.skip("Rust backend not built")

    @pytest.mark.parametrize("seed", [0, 1, 7])
    def test_matches_python(self, seed: int) -> None:
        _assert_parity("rust", seed, atol=1e-10)


class TestJuliaParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "julia" not in AVAILABLE_BACKENDS:
            pytest.skip("Julia backend not available")

    @pytest.mark.parametrize("seed", [0, 42])
    def test_matches_python(self, seed: int) -> None:
        _assert_parity("julia", seed, atol=1e-10)


class TestGoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "go" not in AVAILABLE_BACKENDS:
            pytest.skip("Go backend not built")

    @pytest.mark.parametrize("seed", [0, 13])
    def test_matches_python(self, seed: int) -> None:
        _assert_parity("go", seed, atol=1e-10)


class TestMojoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "mojo" not in AVAILABLE_BACKENDS:
            pytest.skip("Mojo backend not built")

    @pytest.mark.parametrize("seed", [0, 77])
    def test_matches_python(self, seed: int) -> None:
        _assert_parity("mojo", seed, atol=1e-6)


class TestCrossBackendConsistency:
    @pytest.mark.skipif(
        len(AVAILABLE_BACKENDS) < 2,
        reason="Only Python fallback available",
    )
    def test_all_backends_agree(self) -> None:
        history, ga, gb = _problem(2026, t=1200, n=10)
        ref_r, ref_s = _reference(history, ga, gb, 12)
        tol = {"rust": 1e-10, "julia": 1e-10, "go": 1e-10, "mojo": 1e-6, "python": 0.0}
        for backend in AVAILABLE_BACKENDS:
            prev = _force(backend)
            try:
                got_r = redundancy(history, ga, gb, 12)
                got_s = synergy(history, ga, gb, 12)
            finally:
                _reset(prev)
            assert abs(got_r - ref_r) <= tol[backend]
            assert abs(got_s - ref_s) <= tol[backend]


class TestDirectBackendBoundaryContracts:
    @pytest.mark.parametrize(
        "backend",
        [pid_decomposition_go, pid_decomposition_julia, pid_decomposition_mojo],
    )
    @pytest.mark.parametrize(
        ("history", "t", "n", "ga", "gb", "n_bins", "match"),
        [
            (np.array([True, False, False, True]), 2, 2, [0], [1], 8, "phase_history"),
            (np.array([0.0, np.nan, 0.0, 0.0]), 2, 2, [0], [1], 8, "finite"),
            (np.array([0.0, 1.0 + 0j, 0.0, 0.0]), 2, 2, [0], [1], 8, "real-valued"),
            (np.zeros(3), 2, 2, [0], [1], 8, "t\\*n"),
            (np.zeros(4), 2, 2, [0], [1], 1, "n_bins"),
            (np.zeros(4), 2, 2, [9], [1], 8, r"\[0, 2\)"),
            (np.zeros(4), 2, 2, [0], [True], 8, "boolean"),
            (np.zeros(4), 2, 1, [0], [0], 8, "t\\*n"),
        ],
    )
    def test_validation_precedes_runtime_load(
        self, backend, history, t, n, ga, gb, n_bins, match
    ) -> None:
        with pytest.raises(ValueError, match=match):
            backend(history, t, n, np.asarray(ga), np.asarray(gb), n_bins)


def _matrix_backend(*args: object) -> tuple:
    return (0.5, 0.25)


class TestBackendLoaderDispatch:
    def test_rust_loader_wraps_spo_kernel(self, monkeypatch) -> None:
        calls: list[tuple] = []

        def fake_rust(history, t, n, ga, gb, n_bins):
            calls.append((t, n, n_bins))
            return (0.3, 0.7)

        fake_module = types.ModuleType("spo_kernel")
        fake_module.pid_decomposition_rust = fake_rust
        monkeypatch.setitem(sys.modules, "spo_kernel", fake_module)

        backend = pid_module._load_rust_fn()
        red, syn = backend(
            np.zeros(4), 2, 2, np.array([0]), np.array([1]), 8
        )
        assert (red, syn) == (0.3, 0.7)
        assert calls[0] == (2, 2, 8)

    def test_resolve_backends_falls_back_to_python(self, monkeypatch) -> None:
        def _fail() -> pid_module.PidBackend:
            raise RuntimeError("unavailable")

        for name in ("rust", "mojo", "julia", "go"):
            monkeypatch.setitem(pid_module._LOADERS, name, _fail)
        active, available = pid_module._resolve_backends()
        assert active == "python"
        assert available == ["python"]
