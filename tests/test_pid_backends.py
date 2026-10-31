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

import cmath
import collections
import math
from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

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


def _problem(
    seed: int, t: int = 1500, n: int = 8
) -> tuple[NDArray[np.float64], list[int], list[int]]:
    rng = np.random.default_rng(seed)
    history = rng.uniform(0, TWO_PI, (t, n))
    na = n // 2
    return history, list(range(na)), list(range(na, n))


def _reference(
    history: NDArray[np.float64], ga: list[int], gb: list[int], n_bins: int
) -> tuple[float, float]:
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


def test__pid_validation_helper_is_directly_linked_to_backend_tests() -> None:
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
            (
                np.array(["0.0", "1.0", "2.0", "3.0"], dtype=object),
                2,
                2,
                [0],
                [1],
                8,
                "numeric-string",
            ),
            (np.array([0.0, np.nan, 0.0, 0.0]), 2, 2, [0], [1], 8, "finite"),
            (np.array([0.0, 1.0 + 0j, 0.0, 0.0]), 2, 2, [0], [1], 8, "real-valued"),
            (np.zeros(3), 2, 2, [0], [1], 8, "t\\*n"),
            (np.zeros(4), 2, 2, [0], [1], 1, "n_bins"),
            (np.zeros(4), 2, 2, [9], [1], 8, r"\[0, 2\)"),
            (np.zeros(4), 2, 2, ["0"], [1], 8, "numeric-string"),
            (np.zeros(4), 2, 2, [0], [True], 8, "boolean"),
            (np.zeros(4), 2, 1, [0], [0], 8, "t\\*n"),
        ],
    )
    def test_validation_precedes_runtime_load(
        self,
        backend: Callable[..., tuple[float, float]],
        history: NDArray[np.generic],
        t: int,
        n: int,
        ga: list[object],
        gb: list[object],
        n_bins: int,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            backend(history, t, n, np.asarray(ga), np.asarray(gb), n_bins)


def _matrix_backend(*args: object) -> tuple[float, float]:
    return (0.5, 0.25)


def _oracle_pid(
    rows: list[list[float]], group_a: list[int], group_b: list[int], n_bins: int
) -> tuple[float, float]:
    """Return Williams & Beer ``(I_min redundancy, synergy)`` by plain counting."""

    def phase_bin(values: list[float]) -> int:
        """Bin the order-parameter phase of ``values`` into equal arcs."""
        angle = cmath.phase(sum(cmath.exp(1j * v) for v in values) / len(values))
        index = math.floor((angle % (2 * math.pi)) / (2 * math.pi / n_bins))
        return min(max(index, 0), n_bins - 1)

    y = [phase_bin(r) for r in rows]
    a = [phase_bin([r[k] for k in group_a]) for r in rows]
    b = [phase_bin([r[k] for k in group_b]) for r in rows]
    total = len(rows)
    py = collections.Counter(y)

    def mutual_information(x: list[object]) -> float:
        """Return ``MI(X; Y)`` from the joint and marginal frequencies."""
        px, pxy = collections.Counter(x), collections.Counter(zip(x, y, strict=True))
        return max(
            0.0,
            math.fsum(
                c / total * math.log((c / total) / (px[xv] / total * py[yv] / total))
                for (xv, yv), c in pxy.items()
            ),
        )

    def specific(x: list[int], yv: int) -> float:
        """Return the specific information ``I(Y = y; X)``."""
        px, pxy = collections.Counter(x), collections.Counter(zip(x, y, strict=True))
        return math.fsum(
            pxy[(xv, yv)]
            / py[yv]
            * math.log((pxy[(xv, yv)] / px[xv]) / (py[yv] / total))
            for xv in px
            if pxy[(xv, yv)] > 0
        )

    redundancy = max(
        0.0,
        math.fsum(py[yv] / total * min(specific(a, yv), specific(b, yv)) for yv in py),
    )
    pair: list[object] = list(zip(a, b, strict=True))
    synergy = max(
        0.0,
        mutual_information(pair)
        - mutual_information(list(a))
        - mutual_information(list(b))
        + redundancy,
    )
    return redundancy, synergy


class TestBackendLoaderDispatch:
    @pytest.mark.native_runtime
    def test_rust_loader_wraps_spo_kernel(self) -> None:
        """The compiled Rust loader returns the Williams & Beer PID of the oracle."""
        if "rust" not in pid_module.AVAILABLE_BACKENDS:
            pytest.skip("rust unavailable")
        backend = pid_module._load_rust_fn()
        rng = np.random.default_rng(71)
        history = rng.uniform(-np.pi, np.pi, (60, 5)) + np.linspace(0, 6, 60)[:, None]
        group_a, group_b = np.array([0, 1]), np.array([2, 3, 4])
        for n_bins in (4, 8):
            red, syn = backend(history.ravel(), 60, 5, group_a, group_b, n_bins)
            exp_red, exp_syn = _oracle_pid(history.tolist(), [0, 1], [2, 3, 4], n_bins)
            assert red == pytest.approx(exp_red, abs=1e-12)
            assert syn == pytest.approx(exp_syn, abs=1e-12)

    def test_resolve_backends_falls_back_to_python(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _fail() -> pid_module.PidBackend:
            raise RuntimeError("unavailable")

        for name in ("rust", "mojo", "julia", "go"):
            monkeypatch.setitem(pid_module._LOADERS, name, _fail)
        active, available = pid_module._resolve_backends()
        assert active == "python"
        assert available == ["python"]
