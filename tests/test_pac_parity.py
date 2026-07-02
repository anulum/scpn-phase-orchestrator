# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — PAC parity tests

"""Cross-validate PAC public contracts and fallback behaviour.

The Python API exposes a dispatcher with optional Rust/alternative kernels.
These tests cover contract behaviour both with and without Rust: malformed
inputs, shape validation, finite bounded outputs, and dispatcher fallback.
"""

from __future__ import annotations

import importlib
from typing import Protocol, TypeAlias, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde import pac

FloatArray: TypeAlias = NDArray[np.float64]


class _SpoKernelPAC(Protocol):
    """Typed subset of the optional Rust PAC extension used by these tests."""

    def pac_modulation_index(
        self,
        theta: FloatArray,
        amp: FloatArray,
        n_bins: int,
    ) -> float:
        """Return the Rust modulation-index result."""

    def pac_matrix_compute(
        self,
        phases_flat: FloatArray,
        amps_flat: FloatArray,
        t: int,
        n: int,
        n_bins: int,
    ) -> FloatArray:
        """Return the flattened Rust PAC-matrix result."""


try:
    _SPO_KERNEL: _SpoKernelPAC | None = cast(
        _SpoKernelPAC,
        importlib.import_module("spo_kernel"),
    )
except Exception:
    _SPO_KERNEL = None


def _reference_matrix(
    phases: FloatArray,
    amplitudes: FloatArray,
    n_bins: int,
) -> FloatArray:
    n = phases.shape[1]
    result = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            result[i, j] = pac._modulation_index_python(
                np.asarray(phases[:, i], dtype=np.float64),
                np.asarray(amplitudes[:, j], dtype=np.float64),
                n_bins,
            )
    return result


def _make_signals(seed: int = 0) -> tuple[FloatArray, FloatArray]:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * np.pi, 500)
    amp = 0.5 + rng.uniform(0.1, 0.9, 500)
    return theta, amp


def test_modulation_index_rejects_invalid_n_bins_types() -> None:
    theta, amp = _make_signals(1)
    for n_bins in (True, 1.5, None, "18", complex(8.0), [18]):
        with pytest.raises(ValueError):
            pac.modulation_index(theta, amp, cast("int", n_bins))


def test_pac_matrix_rejects_invalid_n_bins_types() -> None:
    phases = np.zeros((12, 3), dtype=np.float64)
    amplitudes = np.ones((12, 3), dtype=np.float64)
    for n_bins in (True, 1.5, None, "18", complex(8.0), [18]):
        with pytest.raises(ValueError):
            pac.pac_matrix(phases, amplitudes, cast("int", n_bins))


def test_modulation_index_rejects_malformed_vector_inputs() -> None:
    amp = np.linspace(0.2, 0.4, 10)

    with pytest.raises(ValueError):
        pac.modulation_index(
            cast("FloatArray", np.array([True] * 10, dtype=bool)),
            amp,
            18,
        )

    with pytest.raises(ValueError):
        pac.modulation_index(cast("FloatArray", np.array([[0.1, 0.2]])), amp, 18)

    with pytest.raises(ValueError):
        pac.modulation_index(np.array([0.1, np.nan, 0.2]), amp[:3], 18)


def test_pac_matrix_rejects_malformed_history_inputs() -> None:
    with pytest.raises(ValueError):
        pac.pac_matrix(np.array([0.1, 0.2]), np.array([0.2, 0.3]), 18)

    with pytest.raises(ValueError):
        pac.pac_matrix(np.zeros((8, 3)), np.zeros((8, 2)), 18)

    with pytest.raises(ValueError):
        phases = np.full((4, 3), np.nan)
        amps = np.ones((4, 3))
        pac.pac_matrix(phases, amps, 18)


def test_modulation_index_empty_and_mismatched_lengths() -> None:
    theta = np.array([0.0, 0.2, 0.4, 0.6, 0.8], dtype=np.float64)
    amp_short = np.array([1.0, 1.1], dtype=np.float64)
    amp_long = np.array([1.0, 1.1, 0.9, 0.95], dtype=np.float64)

    assert pac.modulation_index(theta, amp_short, 18) == pac._modulation_index_python(
        theta, amp_short, 18
    )
    assert pac.modulation_index(theta, np.array([], dtype=np.float64), 18) == 0.0
    assert pac.modulation_index(np.array([], dtype=np.float64), amp_long, 18) == 0.0


def test_modulation_index_bounded_and_finite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    amp = np.abs(np.sin(theta)) + 0.25

    monkeypatch.setattr(pac, "_dispatch", lambda _: None)
    mi = pac.modulation_index(theta, amp, 18)

    assert np.isfinite(mi)
    assert 0.0 <= mi <= 1.0


def test_pac_matrix_returns_finite_square_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t, n = 16, 4
    phases = np.linspace(0.0, 2.0 * np.pi, t * n, dtype=np.float64).reshape(t, n)
    amps = np.sin(phases) + 1.5

    monkeypatch.setattr(pac, "_dispatch", lambda _: None)
    mat = pac.pac_matrix(phases, amps, 18)

    assert mat.shape == (n, n)
    assert np.all(np.isfinite(mat))
    assert np.all(mat >= 0.0)
    assert np.all(mat <= 1.0)


def test_pac_matrix_matches_python_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    phases = np.array(
        [[0.0, 1.0, 2.0], [1.2, 2.8, 4.2], [2.4, 4.0, 0.8], [3.6, 5.6, 2.2]],
        dtype=np.float64,
    )
    amps = np.array(
        [[1.1, 0.6, 1.0], [0.9, 1.2, 1.4], [0.8, 1.1, 0.5], [1.3, 0.7, 1.5]],
        dtype=np.float64,
    )

    monkeypatch.setattr(pac, "_dispatch", lambda _: None)
    observed = pac.pac_matrix(phases, amps, 12)
    expected = _reference_matrix(phases, amps, 12)

    assert observed.shape == expected.shape
    assert np.allclose(observed, expected, atol=1e-12)


def test_pac_matrix_layout_and_empty_inputs() -> None:
    phases = np.zeros((0, 3), dtype=np.float64)
    amps = np.zeros((0, 3), dtype=np.float64)

    matrix = pac.pac_matrix(phases, amps, 18)
    assert matrix.shape == (3, 3)
    assert np.all(matrix == 0.0)


def test_modulation_index_falls_back_when_backend_returns_invalid_scalar() -> None:
    theta, amp = _make_signals(3)

    def fake_backend(*_args: object) -> float:
        return float("nan")

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(pac, "_dispatch", lambda _: fake_backend)
        with pytest.raises(ValueError):
            pac.modulation_index(theta, amp, 18)


def test_modulation_index_backend_rejects_out_of_bounds_scalar() -> None:
    theta, amp = _make_signals(4)

    def fake_backend(*_args: object) -> float:
        return 1.5

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(pac, "_dispatch", lambda _: fake_backend)
        with pytest.raises(ValueError, match="modulation index must lie in"):
            pac.modulation_index(theta, amp, 18)


def test_pac_matrix_rejects_backend_layout_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phases = np.zeros((6, 2), dtype=np.float64)
    amps = np.ones((6, 2), dtype=np.float64)

    def malformed_backend(*_args: object) -> list[float]:
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(pac, "_dispatch", lambda _: malformed_backend)

    with pytest.raises(ValueError):
        pac.pac_matrix(phases, amps, 18)


def test_pac_gate_threshold_contract() -> None:
    assert pac.pac_gate(0.2, 0.3) is False
    assert pac.pac_gate(0.3, 0.3) is True


def test_pac_gate_rejects_invalid_inputs() -> None:
    for pac_value in (True, float("nan"), float("inf"), "0.4", object()):
        with pytest.raises(ValueError):
            pac.pac_gate(cast("float", pac_value), 0.3)

    for threshold in (True, float("nan"), float("inf"), "0.4", object()):
        with pytest.raises(ValueError):
            pac.pac_gate(0.4, cast("float", threshold))


@pytest.mark.parametrize("n_bins", (4, 12, 24))
def test_modulation_index_matches_rust_when_available(n_bins: int) -> None:
    if _SPO_KERNEL is None:
        return

    theta, amp = _make_signals(2)
    expected = _SPO_KERNEL.pac_modulation_index(theta, amp, n_bins)
    observed = pac.modulation_index(theta, amp, n_bins)

    assert abs(observed - expected) < 1e-10


@pytest.mark.parametrize("n_bins", (4, 12, 24))
def test_pac_matrix_matches_rust_diagonal_when_available(n_bins: int) -> None:
    if _SPO_KERNEL is None:
        return

    t, n = 12, 3
    rng = np.random.default_rng(13)
    phases = rng.uniform(0.0, 2.0 * np.pi, (t, n))
    amps = np.abs(np.sin(phases)) + 0.2

    result = pac.pac_matrix(phases, amps, n_bins)
    flat = _SPO_KERNEL.pac_matrix_compute(
        np.asarray(phases, dtype=np.float64).ravel(order="C"),
        np.asarray(amps, dtype=np.float64).ravel(order="C"),
        t,
        n,
        n_bins,
    )
    rust_mat = cast("FloatArray", np.asarray(flat, dtype=np.float64).reshape(n, n))

    assert result.shape == (n, n)
    for i in range(n):
        assert abs(result[i, i] - rust_mat[i, i]) < 1e-12
