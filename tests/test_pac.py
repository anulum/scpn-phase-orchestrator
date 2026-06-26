# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Phase-amplitude coupling tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.upde import pac as pac_mod
from scpn_phase_orchestrator.upde.pac import modulation_index, pac_gate, pac_matrix


def test_modulation_index_is_bounded_for_locked_amplitude_envelope() -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    amplitude = 1.0 + np.cos(theta)

    mi = modulation_index(theta, amplitude, n_bins=18)

    assert 0.0 < mi < 1.0


def test_modulation_index_is_zero_for_uniform_amplitude() -> None:
    theta = np.linspace(0.0, 4.0 * np.pi, 720, endpoint=False)
    amplitude = np.ones_like(theta)

    assert modulation_index(theta, amplitude, n_bins=18) == pytest.approx(0.0)


def test_modulation_index_rejects_negative_amplitudes() -> None:
    theta = np.array([0.0, 0.2, 0.4], dtype=np.float64)
    amplitude = np.array([1.0, -0.1, 1.0], dtype=np.float64)

    with pytest.raises(ValueError, match="non-negative amplitudes"):
        modulation_index(theta, amplitude)


def test_modulation_index_rejects_non_numeric_vector_inputs() -> None:
    amplitude = np.ones(1, dtype=np.float64)

    with pytest.raises(ValueError, match="theta_low must be numeric"):
        modulation_index(np.array([object()], dtype=object), amplitude)


def test_modulation_index_zero_amplitude_envelope_has_zero_mi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    amplitude = np.zeros_like(theta)
    monkeypatch.setattr(pac_mod, "ACTIVE_BACKEND", "python")
    monkeypatch.setattr(pac_mod, "AVAILABLE_BACKENDS", ["python"])

    assert modulation_index(theta, amplitude, n_bins=18) == 0.0


def test_pac_matrix_rejects_negative_amplitude_history() -> None:
    phases = np.zeros((3, 2), dtype=np.float64)
    amplitudes = np.ones((3, 2), dtype=np.float64)
    amplitudes[1, 0] = -0.2

    with pytest.raises(ValueError, match="amplitudes_history"):
        pac_matrix(phases, amplitudes)


def test_pac_matrix_rejects_bool_and_non_numeric_histories() -> None:
    amplitudes = np.ones((2, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="phases_history must not contain boolean"):
        pac_matrix(np.zeros((2, 2), dtype=bool), amplitudes)
    with pytest.raises(ValueError, match="phases_history must be numeric"):
        pac_matrix(np.array([[object(), object()]], dtype=object), amplitudes[:1])


def test_pac_matrix_shape_and_range_for_cross_channel_history() -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    phases = np.column_stack([theta, theta + np.pi / 2.0])
    amplitudes = np.column_stack([1.0 + np.cos(theta), np.ones_like(theta)])

    matrix = pac_matrix(phases, amplitudes, n_bins=18)

    assert matrix.shape == (2, 2)
    assert np.all(np.isfinite(matrix))
    assert np.all(matrix >= 0.0)
    assert np.all(matrix <= 1.0)
    assert matrix[0, 0] > matrix[0, 1]


@pytest.mark.parametrize("n_bins", [True, 1, 0, -3])
def test_pac_rejects_invalid_bin_counts(n_bins: object) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    amplitude = np.ones_like(theta)

    with pytest.raises(ValueError, match="n_bins"):
        modulation_index(theta, amplitude, n_bins=n_bins)


@pytest.mark.parametrize(
    ("value", "threshold", "expected"),
    [(0.31, 0.3, True), (0.29, 0.3, False), (0.3, 0.3, True)],
)
def test_pac_gate_uses_closed_threshold(
    value: float,
    threshold: float,
    expected: bool,
) -> None:
    assert pac_gate(value, threshold) is expected


def test_modulation_index_rejects_invalid_backend_scalar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    amplitude = np.ones_like(theta)

    def invalid_backend(*_args: object) -> bool:
        return True

    monkeypatch.setattr(pac_mod, "_dispatch", lambda _name: invalid_backend)

    with pytest.raises(ValueError, match="modulation_index backend"):
        modulation_index(theta, amplitude, n_bins=18)


def test_pac_matrix_rejects_non_finite_backend_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phases = np.zeros((3, 2), dtype=np.float64)
    amplitudes = np.ones((3, 2), dtype=np.float64)

    def non_finite_backend(*_args: object) -> np.ndarray:
        return np.array([0.1, np.nan, 0.2, 0.3], dtype=np.float64)

    monkeypatch.setattr(pac_mod, "_dispatch", lambda _name: non_finite_backend)

    with pytest.raises(ValueError, match="pac_matrix backend must return finite"):
        pac_matrix(phases, amplitudes, n_bins=18)


def test_public_dispatch_falls_back_across_missing_duplicate_and_empty_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    amplitude = np.ones_like(theta)

    monkeypatch.setattr(pac_mod, "ACTIVE_BACKEND", "ghost")
    monkeypatch.setattr(pac_mod, "AVAILABLE_BACKENDS", ["ghost", "python"])
    monkeypatch.setattr(pac_mod, "_BACKEND_CACHE", {})
    monkeypatch.setattr(
        pac_mod,
        "_LOADERS",
        {"ghost": lambda: {"missing": lambda *_args: 1.0}},
    )

    assert modulation_index(theta, amplitude, n_bins=18) == 0.0


def test_public_dispatch_falls_back_when_backend_lookup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    amplitude = np.ones_like(theta)

    monkeypatch.setattr(pac_mod, "ACTIVE_BACKEND", "missing")
    monkeypatch.setattr(pac_mod, "AVAILABLE_BACKENDS", [])
    monkeypatch.setattr(pac_mod, "_BACKEND_CACHE", {})

    assert modulation_index(theta, amplitude, n_bins=18) == 0.0


def test_backend_resolution_reports_python_floor_when_optional_loaders_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_loader() -> dict[str, object]:
        raise ImportError("optional backend unavailable")

    monkeypatch.setattr(pac_mod, "_BACKEND_CACHE", {})
    monkeypatch.setattr(
        pac_mod,
        "_LOADERS",
        {
            "rust": fail_loader,
            "mojo": fail_loader,
            "julia": fail_loader,
            "go": fail_loader,
        },
    )

    active, available = pac_mod._resolve_backends()

    assert active == "python"
    assert available == ["python"]
