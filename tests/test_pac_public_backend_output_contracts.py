# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (C) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (C) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - PAC public backend output contracts

"""Public dispatcher contracts for phase-amplitude-coupling backend outputs."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde import pac as pac_mod
from scpn_phase_orchestrator.upde.pac import modulation_index, pac_matrix

FloatArray: TypeAlias = NDArray[np.float64]
MIBackendFn: TypeAlias = Callable[[FloatArray, FloatArray, int], object]
MatrixBackendFn: TypeAlias = Callable[
    [FloatArray, FloatArray, int, int, int],
    object,
]


def _install_optional_backend(
    monkeypatch: pytest.MonkeyPatch,
    *,
    modulation_output: object,
    matrix_output: object,
) -> None:
    """Install deterministic optional PAC backend callables for public tests."""

    def modulation_backend(
        _theta_low: FloatArray,
        _amp_high: FloatArray,
        _n_bins: int,
    ) -> object:
        """Return the configured modulation-index payload."""
        return modulation_output

    def matrix_backend(
        _phases_flat: FloatArray,
        _amplitudes_flat: FloatArray,
        _t: int,
        _n: int,
        _n_bins: int,
    ) -> object:
        """Return the configured flattened PAC-matrix payload."""
        return matrix_output

    def load_backend() -> dict[str, object]:
        """Return the configured optional backend table."""
        return {
            "modulation_index": modulation_backend,
            "pac_matrix": matrix_backend,
        }

    monkeypatch.setattr(pac_mod, "_BACKEND_CACHE", {})
    monkeypatch.setattr(pac_mod, "ACTIVE_BACKEND", "rust")
    monkeypatch.setattr(pac_mod, "AVAILABLE_BACKENDS", ["rust", "python"])
    monkeypatch.setattr(pac_mod, "_LOADERS", {"rust": load_backend})


def _signals() -> tuple[FloatArray, FloatArray]:
    """Return a small public modulation-index input pair."""
    theta = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False, dtype=np.float64)
    amplitude = 1.0 + 0.25 * np.cos(theta)
    return theta, amplitude


def _histories() -> tuple[FloatArray, FloatArray]:
    """Return a small public PAC-matrix input pair."""
    theta = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False, dtype=np.float64)
    phases = np.column_stack((theta, theta + 0.2)).astype(np.float64, copy=False)
    amplitudes = (1.0 + 0.25 * np.cos(phases)).astype(np.float64, copy=False)
    return phases, amplitudes


def test_public_modulation_index_rejects_out_of_range_optional_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject optional backend modulation-index values outside [0, 1]."""
    _install_optional_backend(
        monkeypatch,
        modulation_output=1.25,
        matrix_output=np.zeros(4, dtype=np.float64),
    )
    theta, amplitude = _signals()

    with pytest.raises(ValueError, match="modulation index must lie in \\[0, 1\\]"):
        modulation_index(theta, amplitude, n_bins=4)


def test_public_pac_matrix_rejects_out_of_range_optional_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject optional backend PAC matrix entries outside [0, 1]."""
    _install_optional_backend(
        monkeypatch,
        modulation_output=0.5,
        matrix_output=np.array([0.0, 0.25, 1.2, 1.0], dtype=np.float64),
    )
    phases, amplitudes = _histories()

    with pytest.raises(ValueError, match="PAC matrix values must lie in \\[0, 1\\]"):
        pac_matrix(phases, amplitudes, n_bins=4)


def test_public_pac_matrix_rejects_boolean_optional_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject boolean optional backend PAC matrices before numeric coercion."""
    _install_optional_backend(
        monkeypatch,
        modulation_output=0.5,
        matrix_output=np.array([True, False, False, True]),
    )
    phases, amplitudes = _histories()

    with pytest.raises(TypeError, match="PAC matrix must be real-valued"):
        pac_matrix(phases, amplitudes, n_bins=4)


def test_public_pac_accepts_valid_optional_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return validated optional PAC outputs when payloads are physical."""
    expected_matrix = np.array([0.0, 0.2, 0.3, 1.0], dtype=np.float64)
    _install_optional_backend(
        monkeypatch,
        modulation_output=0.125,
        matrix_output=expected_matrix,
    )
    theta, amplitude = _signals()
    phases, amplitudes = _histories()

    assert modulation_index(theta, amplitude, n_bins=4) == pytest.approx(0.125)
    np.testing.assert_allclose(
        pac_matrix(phases, amplitudes, n_bins=4),
        expected_matrix.reshape((2, 2), order="C"),
    )


def test_public_pac_uses_python_floor_after_stale_loader_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the Python floor when stale optional backend metadata cannot load."""

    def fail_loader() -> dict[str, object]:
        """Raise the optional-backend unavailability signal."""
        raise ImportError("compiled PAC backend unavailable")

    monkeypatch.setattr(pac_mod, "_BACKEND_CACHE", {})
    monkeypatch.setattr(pac_mod, "ACTIVE_BACKEND", "go")
    monkeypatch.setattr(pac_mod, "AVAILABLE_BACKENDS", ["go"])
    monkeypatch.setattr(pac_mod, "_LOADERS", {"go": fail_loader})

    theta, amplitude = _signals()
    phases, amplitudes = _histories()

    assert 0.0 <= modulation_index(theta, amplitude, n_bins=4) <= 1.0
    np.testing.assert_allclose(
        pac_matrix(phases, amplitudes, n_bins=4),
        np.array(
            [
                [0.011391499268758763, 0.01138224790719715],
                [0.011391499268758763, 0.01138224790719715],
            ],
            dtype=np.float64,
        ),
    )


def test_public_probe_marks_unloadable_optional_backend_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Map optional backend loader failures to an infinite timing probe."""

    def fail_loader() -> dict[str, object]:
        """Raise the optional-backend unavailability signal."""
        raise ImportError("compiled PAC backend unavailable")

    monkeypatch.setattr(pac_mod, "_BACKEND_CACHE", {})
    monkeypatch.setattr(pac_mod, "_LOADERS", {"rust": fail_loader})

    assert pac_mod._modulation_index_probe_seconds("rust") == float("inf")
