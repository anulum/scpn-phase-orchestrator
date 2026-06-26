# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — PID defensive contract tests

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor import pid as pid_module
from scpn_phase_orchestrator.monitor.pid import redundancy, synergy


def _history() -> NDArray[np.float64]:
    """Return a deterministic phase history for PID boundary tests."""
    t = np.arange(32, dtype=np.float64)
    return np.column_stack(
        (
            np.sin(t / 3.0),
            np.cos(t / 3.0),
            np.sin(t / 5.0),
            np.cos(t / 5.0),
        )
    )


def test_public_pid_skips_duplicate_failed_accelerated_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed accelerated backend falls through to Python once per name."""

    def _fail_loader() -> pid_module.PidBackend:
        raise RuntimeError("backend unavailable")

    pid_module._BACKEND_CACHE.clear()
    monkeypatch.setattr(pid_module, "ACTIVE_BACKEND", "rust")
    monkeypatch.setattr(pid_module, "AVAILABLE_BACKENDS", ["rust", "python"])
    monkeypatch.setitem(pid_module._LOADERS, "rust", _fail_loader)

    assert redundancy(_history(), [0, 1], [2, 3], 4) >= 0.0


def test_public_pid_uses_python_when_all_accelerated_backends_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If every configured accelerated backend fails, Python remains available."""

    def _fail_loader() -> pid_module.PidBackend:
        raise RuntimeError("backend unavailable")

    pid_module._BACKEND_CACHE.clear()
    monkeypatch.setattr(pid_module, "ACTIVE_BACKEND", "rust")
    monkeypatch.setattr(pid_module, "AVAILABLE_BACKENDS", ["rust"])
    monkeypatch.setitem(pid_module._LOADERS, "rust", _fail_loader)

    assert synergy(_history(), [0, 1], [2, 3], 4) >= 0.0


def test_public_pid_zero_timestep_nonempty_history_returns_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero-timestep history with declared oscillators has no distribution."""
    history = np.empty((0, 2), dtype=np.float64)

    monkeypatch.setattr(pid_module, "ACTIVE_BACKEND", "python")

    assert redundancy(history, [0], [1], 4) == 0.0
    assert synergy(history, [0], [1], 4) == 0.0


def test_public_pid_rejects_object_history_that_cannot_cast() -> None:
    """Object histories must still cast to finite real-valued samples."""
    history = np.array([["bad", "phase"]], dtype=object)

    with pytest.raises(ValueError, match="finite"):
        redundancy(history, [0], [1], 4)


def test_public_pid_rejects_complex_group_indices() -> None:
    """Group indices cannot carry complex payloads."""
    with pytest.raises(TypeError, match="real integer indices"):
        redundancy(_history(), [1 + 0j], [2], 4)  # type: ignore[list-item]


def test_public_pid_rejects_non_castable_group_indices() -> None:
    """Group indices must be numeric integers."""
    with pytest.raises(TypeError, match="integer indices"):
        synergy(_history(), ["bad"], [2], 4)  # type: ignore[list-item]


def test_public_pid_rejects_non_finite_group_indices() -> None:
    """Group indices cannot contain infinities or NaNs."""
    with pytest.raises(ValueError, match="finite integer indices"):
        redundancy(_history(), [np.inf], [2], 4)  # type: ignore[list-item]


def test_public_pid_validates_backend_scalar_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend PID components must be finite and non-negative."""

    def _bad_backend(*_args: Any) -> pid_module.PidTuple:
        return -1.0, 0.0

    monkeypatch.setattr(pid_module, "_dispatch", lambda: _bad_backend)

    with pytest.raises(ValueError, match="redundancy"):
        redundancy(_history(), [0, 1], [2, 3], 4)


class _ArrayRaises:
    """Array protocol object that simulates a NumPy coercion failure."""

    def __array__(self, dtype: object | None = None) -> NDArray[np.float64]:
        raise TypeError("cannot coerce")


def test_pid_alias_detectors_tolerate_array_protocol_failures() -> None:
    """Alias detectors fail closed to no-alias when NumPy refuses the object."""
    value = _ArrayRaises()

    assert pid_module._contains_boolean_alias(value) is False
    assert pid_module._contains_complex_alias(value) is False
    assert pid_module._has_complex_payload(value) is False


def test_pid_information_primitives_return_zero_for_empty_counts() -> None:
    """Zero-count histograms carry no mutual or redundant information."""
    joint = np.zeros((2, 2), dtype=np.float64)
    marginal = np.zeros(2, dtype=np.float64)

    assert pid_module._mutual_information(joint, marginal, marginal) == 0.0
    assert (
        pid_module._i_min_redundancy(joint, joint, marginal, marginal, marginal)
        == 0.0
    )
