# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Fallback boundaries and numerical edge cases for recurrence monitors."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor import recurrence as recurrence_mod
from scpn_phase_orchestrator.monitor.recurrence import (
    cross_recurrence_matrix,
    recurrence_matrix,
    rqa,
)


def _broken_rust_loader(*_args: object) -> dict[str, object]:
    raise RuntimeError("broken backend")


def _missing_cross_kernel_loader(*_args: object) -> dict[str, object]:
    return {"rm": lambda *_args: np.zeros((0,), dtype=np.uint8)}


def test_recurrence_matrix_falls_back_to_python_when_loader_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = np.array([[0.0], [0.3], [1.2]], dtype=np.float64)

    monkeypatch.setattr(recurrence_mod, "ACTIVE_BACKEND", "python")
    expected = recurrence_matrix(trajectory, 0.75)

    monkeypatch.setattr(recurrence_mod, "ACTIVE_BACKEND", "rust")
    monkeypatch.setattr(recurrence_mod, "AVAILABLE_BACKENDS", ["rust", "python"])
    monkeypatch.setitem(recurrence_mod._LOADERS, "rust", _broken_rust_loader)

    result = recurrence_matrix(trajectory, 0.75)
    np.testing.assert_array_equal(result, expected)


def test_cross_recurrence_matrix_falls_back_to_python_when_kernel_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a = np.array([[0.0], [1.0]], dtype=np.float64)
    b = np.array([[0.0], [2.0]], dtype=np.float64)

    monkeypatch.setattr(recurrence_mod, "ACTIVE_BACKEND", "python")
    expected = cross_recurrence_matrix(a, b, 1.5)

    monkeypatch.setattr(recurrence_mod, "ACTIVE_BACKEND", "go")
    monkeypatch.setattr(recurrence_mod, "AVAILABLE_BACKENDS", ["go", "python"])
    monkeypatch.setitem(recurrence_mod._LOADERS, "go", _missing_cross_kernel_loader)

    result = cross_recurrence_matrix(a, b, 1.5)
    np.testing.assert_array_equal(result, expected)


def test_rqa_constant_trajectory_has_maximal_recurrence_metrics() -> None:
    trajectory = np.array(
        [[1.0], [1.0], [1.0], [1.0]],
        dtype=np.float64,
    )
    result = rqa(trajectory, epsilon=0.5)

    assert result.recurrence_rate == pytest.approx(1.0)
    assert result.determinism == pytest.approx(5 / 12)
    assert result.laminarity == pytest.approx(5 / 6)
    assert result.max_diagonal == 3
    assert result.max_vertical == 3
    assert result.avg_diagonal == pytest.approx(2.5)
    assert result.entropy_diagonal == pytest.approx(np.log(2.0))


def test_recurrence_matrix_rejects_nonfinite_epsilon() -> None:
    trajectory = np.array([[0.0], [1.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="epsilon"):
        recurrence_matrix(trajectory, float("nan"))
