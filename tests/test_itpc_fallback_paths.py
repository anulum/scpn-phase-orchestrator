# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — ITPC fallback path contracts

"""
Fallback and deterministic edge contracts for ITPC monitors.  These tests verify
backend-dispatch failure behaviour, kernel-missing behaviour, and pause-index
filtering for the ITPC public surface.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor import itpc as itpc_mod
from scpn_phase_orchestrator.monitor.itpc import compute_itpc, itpc_persistence


def test_compute_itpc_falls_back_to_python_when_active_backend_loader_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phases = np.array([[0.0, 0.5], [1.0, 1.5]], dtype=np.float64)
    monkeypatch.setattr(itpc_mod, "ACTIVE_BACKEND", "python")
    expected = compute_itpc(phases)

    monkeypatch.setattr(itpc_mod, "ACTIVE_BACKEND", "rust")
    monkeypatch.setattr(itpc_mod, "AVAILABLE_BACKENDS", ["rust", "python"])
    monkeypatch.setitem(
        itpc_mod._LOADERS,
        "rust",
        lambda: (_ for _ in ()).throw(RuntimeError("missing rust backend")),
    )

    result = compute_itpc(phases)
    np.testing.assert_allclose(result, expected)


def test_itpc_persistence_falls_back_to_python_when_persistence_kernel_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phases = np.array([[0.0, 0.3, 0.6], [0.0, 0.3, 0.6]], dtype=np.float64)
    pause_idx = np.array([0, 1, 4], dtype=np.int64)

    monkeypatch.setattr(itpc_mod, "ACTIVE_BACKEND", "python")
    expected = itpc_persistence(phases, pause_idx)

    monkeypatch.setattr(itpc_mod, "ACTIVE_BACKEND", "go")
    monkeypatch.setattr(itpc_mod, "AVAILABLE_BACKENDS", ["go", "python"])
    monkeypatch.setitem(
        itpc_mod._LOADERS,
        "go",
        lambda: {"itpc": lambda *args: np.array([1.0, 1.0, 1.0])},
    )

    result = itpc_persistence(phases, [0, 1, 4])
    assert result == pytest.approx(expected)


def test_itpc_persistence_filters_pause_indices_and_means_valid_positions() -> None:
    phases = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    val = itpc_persistence(phases, [-1, 0, 1, 10, 2, 1])

    # Pause indices are filtered to valid positions before averaging.
    expected = 1.0
    assert val == pytest.approx(expected)
