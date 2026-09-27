# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Spectral frequency measurement contracts

"""Frequency source contracts through the public spectral coupling API."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.coupling.spectral import critical_coupling


def test_critical_coupling_rejects_real_objects_outside_float_range() -> None:
    knm = np.array([[0.0, 1.0], [1.0, 0.0]])
    omegas = np.array([10**400, 2], dtype=object)
    with pytest.raises(ValueError, match="finite 1-D frequency vector") as error:
        critical_coupling(omegas, knm)
    assert isinstance(error.value.__cause__, OverflowError)


def test_critical_coupling_preserves_numeric_objects() -> None:
    knm = np.array([[0.0, 1.0], [1.0, 0.0]])
    omegas = np.array([1.0, 2.0])
    result = critical_coupling(omegas.astype(object), knm)
    assert result == pytest.approx(critical_coupling(omegas, knm), abs=1e-12)
    assert result > 0.0
