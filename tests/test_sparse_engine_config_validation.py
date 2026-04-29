# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — sparse engine config validation tests

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.sparse_engine import SparseUPDEEngine


@pytest.mark.parametrize("n_oscillators", [False, 0, -1, 1.5, "4"])
def test_sparse_engine_rejects_invalid_oscillator_count(
    n_oscillators: Any,
) -> None:
    with pytest.raises(ValueError, match="n_oscillators must be >= 1"):
        SparseUPDEEngine(n_oscillators=n_oscillators, dt=0.01)


@pytest.mark.parametrize("dt", [False, 0.0, -0.01, float("nan"), float("inf"), "0.01"])
def test_sparse_engine_rejects_invalid_timestep(dt: Any) -> None:
    with pytest.raises(ValueError, match="dt must be a finite positive real"):
        SparseUPDEEngine(n_oscillators=4, dt=dt)


@pytest.mark.parametrize("field", ["atol", "rtol"])
@pytest.mark.parametrize(
    "value", [False, 0.0, -1e-6, float("nan"), float("inf"), "1e-6"]
)
def test_sparse_engine_rejects_invalid_tolerances(
    field: str,
    value: Any,
) -> None:
    kwargs: dict[str, Any] = {"n_oscillators": 4, "dt": 0.01, field: value}

    with pytest.raises(ValueError, match=f"{field} must be a finite positive real"):
        SparseUPDEEngine(**kwargs)


def test_sparse_engine_normalises_accepted_numpy_scalars() -> None:
    engine = SparseUPDEEngine(
        n_oscillators=np.int64(4),
        dt=np.float64(0.01),
        atol=np.float64(1e-6),
        rtol=np.float64(1e-3),
    )

    assert engine._n == 4
    assert pytest.approx(0.01) == engine._dt
    assert pytest.approx(1e-6) == engine._atol
    assert pytest.approx(1e-3) == engine._rtol
