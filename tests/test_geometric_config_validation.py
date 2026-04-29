# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — geometric engine config validation tests

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.geometric import TorusEngine


@pytest.mark.parametrize("n_oscillators", [False, 0, -1, 1.5, "4"])
def test_torus_engine_rejects_invalid_oscillator_count(
    n_oscillators: Any,
) -> None:
    with pytest.raises(ValueError, match="n_oscillators must be >= 1"):
        TorusEngine(n_oscillators=n_oscillators, dt=0.01)


@pytest.mark.parametrize("dt", [False, 0.0, -0.01, float("nan"), float("inf"), "0.01"])
def test_torus_engine_rejects_invalid_timestep(dt: Any) -> None:
    with pytest.raises(ValueError, match="dt must be positive"):
        TorusEngine(n_oscillators=4, dt=dt)


def test_torus_engine_normalises_accepted_numpy_scalars() -> None:
    engine = TorusEngine(n_oscillators=np.int64(4), dt=np.float64(0.01))

    assert engine._n == 4
    assert pytest.approx(0.01) == engine._dt
