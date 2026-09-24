# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — a non-finite gradient leaves the SSGF carrier intact

"""A NaN from the cost callback must not corrupt the geometry carrier.

``GeometryCarrier.update`` subtracted the finite-difference gradient from z
before checking it; one NaN cost wrote NaN into z, the post-update decode
raised, and every later decode failed: the carrier could not recover.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.ssgf.carrier import GeometryCarrier


def test_nan_cost_is_refused_and_state_is_preserved() -> None:
    carrier = GeometryCarrier(4, seed=0)
    before_w = carrier.decode().copy()
    calls = {"n": 0}

    def flaky(w: np.ndarray) -> float:
        calls["n"] += 1
        return float("nan") if calls["n"] == 3 else float(np.sum(w))

    with pytest.raises(ValueError, match="non-finite gradient"):
        carrier.update(cost=1.0, cost_fn=flaky)
    np.testing.assert_array_equal(carrier.decode(), before_w)
    state = carrier.update(cost=1.0, cost_fn=lambda w: float(np.sum(w**2)))
    assert state.step == 1
    assert np.all(np.isfinite(carrier.decode()))


def test_finite_cost_still_descends() -> None:
    carrier = GeometryCarrier(4, seed=1)

    def energy(w: np.ndarray) -> float:
        return float(np.sum(w**2))

    first = energy(carrier.decode())
    for _ in range(5):
        carrier.update(cost=energy(carrier.decode()), cost_fn=energy)
    assert energy(carrier.decode()) < first
