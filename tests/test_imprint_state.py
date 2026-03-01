# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.imprint.state import ImprintState


def test_creation_with_array():
    m_k = np.array([0.1, 0.2, 0.3, 0.4])
    state = ImprintState(m_k=m_k, last_update=0.0)
    np.testing.assert_array_equal(state.m_k, m_k)
    assert state.last_update == 0.0


def test_attribution_records_contributions():
    state = ImprintState(
        m_k=np.zeros(4),
        last_update=1.0,
        attribution={"layer_0": 0.5, "layer_1": 0.3},
    )
    assert state.attribution["layer_0"] == 0.5
    assert state.attribution["layer_1"] == 0.3


def test_default_attribution_empty():
    state = ImprintState(m_k=np.zeros(2), last_update=0.0)
    assert state.attribution == {}


def test_m_k_dtype():
    state = ImprintState(m_k=np.array([1, 2, 3], dtype=np.float64), last_update=0.0)
    assert state.m_k.dtype == np.float64
