# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.oscillators.symbolic import SymbolicExtractor

TWO_PI = 2.0 * np.pi


def test_ring_phase_mapping():
    """States [0,1,2,3] with N=4 -> theta = [0, pi/2, pi, 3*pi/2]."""
    extractor = SymbolicExtractor(n_states=4, mode="ring")
    states = extractor.extract(np.array([0, 1, 2, 3]), sample_rate=1.0)
    thetas = [s.theta for s in states]
    expected = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
    np.testing.assert_allclose(thetas, expected, atol=1e-12)


def test_graph_walk_phases_in_range():
    extractor = SymbolicExtractor(n_states=10, mode="graph")
    signal = np.array([3, 5, 7, 2, 9])
    states = extractor.extract(signal, sample_rate=1.0)
    for s in states:
        assert 0.0 <= s.theta < TWO_PI


def test_stalled_state_penalised():
    """Repeated identical state should get low quality (0.2)."""
    extractor = SymbolicExtractor(n_states=5, mode="ring")
    signal = np.array([2, 2, 2, 2])
    states = extractor.extract(signal, sample_rate=1.0)
    # First state gets default 0.5, subsequent stalled states get 0.2
    for s in states[1:]:
        assert s.quality == pytest.approx(0.2)


def test_single_step_transitions_high_quality():
    extractor = SymbolicExtractor(n_states=8, mode="ring")
    signal = np.array([0, 1, 2, 3, 4])
    states = extractor.extract(signal, sample_rate=1.0)
    for s in states[1:]:
        assert s.quality == pytest.approx(1.0)


def test_channel_is_symbolic():
    extractor = SymbolicExtractor(n_states=4, node_id="sym_q")
    states = extractor.extract(np.array([0, 1]), sample_rate=1.0)
    assert states[0].channel == "S"
    assert states[0].node_id == "sym_q"


def test_n_states_below_two_raises():
    with pytest.raises(ValueError, match="n_states must be >= 2"):
        SymbolicExtractor(n_states=1)


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode must be"):
        SymbolicExtractor(n_states=4, mode="invalid")
