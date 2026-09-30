# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Informational timestamp admission tests

"""Exercise timestamp-rank refusal and recovery through phase extraction."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.oscillators.informational import InformationalExtractor
from scpn_phase_orchestrator.upde.engine import UPDEEngine


@pytest.mark.parametrize(
    "shape",
    [(), (1, 4), (4, 1), (1, 2, 2), (0, 2)],
    ids=["scalar", "row", "column", "rank-three", "empty-matrix"],
)
def test_nonvector_timestamps_refuse_without_poisoning_extraction(
    shape: tuple[int, ...],
) -> None:
    """Refuse every nonvector rank before extracting a usable recovery state."""
    timestamps = np.arange(int(np.prod(shape)), dtype=np.float64).reshape(shape)
    original = timestamps.copy()
    extractor = InformationalExtractor(node_id="request_events")

    with pytest.raises(ValueError, match="signal must be 1-D, got shape"):
        extractor.extract(timestamps, sample_rate=0.0)

    np.testing.assert_array_equal(timestamps, original)
    assert timestamps.shape == shape

    valid_timestamps = np.array([0.0, 0.5, 1.0, 1.5])
    valid_original = valid_timestamps.copy()
    states = extractor.extract(valid_timestamps, sample_rate=0.0)
    assert len(states) == 1
    state = states[0]
    assert state.theta == pytest.approx(0.0, abs=1e-12)
    assert state.omega == pytest.approx(4.0 * np.pi)
    assert state.amplitude == pytest.approx(2.0)
    assert state.quality == pytest.approx(1.0)
    assert state.channel == "I"
    assert state.node_id == "request_events"
    assert extractor.quality_score(states) == pytest.approx(1.0)
    np.testing.assert_array_equal(valid_timestamps, valid_original)

    engine = UPDEEngine(n_oscillators=1, dt=0.001)
    phase = engine.step(
        np.array([state.theta]),
        np.array([state.omega]),
        np.zeros((1, 1)),
        0.0,
        0.0,
        np.zeros((1, 1)),
    )
    np.testing.assert_allclose(phase, [0.004 * np.pi], atol=1e-12, rtol=0.0)


def test_empty_timestamp_vector_remains_a_valid_degenerate_train() -> None:
    """Distinguish an empty event vector from an invalid empty event matrix."""
    timestamps = np.array([], dtype=np.float64)
    extractor = InformationalExtractor(node_id="idle_requests")

    states = extractor.extract(timestamps, sample_rate=0.0)

    assert len(states) == 1
    state = states[0]
    assert (state.theta, state.omega, state.amplitude, state.quality) == (
        0.0,
        0.0,
        0.0,
        0.0,
    )
    assert state.channel == "I"
    assert state.node_id == "idle_requests"
    assert extractor.quality_score(states) == 0.0
    assert timestamps.shape == (0,)
