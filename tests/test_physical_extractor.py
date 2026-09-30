# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Physical extractor contracts

"""Validate physical extraction through its public waveform interface."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.oscillators.physical import PhysicalExtractor


class TestPhysicalExtractor:
    """Public input refusal and returned phase-state contracts."""

    def test_invalid_signal_shape(self) -> None:
        """Multichannel arrays require separate per-channel extraction."""
        with pytest.raises(ValueError, match="1-D"):
            PhysicalExtractor().extract(np.zeros((2, 3)), 1000.0)

    def test_single_sample_raises(self) -> None:
        """A single sample cannot supply the phase-gradient frequency."""
        with pytest.raises(ValueError, match=">= 2"):
            PhysicalExtractor().extract(np.array([1.0]), 1000.0)

    def test_zero_envelope_quality(self) -> None:
        """A silent waveform returns zero amplitude and quality publicly."""
        state = PhysicalExtractor().extract(np.zeros(100), 1000.0)[0]
        assert state.amplitude == 0.0
        assert state.quality == 0.0

    def test_extract_periodic_waveform(self) -> None:
        """The configured node returns the analytic periodic phase and frequency."""
        time = np.arange(500, dtype=np.float64) / 1000.0
        phase = 2.0 * np.pi * 10.0 * time
        states = PhysicalExtractor(node_id="periodic").extract(np.sin(phase), 1000.0)
        assert len(states) == 1
        state = states[0]
        assert state.theta == pytest.approx(
            float((phase[-1] - np.pi / 2.0) % (2.0 * np.pi)), abs=1e-12
        )
        assert state.omega == pytest.approx(20.0 * np.pi, abs=1e-12)
        assert state.amplitude == pytest.approx(1.0, abs=1e-12)
        assert state.quality == pytest.approx(1.0, abs=1e-12)
        assert state.channel == "P"
        assert state.node_id == "periodic"
