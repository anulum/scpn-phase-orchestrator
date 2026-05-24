# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Physical extractor contracts

"""Validation and Python-path contracts for oscillators.physical.PhysicalExtractor."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.oscillators.physical import PhysicalExtractor

TWO_PI = 2.0 * np.pi


class TestPhysicalExtractor:
    def test_invalid_signal_shape(self):
        ext = PhysicalExtractor()
        with pytest.raises(ValueError, match="1-D"):
            ext.extract(np.zeros((2, 3)), 1000.0)

    def test_single_sample_raises(self):
        ext = PhysicalExtractor()
        with pytest.raises(ValueError, match=">= 2"):
            ext.extract(np.array([1.0]), 1000.0)

    def test_zero_envelope_quality(self):
        from scipy.signal import hilbert

        signal = np.zeros(100)
        quality = PhysicalExtractor._envelope_quality(signal, hilbert(signal))
        assert quality == 0.0

    def test_extract_python_path(self, monkeypatch):
        """Force Python fallback by disabling the optional Rust extractor."""
        import scpn_phase_orchestrator.oscillators.physical as phys_mod

        monkeypatch.setattr(phys_mod, "_rust_physical_extract", None)
        ext = PhysicalExtractor(node_id="py_test")
        t = np.arange(0, 0.5, 1.0 / 1000)
        signal = np.sin(TWO_PI * 10.0 * t)
        states = ext.extract(signal, 1000.0)
        assert len(states) == 1
        assert 0.0 <= states[0].theta < TWO_PI
        assert states[0].channel == "P"
        assert states[0].quality > 0.5


# ──────────────────────────────────────────────────────────────────────
# pac.py: force Python fallback for modulation_index, pac_matrix, pac_gate
# ──────────────────────────────────────────────────────────────────────
