# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Extractor factory tests

"""Tests for `build_extractor`: every extractor_type and channel alias maps to
its concrete `PhaseExtractor`, the built extractor runs, and an unknown type
fails closed.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.oscillators.base import PhaseState
from scpn_phase_orchestrator.oscillators.factory import build_extractor
from scpn_phase_orchestrator.oscillators.informational import InformationalExtractor
from scpn_phase_orchestrator.oscillators.physical import PhysicalExtractor
from scpn_phase_orchestrator.oscillators.symbolic import SymbolicExtractor
from scpn_phase_orchestrator.oscillators.wavelet import WaveletExtractor
from scpn_phase_orchestrator.oscillators.zero_crossing import ZeroCrossingExtractor

TWO_PI = 2.0 * np.pi


@pytest.mark.parametrize(
    ("extractor_type", "expected"),
    [
        ("hilbert", PhysicalExtractor),
        ("physical", PhysicalExtractor),
        ("wavelet", WaveletExtractor),
        ("zero_crossing", ZeroCrossingExtractor),
        ("event", InformationalExtractor),
        ("informational", InformationalExtractor),
        ("ring", SymbolicExtractor),
        ("symbolic", SymbolicExtractor),
        ("graph", SymbolicExtractor),
    ],
)
def test_maps_type_to_extractor(extractor_type: str, expected: type) -> None:
    assert isinstance(build_extractor(extractor_type, n_states=3), expected)


def test_built_extractor_runs() -> None:
    t = np.arange(512) / 256.0
    signal = np.sin(TWO_PI * 8.0 * t)
    out = build_extractor("wavelet").extract(signal, 256.0)
    assert isinstance(out[0], PhaseState)


def test_unknown_type_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="unknown extractor_type"):
        build_extractor("fourier")
