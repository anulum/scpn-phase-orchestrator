# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Physical signal factory ingress contracts

"""Raw measurement type contracts through the production extractor factory."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.oscillators.factory import build_extractor


@pytest.mark.parametrize("algorithm", ["hilbert", "wavelet", "zero_crossing"])
@pytest.mark.parametrize(
    "dtype",
    [
        "U",
        "S",
        "bool",
        "complex128",
        "object",
        "timedelta64[ms]",
        "timedelta64[ns]",
        "datetime64[ms]",
    ],
)
def test_factory_extractors_reject_non_real_measurement_arrays(
    algorithm: str, dtype: str
) -> None:
    extractor = build_extractor(algorithm, node_id="sensor")
    raw = np.array([0, 1, 0, -1, 0, 1, 0, -1]).astype(dtype)
    with pytest.raises(ValueError, match="finite real numbers without temporal units"):
        extractor.extract(raw, 32.0)


@pytest.mark.parametrize("algorithm", ["hilbert", "wavelet", "zero_crossing"])
def test_factory_extractors_reject_booleans_before_list_promotion(
    algorithm: str,
) -> None:
    extractor = build_extractor(algorithm, node_id="sensor")
    with pytest.raises(ValueError, match="boolean values"):
        extractor.extract(
            cast("NDArray[np.float64]", [0.0, np.bool_(True), 0.0, -1.0]), 32.0
        )


@pytest.mark.parametrize("algorithm", ["hilbert", "wavelet", "zero_crossing"])
@pytest.mark.parametrize("dtype", ["int16", "uint16", "float32", "float64"])
def test_factory_extractors_preserve_numeric_measurement_phase_contract(
    algorithm: str, dtype: str
) -> None:
    sample_rate = 128.0
    frequency = 8.0
    t = np.arange(256) / sample_rate
    raw = (100 + 50 * np.sin(2 * np.pi * frequency * t)).astype(dtype)
    extractor = build_extractor(
        algorithm, node_id="sensor", config={"band": (4.0, 16.0)}
    )
    state = extractor.extract(raw, sample_rate)[0]
    assert state.node_id == "sensor"
    assert state.channel == "P"
    assert np.isfinite(state.amplitude) and state.amplitude > 0
    assert 0.0 <= state.theta < 2 * np.pi
    assert 0.0 <= state.quality <= 1.0
    assert state.omega == pytest.approx(2 * np.pi * frequency, rel=0.12)
