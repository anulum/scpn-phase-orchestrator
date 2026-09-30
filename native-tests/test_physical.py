# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Native physical extraction contracts

"""Exercise physical waveform extraction through the actual installed extension."""

from __future__ import annotations

import sys
from types import FrameType

import numpy as np
import pytest
import spo_kernel

from scpn_phase_orchestrator.oscillators.physical import PhysicalExtractor


@pytest.mark.parametrize("length", [127, 128])
@pytest.mark.parametrize("scale", [1.0, 1e160, 1e200, 1e290])
def test_public_extraction_calls_registered_native_owner(
    length: int, scale: float
) -> None:
    """Public extraction invokes the exact registered C callable and retains CV."""
    phase = 2.0 * np.pi * np.arange(length, dtype=np.float64) / length
    signal = scale * (1.0 + 0.6 * np.cos(phase)) * np.cos(8.0 * phase)
    calls: list[object] = []

    def observe(frame: FrameType, event: str, argument: object) -> None:
        """Record real C calls without replacing the native callable or its return."""
        if event == "c_call" and argument is spo_kernel.physical_extract:
            calls.append(argument)

    previous = sys.getprofile()
    sys.setprofile(observe)
    try:
        state = PhysicalExtractor().extract(signal, float(length))[0]
    finally:
        sys.setprofile(previous)
    assert calls == [spo_kernel.physical_extract]
    assert state.amplitude / scale == pytest.approx(1.0, abs=1e-12)
    assert state.quality == pytest.approx(1.0 - 0.6 / np.sqrt(2.0), abs=1e-12)
    assert state.omega == pytest.approx(16.0 * np.pi, rel=1e-12)
    assert state.theta == pytest.approx(float((8 * phase[-1]) % (2 * np.pi)), abs=1e-12)


@pytest.mark.parametrize("scale", [0.0, 1e-17, 1e-15, 1e308])
def test_direct_constant_envelope_retains_mean_and_absolute_quality_gate(
    scale: float,
) -> None:
    """The native mean avoids sum overflow and retains the absolute noise floor."""
    real = np.full(4, scale, dtype=np.float64)
    imag = np.zeros(4, dtype=np.float64)
    real.setflags(write=False)
    theta, omega, amplitude, quality = spo_kernel.physical_extract(real, imag, 1.0)
    assert theta == 0.0
    assert omega == 0.0
    assert amplitude == scale
    assert quality == (0.0 if scale < 1e-15 else 1.0)
    np.testing.assert_array_equal(real, [scale] * 4)


def test_direct_strided_arrays_refuse_and_contiguous_recovery_succeeds() -> None:
    """Readonly slice admission refuses strides without changing source data."""
    real = np.ones(8, dtype=np.float64)[::2]
    imag = np.zeros(4, dtype=np.float64)
    real.setflags(write=False)
    with pytest.raises(ValueError, match="contiguous"):
        spo_kernel.physical_extract(real, imag, 1.0)
    assert spo_kernel.physical_extract(np.ascontiguousarray(real), imag, 1.0) == (
        0.0,
        0.0,
        1.0,
        1.0,
    )
    np.testing.assert_array_equal(real, [1.0] * 4)
