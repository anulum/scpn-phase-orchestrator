# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Driver tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.drivers.psi_informational import InformationalDriver
from scpn_phase_orchestrator.drivers.psi_physical import PhysicalDriver
from scpn_phase_orchestrator.drivers.psi_symbolic import SymbolicDriver


def test_physical_sinusoidal():
    drv = PhysicalDriver(frequency=1.0, amplitude=2.0)
    assert drv.compute(0.0) == pytest.approx(0.0, abs=1e-12)
    assert drv.compute(0.25) == pytest.approx(2.0, abs=1e-6)


def test_physical_batch():
    drv = PhysicalDriver(frequency=1.0)
    t = np.array([0.0, 0.25, 0.5])
    out = drv.compute_batch(t)
    assert out.shape == (3,)
    assert out[0] == pytest.approx(0.0, abs=1e-12)


def test_physical_bad_frequency():
    with pytest.raises(ValueError, match="positive"):
        PhysicalDriver(frequency=-1.0)


def test_informational_ramp():
    drv = InformationalDriver(cadence_hz=1.0)
    assert drv.compute(0.0) == pytest.approx(0.0)
    val = drv.compute(0.5)
    assert 0.0 <= val < 2.0 * np.pi


def test_informational_batch():
    drv = InformationalDriver(cadence_hz=2.0)
    t = np.array([0.0, 0.25, 0.5])
    out = drv.compute_batch(t)
    assert out.shape == (3,)
    assert np.all(out >= 0.0) and np.all(out < 2.0 * np.pi)


def test_informational_bad_cadence():
    with pytest.raises(ValueError, match="positive"):
        InformationalDriver(cadence_hz=0.0)


def test_symbolic_sequence():
    drv = SymbolicDriver(sequence=[1.0, 2.0, 3.0])
    assert drv.compute(0) == 1.0
    assert drv.compute(1) == 2.0
    assert drv.compute(3) == 1.0  # wraps


def test_symbolic_batch():
    drv = SymbolicDriver(sequence=[10.0, 20.0])
    steps = np.array([0, 1, 2, 3])
    out = drv.compute_batch(steps)
    np.testing.assert_array_equal(out, [10.0, 20.0, 10.0, 20.0])


def test_symbolic_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        SymbolicDriver(sequence=[])
