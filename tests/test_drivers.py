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


# ---------------------------------------------------------------------------
# PhysicalDriver: sinusoidal external drive Ψ_P(t) = A·sin(2πft)
# ---------------------------------------------------------------------------


class TestPhysicalDriver:
    """Verify sinusoidal driver against analytical sin function."""

    def test_sin_zero_at_origin(self):
        drv = PhysicalDriver(frequency=1.0, amplitude=2.0)
        assert drv.compute(0.0) == pytest.approx(0.0, abs=1e-12)

    def test_sin_peak_at_quarter_period(self):
        """sin(2π·1·0.25) = sin(π/2) = 1 → amplitude·1 = 2."""
        drv = PhysicalDriver(frequency=1.0, amplitude=2.0)
        assert drv.compute(0.25) == pytest.approx(2.0, abs=1e-6)

    def test_sin_trough_at_three_quarter_period(self):
        drv = PhysicalDriver(frequency=1.0, amplitude=2.0)
        assert drv.compute(0.75) == pytest.approx(-2.0, abs=1e-6)

    def test_periodicity(self):
        """Psi(t + 1/f) must equal Psi(t) for any t."""
        drv = PhysicalDriver(frequency=3.0, amplitude=1.5)
        t = 0.137
        period = 1.0 / 3.0
        assert drv.compute(t) == pytest.approx(drv.compute(t + period), abs=1e-12)
        assert drv.compute(t) == pytest.approx(drv.compute(t + 5 * period), abs=1e-10)

    def test_batch_matches_scalar(self):
        """compute_batch must produce identical results to per-element compute."""
        drv = PhysicalDriver(frequency=2.5, amplitude=0.8)
        t = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        batch = drv.compute_batch(t)
        scalar = np.array([drv.compute(ti) for ti in t])
        np.testing.assert_allclose(batch, scalar, atol=1e-12)

    def test_amplitude_scales_output(self):
        """Doubling amplitude must double the output at the same time."""
        t = 0.13
        drv1 = PhysicalDriver(frequency=1.0, amplitude=1.0)
        drv2 = PhysicalDriver(frequency=1.0, amplitude=3.0)
        assert drv2.compute(t) == pytest.approx(3.0 * drv1.compute(t), abs=1e-12)

    def test_frequency_scales_phase(self):
        """Doubling frequency at same time must advance the phase 2×."""
        drv1 = PhysicalDriver(frequency=1.0)
        drv2 = PhysicalDriver(frequency=2.0)
        # f=1 at t=0.125: sin(π/4) ≈ 0.707
        # f=2 at t=0.125: sin(π/2) = 1.0
        assert abs(drv2.compute(0.125)) > abs(drv1.compute(0.125))

    def test_negative_frequency_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            PhysicalDriver(frequency=-1.0)

    def test_zero_frequency_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            PhysicalDriver(frequency=0.0)

    def test_output_bounded_by_amplitude(self):
        """Output must always be in [-amplitude, +amplitude]."""
        drv = PhysicalDriver(frequency=7.3, amplitude=2.5)
        t = np.linspace(0, 10, 5000)
        out = drv.compute_batch(t)
        assert np.all(out >= -2.5 - 1e-12) and np.all(out <= 2.5 + 1e-12)


# ---------------------------------------------------------------------------
# InformationalDriver: phase ramp Ψ_I(t) = (2πf·t) mod 2π
# ---------------------------------------------------------------------------


class TestInformationalDriver:
    """Verify the informational ramp driver produces monotonically
    increasing phase wrapped to [0, 2π)."""

    def test_zero_at_origin(self):
        drv = InformationalDriver(cadence_hz=1.0)
        assert drv.compute(0.0) == pytest.approx(0.0)

    def test_output_in_phase_range(self):
        drv = InformationalDriver(cadence_hz=1.0)
        for t in [0.0, 0.3, 0.5, 0.99, 1.5, 10.0]:
            val = drv.compute(t)
            assert 0.0 <= val < 2.0 * np.pi, (
                f"t={t}: output {val:.4f} outside [0, 2π)"
            )

    def test_batch_output_range(self):
        drv = InformationalDriver(cadence_hz=2.0)
        t = np.linspace(0, 5, 200)
        out = drv.compute_batch(t)
        assert out.shape == (200,)
        assert np.all(out >= 0.0) and np.all(out < 2.0 * np.pi)

    def test_higher_cadence_faster_ramp(self):
        """Higher cadence should complete more full cycles in same time."""
        drv_slow = InformationalDriver(cadence_hz=1.0)
        drv_fast = InformationalDriver(cadence_hz=5.0)
        t = np.linspace(0, 1, 1000)
        slow = drv_slow.compute_batch(t)
        fast = drv_fast.compute_batch(t)
        # Count zero-crossings (wraps) — fast should have more
        slow_wraps = np.sum(np.abs(np.diff(slow)) > np.pi)
        fast_wraps = np.sum(np.abs(np.diff(fast)) > np.pi)
        assert fast_wraps > slow_wraps

    def test_zero_cadence_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            InformationalDriver(cadence_hz=0.0)

    def test_negative_cadence_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            InformationalDriver(cadence_hz=-1.0)


# ---------------------------------------------------------------------------
# SymbolicDriver: periodic sequence
# ---------------------------------------------------------------------------


class TestSymbolicDriver:
    """Verify the symbolic driver cycles through its sequence and wraps."""

    def test_sequence_values(self):
        drv = SymbolicDriver(sequence=[1.0, 2.0, 3.0])
        assert drv.compute(0) == 1.0
        assert drv.compute(1) == 2.0
        assert drv.compute(2) == 3.0

    def test_wrapping(self):
        """Step beyond sequence length must wrap (modulo)."""
        drv = SymbolicDriver(sequence=[1.0, 2.0, 3.0])
        assert drv.compute(3) == 1.0
        assert drv.compute(5) == 3.0
        assert drv.compute(100) == drv.compute(100 % 3)

    def test_batch_matches_scalar(self):
        drv = SymbolicDriver(sequence=[10.0, 20.0, 30.0])
        steps = np.array([0, 1, 2, 3, 4, 5])
        batch = drv.compute_batch(steps)
        scalar = np.array([drv.compute(s) for s in steps])
        np.testing.assert_array_equal(batch, scalar)

    def test_single_element_sequence(self):
        """Single-element sequence returns the same value always."""
        drv = SymbolicDriver(sequence=[42.0])
        assert drv.compute(0) == 42.0
        assert drv.compute(999) == 42.0

    def test_empty_sequence_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            SymbolicDriver(sequence=[])

    def test_batch_periodicity(self):
        """Batch output must be periodic with period = len(sequence)."""
        seq = [1.0, 2.0, 3.0, 4.0]
        drv = SymbolicDriver(sequence=seq)
        steps = np.arange(20)
        out = drv.compute_batch(steps)
        for i in range(len(seq)):
            np.testing.assert_array_equal(
                out[i::len(seq)], seq[i],
                err_msg=f"Periodicity broken at offset {i}",
            )


class TestPipelineWiring:
    """Pipeline wiring: proves this module is not decorative."""

    def test_wires_into_pipeline(self):
        import numpy as np

        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        n = 8
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        for _ in range(100):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0
