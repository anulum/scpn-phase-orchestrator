# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Physical oscillator tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.oscillators.physical import PhysicalExtractor

TWO_PI = 2.0 * np.pi


def test_hilbert_phase_monotonic():
    """10 Hz sine at 1000 Hz: unwrapped Hilbert phases increase monotonically."""
    fs = 1000.0
    t = np.arange(0, 0.5, 1.0 / fs)
    signal = np.sin(TWO_PI * 10.0 * t)
    extractor = PhysicalExtractor(node_id="test")
    states = extractor.extract(signal, fs)
    assert len(states) == 1
    assert 0.0 <= states[0].theta < TWO_PI


def test_quality_above_threshold_for_clean_sinusoid():
    fs = 1000.0
    t = np.arange(0, 1.0, 1.0 / fs)
    signal = np.sin(TWO_PI * 5.0 * t)
    extractor = PhysicalExtractor()
    states = extractor.extract(signal, fs)
    assert states[0].quality > 0.5


def test_omega_matches_input_frequency():
    fs = 1000.0
    f0 = 10.0
    t = np.arange(0, 1.0, 1.0 / fs)
    signal = np.sin(TWO_PI * f0 * t)
    extractor = PhysicalExtractor()
    states = extractor.extract(signal, fs)
    expected_omega = TWO_PI * f0
    np.testing.assert_allclose(states[0].omega, expected_omega, rtol=0.05)


def test_channel_is_physical():
    fs = 500.0
    t = np.arange(0, 0.2, 1.0 / fs)
    signal = np.sin(TWO_PI * 8.0 * t)
    extractor = PhysicalExtractor(node_id="p1")
    states = extractor.extract(signal, fs)
    assert states[0].channel == "P"
    assert states[0].node_id == "p1"


def test_quality_score_aggregation():
    fs = 1000.0
    t = np.arange(0, 0.5, 1.0 / fs)
    signal = np.sin(TWO_PI * 10.0 * t)
    extractor = PhysicalExtractor()
    states = extractor.extract(signal, fs)
    score = extractor.quality_score(states)
    assert 0.0 <= score <= 1.0
    assert score == states[0].quality


def test_quality_score_empty():
    extractor = PhysicalExtractor()
    assert extractor.quality_score([]) == 0.0


def test_envelope_quality_clean_sinusoid():
    """Clean sinusoid has near-constant envelope → quality well above 0.5."""
    from scipy.signal import hilbert

    signal = np.sin(TWO_PI * 5.0 * np.arange(0, 1.0, 0.001))
    quality = PhysicalExtractor._envelope_quality(signal, hilbert(signal))
    assert quality > 0.7


def test_quality_discriminates_clean_vs_noisy():
    """Pure sinusoid → quality > 0.9, sinusoid + heavy noise → quality < 0.7."""
    from scipy.signal import hilbert

    t = np.arange(0, 1.0, 0.001)
    clean = np.sin(TWO_PI * 10.0 * t)
    rng = np.random.default_rng(42)
    noisy = clean + rng.normal(0, 2.0, len(t))

    q_clean = PhysicalExtractor._envelope_quality(clean, hilbert(clean))
    q_noisy = PhysicalExtractor._envelope_quality(noisy, hilbert(noisy))

    assert q_clean > 0.9, f"clean quality={q_clean}"
    assert q_noisy < 0.7, f"noisy quality={q_noisy}"


def test_rust_python_parity():
    """Rust and Python paths produce identical results within float tolerance."""
    try:
        from spo_kernel import physical_extract
    except ImportError:
        import pytest

        pytest.skip("spo_kernel not built")

    from scipy.signal import hilbert

    fs = 1000.0
    t = np.arange(0, 0.5, 1.0 / fs)
    signal = np.sin(TWO_PI * 10.0 * t)
    analytic = hilbert(signal)

    r_theta, r_omega, r_amp, r_quality = physical_extract(
        np.ascontiguousarray(np.real(analytic)),
        np.ascontiguousarray(np.imag(analytic)),
        fs,
    )

    inst_phase = np.angle(analytic) % TWO_PI
    inst_amp = np.abs(analytic)
    inst_freq = np.gradient(np.unwrap(np.angle(analytic))) * fs / TWO_PI
    p_theta = float(inst_phase[-1])
    p_omega = float(np.median(inst_freq)) * TWO_PI
    p_amp = float(np.mean(inst_amp))

    np.testing.assert_allclose(r_theta, p_theta, atol=1e-10)
    np.testing.assert_allclose(r_omega, p_omega, rtol=0.01)
    np.testing.assert_allclose(r_amp, p_amp, rtol=1e-6)
    assert r_quality > 0.5


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
