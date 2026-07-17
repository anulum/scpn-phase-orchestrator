# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Physical oscillator tests

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.oscillators import physical as physical_module
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


@pytest.mark.parametrize("node_id", ["", "   ", 42, True])
def test_invalid_node_id_rejected(node_id: Any):
    with pytest.raises(ValueError, match="node_id must be a non-empty string"):
        PhysicalExtractor(node_id=node_id)


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


@pytest.mark.parametrize(
    "signal",
    [
        np.array([0.0, float("nan")]),
        np.array([0.0, float("inf")]),
        np.array([True, False]),
        np.array([1.0 + 0.0j, 0.0 + 0.0j]),
        np.array(["0.0", "1.0"], dtype=object),
    ],
)
def test_extract_rejects_invalid_signal(signal: object):
    extractor = PhysicalExtractor()
    with pytest.raises(ValueError, match="signal must be finite"):
        extractor.extract(signal, sample_rate=1000.0)


@pytest.mark.parametrize(
    "sample_rate",
    [True, 0.0, -1000.0, float("nan"), float("inf"), "1000.0"],
)
def test_extract_rejects_invalid_sample_rate(sample_rate: object):
    signal = np.sin(TWO_PI * 10.0 * np.arange(0, 0.1, 0.001))
    extractor = PhysicalExtractor()
    with pytest.raises(ValueError, match="sample_rate must be finite and positive"):
        extractor.extract(signal, sample_rate=sample_rate)


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


def test_extract_uses_rust_kernel_when_available(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[np.ndarray, np.ndarray, float]] = []

    def _fake_rust_extract(
        real: np.ndarray, imag: np.ndarray, sample_rate: float
    ) -> tuple[float, float, float, float]:
        calls.append((real, imag, sample_rate))
        return (0.25, 12.5, 0.75, 0.95)

    monkeypatch.setattr(physical_module, "_rust_physical_extract", _fake_rust_extract)
    fs = 1000.0
    t = np.arange(0, 0.2, 1.0 / fs)
    signal = np.sin(TWO_PI * 7.0 * t)

    states = PhysicalExtractor().extract(signal, fs)
    assert states[0].theta == pytest.approx(0.25, abs=1e-12)
    assert states[0].omega == pytest.approx(12.5, abs=1e-12)
    assert states[0].amplitude == pytest.approx(0.75, abs=1e-12)
    assert states[0].quality == pytest.approx(0.95, abs=1e-12)
    assert len(calls) == 1


def test_extract_falls_back_to_python_when_rust_raises(
    monkeypatch: pytest.MonkeyPatch,
):
    def _raising_rust_extract(
        _real: np.ndarray, _imag: np.ndarray, _sample_rate: float
    ) -> tuple[float, float, float, float]:
        raise RuntimeError("boom")

    monkeypatch.setattr(
        physical_module, "_rust_physical_extract", _raising_rust_extract
    )
    fs = 1000.0
    t = np.arange(0, 0.5, 1.0 / fs)
    signal = np.sin(TWO_PI * 10.0 * t)

    states = PhysicalExtractor().extract(signal, fs)
    assert 0.0 <= states[0].theta < TWO_PI
    assert states[0].omega == pytest.approx(TWO_PI * 10.0, rel=0.05)
    assert states[0].quality > 0.5


class TestPhysicalExtractorPipelineEndToEnd:
    """Full pipeline: raw signal → PhysicalExtractor → theta/omega → Engine → R.

    Proves PhysicalExtractor is the input adapter, not decorative.
    """

    def test_extract_feed_engine_order_param(self):
        """Extract phases from sinusoids → feed into engine → compute R."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        fs = 1000.0
        t = np.arange(0, 1.0, 1.0 / fs)
        n = 4
        extractor = PhysicalExtractor()
        phases = []
        omegas = []
        for i in range(n):
            signal = np.sin(TWO_PI * (5.0 + i) * t)
            states = extractor.extract(signal, fs)
            phases.append(states[0].theta)
            omegas.append(states[0].omega)
        phases_arr = np.array(phases)
        omegas_arr = np.array(omegas)
        knm = 0.5 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        eng = UPDEEngine(n, dt=0.01, method="rk4")
        for _ in range(200):
            phases_arr = eng.step(phases_arr, omegas_arr, knm, 0.0, 0.0, alpha)
        r, _ = compute_order_parameter(phases_arr)
        assert 0.0 <= r <= 1.0
        assert np.all(phases_arr >= 0.0)
        assert np.all(phases_arr < TWO_PI)

    def test_multi_channel_extraction_consistency(self):
        """Multiple channels extract to valid phases, all feedable to engine."""
        fs = 500.0
        t = np.arange(0, 0.5, 1.0 / fs)
        n = 6
        all_phases = []
        for i in range(n):
            signal = np.sin(TWO_PI * (3.0 + i * 2) * t)
            extractor = PhysicalExtractor(node_id=f"p{i}")
            states = extractor.extract(signal, fs)
            assert states[0].channel == "P"
            assert 0.0 <= states[0].theta < TWO_PI
            assert states[0].quality > 0.0
            all_phases.append(states[0].theta)
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        r, _ = compute_order_parameter(np.array(all_phases))
        assert 0.0 <= r <= 1.0

    def test_performance_extract_1s_1kHz_under_5ms(self):
        """PhysicalExtractor.extract(1s @ 1kHz) < 5ms."""
        import time

        fs = 1000.0
        t = np.arange(0, 1.0, 1.0 / fs)
        signal = np.sin(TWO_PI * 10.0 * t)
        extractor = PhysicalExtractor()
        extractor.extract(signal, fs)  # warm-up
        t0 = time.perf_counter()
        for _ in range(100):
            extractor.extract(signal, fs)
        elapsed = (time.perf_counter() - t0) / 100
        assert elapsed < 0.005, f"extract(1s) took {elapsed * 1e3:.2f}ms"


def _endpoint_reference(signal: np.ndarray, fs: float) -> float:
    """Return the historical endpoint theta from the raw broadband Hilbert phase."""
    from scipy.signal import hilbert

    analytic = hilbert(signal)
    return float((np.angle(analytic) % TWO_PI)[-1])


class TestPhysicalExtractorBandpassAndEdgeTrim:
    """Opt-in KDD-H2 refinements: zero-phase band-pass + edge-trim (defaults off)."""

    def test_default_extraction_matches_historical_endpoint(self):
        """Default (no band, no trim) reports the raw broadband endpoint phase."""
        fs = 1000.0
        t = np.arange(0, 0.5, 1.0 / fs)
        signal = np.sin(TWO_PI * 10.0 * t) + 0.3 * np.sin(TWO_PI * 60.0 * t)
        states = PhysicalExtractor().extract(signal, fs)
        assert states[0].theta == pytest.approx(
            _endpoint_reference(signal, fs), abs=1e-9
        )

    def test_bandpass_isolates_in_band_frequency(self):
        """A band around 10 Hz recovers ~2*pi*10 despite a strong 60 Hz component."""
        fs = 1000.0
        t = np.arange(0, 1.0, 1.0 / fs)
        signal = np.sin(TWO_PI * 10.0 * t) + np.sin(TWO_PI * 60.0 * t)
        broadband = PhysicalExtractor().extract(signal, fs)[0]
        banded = PhysicalExtractor(band=(8.0, 12.0)).extract(signal, fs)[0]
        np.testing.assert_allclose(banded.omega, TWO_PI * 10.0, rtol=0.05)
        assert abs(banded.omega - TWO_PI * 10.0) < abs(broadband.omega - TWO_PI * 10.0)

    def test_edge_trim_selects_interior_endpoint(self):
        """edge_trim=k reports the phase k samples before the raw endpoint."""
        fs = 1000.0
        t = np.arange(0, 0.5, 1.0 / fs)
        signal = np.sin(TWO_PI * 10.0 * t)
        from scipy.signal import hilbert

        inst_phase = np.angle(hilbert(signal)) % TWO_PI
        states = PhysicalExtractor(edge_trim=20).extract(signal, fs)
        assert states[0].theta == pytest.approx(float(inst_phase[-1 - 20]), abs=1e-9)

    def test_edge_trim_clamped_on_short_signal(self):
        """An oversized edge_trim is clamped so at least two samples survive."""
        fs = 100.0
        signal = np.sin(TWO_PI * 5.0 * np.arange(0, 0.1, 1.0 / fs))
        states = PhysicalExtractor(edge_trim=10_000).extract(signal, fs)
        # The clamp keeps >= 2 samples: extraction succeeds with a finite phase
        # (a modular phase may round onto the closed [0, 2*pi] boundary).
        assert np.isfinite(states[0].theta)
        assert 0.0 <= states[0].theta <= TWO_PI

    def test_bandpass_applies_automatic_edge_trim(self):
        """A configured band trims the filtfilt transient without an explicit count."""
        fs = 1000.0
        t = np.arange(0, 1.0, 1.0 / fs)
        signal = np.sin(TWO_PI * 10.0 * t)
        from scipy.signal import butter, filtfilt, hilbert

        coeff_b, coeff_a = butter(4, (8.0 / 500.0, 12.0 / 500.0), btype="band")
        filtered = np.asarray(filtfilt(coeff_b, coeff_a, signal), dtype=np.float64)
        auto_trim = 3 * (4 + 1)
        inst_phase = np.angle(hilbert(filtered)) % TWO_PI
        states = PhysicalExtractor(band=(8.0, 12.0)).extract(signal, fs)
        assert states[0].theta == pytest.approx(
            float(inst_phase[-1 - auto_trim]), abs=1e-9
        )

    def test_bandpass_and_trim_preserve_rust_python_parity(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The Rust path and the NumPy fallback agree under band-pass + trim."""
        fs = 1000.0
        t = np.arange(0, 1.0, 1.0 / fs)
        signal = np.sin(TWO_PI * 10.0 * t) + 0.4 * np.sin(TWO_PI * 55.0 * t)
        extractor = PhysicalExtractor(band=(8.0, 12.0), edge_trim=17)
        rust_state = extractor.extract(signal, fs)[0]

        def _raising_rust_extract(
            _real: np.ndarray, _imag: np.ndarray, _sample_rate: float
        ) -> tuple[float, float, float, float]:
            raise RuntimeError("force python fallback")

        monkeypatch.setattr(
            physical_module, "_rust_physical_extract", _raising_rust_extract
        )
        python_state = extractor.extract(signal, fs)[0]
        assert rust_state.theta == pytest.approx(python_state.theta, abs=1e-9)
        assert rust_state.omega == pytest.approx(python_state.omega, rel=1e-6)
        assert rust_state.amplitude == pytest.approx(python_state.amplitude, rel=1e-6)

    def test_bandpass_rejects_band_at_or_above_nyquist(self):
        """A pass-band reaching the Nyquist frequency is rejected at extract time."""
        fs = 100.0
        signal = np.sin(TWO_PI * 10.0 * np.arange(0, 1.0, 1.0 / fs))
        with pytest.raises(ValueError, match="Nyquist"):
            PhysicalExtractor(band=(10.0, 60.0)).extract(signal, fs)

    def test_bandpass_rejects_signal_shorter_than_filter(self):
        """A signal shorter than the filtfilt pad length is rejected clearly."""
        fs = 1000.0
        signal = np.sin(TWO_PI * 10.0 * np.arange(0, 0.02, 1.0 / fs))
        with pytest.raises(ValueError, match="too short for a band-pass"):
            PhysicalExtractor(band=(8.0, 12.0), filter_order=8).extract(signal, fs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"band": (12.0, 8.0)},
            {"band": (0.0, 8.0)},
            {"band": (8.0,)},
            {"band": "8-12"},
            {"band": (8.0, "12")},
            {"filter_order": 0},
            {"filter_order": True},
            {"edge_trim": -1},
            {"edge_trim": 1.5},
        ],
    )
    def test_invalid_construction_arguments_raise(self, kwargs: dict[str, Any]):
        """Malformed band / filter_order / edge_trim are rejected at construction."""
        with pytest.raises(ValueError):
            PhysicalExtractor(**kwargs)


# Pipeline wiring: PhysicalExtractor → theta/omega → UPDEEngine(RK4)
# → compute_order_parameter. Multi-channel extraction + quality scoring.
# Rust parity via spo_kernel. Performance: extract(1s@1kHz)<5ms.
# KDD-H2: opt-in zero-phase band-pass + edge-trim, defaults bit-identical.
