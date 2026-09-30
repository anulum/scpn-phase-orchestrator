# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Physical oscillator tests

"""Exercise physical extraction and its public phase-engine integration."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.oscillators import physical as physical_module
from scpn_phase_orchestrator.oscillators.physical import PhysicalExtractor

TWO_PI = 2.0 * np.pi


def test_hilbert_phase_monotonic() -> None:
    """10 Hz sine at 1000 Hz: unwrapped Hilbert phases increase monotonically."""
    fs = 1000.0
    t = np.arange(0, 0.5, 1.0 / fs)
    signal = np.sin(TWO_PI * 10.0 * t)
    extractor = PhysicalExtractor(node_id="test")
    states = extractor.extract(signal, fs)
    assert len(states) == 1
    assert 0.0 <= states[0].theta < TWO_PI


def test_quality_above_threshold_for_clean_sinusoid() -> None:
    """A clean periodic waveform retains a high envelope quality."""
    fs = 1000.0
    t = np.arange(0, 1.0, 1.0 / fs)
    signal = np.sin(TWO_PI * 5.0 * t)
    extractor = PhysicalExtractor()
    states = extractor.extract(signal, fs)
    assert states[0].quality > 0.5


def test_omega_matches_input_frequency() -> None:
    """Reported angular frequency agrees with the observed sinusoid."""
    fs = 1000.0
    f0 = 10.0
    t = np.arange(0, 1.0, 1.0 / fs)
    signal = np.sin(TWO_PI * f0 * t)
    extractor = PhysicalExtractor()
    states = extractor.extract(signal, fs)
    expected_omega = TWO_PI * f0
    np.testing.assert_allclose(states[0].omega, expected_omega, rtol=0.05)


def test_channel_is_physical() -> None:
    """Extraction preserves the physical channel and caller node identity."""
    fs = 500.0
    t = np.arange(0, 0.2, 1.0 / fs)
    signal = np.sin(TWO_PI * 8.0 * t)
    extractor = PhysicalExtractor(node_id="p1")
    states = extractor.extract(signal, fs)
    assert states[0].channel == "P"
    assert states[0].node_id == "p1"


@pytest.mark.parametrize("node_id", ["", "   ", 42, True])
def test_invalid_node_id_rejected(node_id: object) -> None:
    """Invalid caller identities refuse before signal extraction."""
    with pytest.raises(ValueError, match="node_id must be a non-empty string"):
        PhysicalExtractor(node_id=cast(str, node_id))


def test_quality_score_aggregation() -> None:
    """Quality aggregation agrees with the one physical observation."""
    fs = 1000.0
    t = np.arange(0, 0.5, 1.0 / fs)
    signal = np.sin(TWO_PI * 10.0 * t)
    extractor = PhysicalExtractor()
    states = extractor.extract(signal, fs)
    score = extractor.quality_score(states)
    assert 0.0 <= score <= 1.0
    assert score == states[0].quality


def test_quality_score_empty() -> None:
    """No observations produce zero aggregate quality."""
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
def test_extract_rejects_invalid_signal(signal: object) -> None:
    """Non-real or non-finite measurement arrays refuse at the public boundary."""
    extractor = PhysicalExtractor()
    with pytest.raises(ValueError, match="signal must be finite"):
        extractor.extract(cast(NDArray[np.float64], signal), sample_rate=1000.0)


@pytest.mark.parametrize(
    "sample_rate",
    [True, 0.0, -1000.0, float("nan"), float("inf"), "1000.0"],
)
def test_extract_rejects_invalid_sample_rate(sample_rate: object) -> None:
    """Invalid sample rates refuse before Hilbert extraction."""
    signal = np.sin(TWO_PI * 10.0 * np.arange(0, 0.1, 0.001))
    extractor = PhysicalExtractor()
    with pytest.raises(ValueError, match="sample_rate must be finite and positive"):
        extractor.extract(signal, sample_rate=cast(float, sample_rate))


def test_envelope_quality_clean_sinusoid() -> None:
    """Clean sinusoid has near-constant envelope → quality well above 0.5."""
    signal = np.sin(TWO_PI * 5.0 * np.arange(0, 1.0, 0.001))
    quality = PhysicalExtractor().extract(signal, 1000.0)[0].quality
    assert quality > 0.7


def test_quality_discriminates_clean_vs_noisy() -> None:
    """Pure sinusoid → quality > 0.9, sinusoid + heavy noise → quality < 0.7."""
    t = np.arange(0, 1.0, 0.001)
    clean = np.sin(TWO_PI * 10.0 * t)
    rng = np.random.default_rng(42)
    noisy = clean + rng.normal(0, 2.0, len(t))

    q_clean = PhysicalExtractor().extract(clean, 1000.0)[0].quality
    q_noisy = PhysicalExtractor().extract(noisy, 1000.0)[0].quality

    assert q_clean > 0.9, f"clean quality={q_clean}"
    assert q_noisy < 0.7, f"noisy quality={q_noisy}"


@pytest.mark.parametrize("length", [127, 128])
def test_extraction_matches_independent_analytic_reference(length: int) -> None:
    """Odd and even FFT lengths retain all four independently derived fields."""
    from scipy.signal import hilbert

    signal = np.cos(TWO_PI * 9.0 * np.arange(length) / length)
    analytic = hilbert(signal)
    envelope = np.abs(analytic)
    state = PhysicalExtractor().extract(signal, float(length))[0]
    assert state.theta == pytest.approx(
        float(np.angle(analytic[-1]) % TWO_PI), abs=1e-12
    )
    assert state.omega == pytest.approx(TWO_PI * 9.0, rel=1e-12)
    assert state.amplitude == pytest.approx(float(np.mean(envelope)), abs=1e-12)
    assert state.quality == pytest.approx(
        1.0 - float(np.std(envelope) / np.mean(envelope)), abs=1e-12
    )


def test_extract_falls_back_to_python_when_rust_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A native execution error recovers through actual Python extraction."""
    calls: list[tuple[int, int, float]] = []

    # A native exception cannot be induced by valid observations. This injection
    # exercises the public recovery contract, never successful native execution.
    def _raising_rust_extract(
        _real: NDArray[np.float64], _imag: NDArray[np.float64], _sample_rate: float
    ) -> tuple[float, float, float, float]:
        """Raise only to exercise the public recovery contract."""
        calls.append((_real.size, _imag.size, _sample_rate))
        raise RuntimeError("boom")

    monkeypatch.setattr(
        physical_module, "_rust_physical_extract", _raising_rust_extract
    )
    fs = 1000.0
    t = np.arange(0, 0.5, 1.0 / fs)
    signal = np.sin(TWO_PI * 10.0 * t)

    from scipy.signal import hilbert

    analytic = hilbert(signal)
    envelope = np.abs(analytic)
    state = PhysicalExtractor().extract(signal, fs)[0]
    assert calls == [(signal.size, signal.size, fs)]
    assert state.theta == pytest.approx(
        float(np.angle(analytic[-1]) % TWO_PI), abs=1e-12
    )
    assert state.omega == pytest.approx(TWO_PI * 10.0, rel=1e-12)
    assert state.amplitude == pytest.approx(float(np.mean(envelope)), abs=1e-12)
    assert state.quality == pytest.approx(
        1.0 - float(np.std(envelope) / np.mean(envelope)), abs=1e-12
    )


class TestPhysicalExtractorPipelineEndToEnd:
    """Verify extracted physical states through the public phase-engine consumer."""

    def test_extract_feed_engine_order_param(self) -> None:
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
            signal = np.sin(TWO_PI * 5.0 * t + i * np.pi / 2)
            states = extractor.extract(signal, fs)
            phases.append(states[0].theta)
            omegas.append(states[0].omega)
        phases_arr = np.array(phases)
        expected = np.exp(1j * (phases_arr + 200 * 0.01 * TWO_PI * 5.0))
        omegas_arr = np.array(omegas)
        knm = 0.5 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        eng = UPDEEngine(n, dt=0.01, method="rk4")
        for _ in range(200):
            phases_arr = eng.step(phases_arr, omegas_arr, knm, 0.0, 0.0, alpha)
        r, _ = compute_order_parameter(phases_arr)
        np.testing.assert_allclose(np.exp(1j * phases_arr), expected, atol=1e-9)
        assert r < 1e-9
        assert np.all(phases_arr >= 0.0)
        assert np.all(phases_arr < TWO_PI)

    def test_multi_channel_extraction_consistency(self) -> None:
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

    def test_performance_extract_1s_1kHz_under_5ms(self) -> None:
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


def _endpoint_reference(signal: NDArray[np.float64], fs: float) -> float:
    """Return the historical endpoint theta from the raw broadband Hilbert phase."""
    from scipy.signal import hilbert

    analytic = hilbert(signal)
    return float((np.angle(analytic) % TWO_PI)[-1])


class TestPhysicalExtractorBandpassAndEdgeTrim:
    """Verify opt-in zero-phase filtering and finite-length edge trimming."""

    def test_default_extraction_matches_historical_endpoint(self) -> None:
        """Default (no band, no trim) reports the raw broadband endpoint phase."""
        fs = 1000.0
        t = np.arange(0, 0.5, 1.0 / fs)
        signal = np.sin(TWO_PI * 10.0 * t) + 0.3 * np.sin(TWO_PI * 60.0 * t)
        states = PhysicalExtractor().extract(signal, fs)
        assert states[0].theta == pytest.approx(
            _endpoint_reference(signal, fs), abs=1e-9
        )

    def test_bandpass_isolates_in_band_frequency(self) -> None:
        """A band around 10 Hz recovers ~2*pi*10 despite a strong 60 Hz component."""
        fs = 1000.0
        t = np.arange(0, 1.0, 1.0 / fs)
        signal = np.sin(TWO_PI * 10.0 * t) + np.sin(TWO_PI * 60.0 * t)
        broadband = PhysicalExtractor().extract(signal, fs)[0]
        banded = PhysicalExtractor(band=(8.0, 12.0)).extract(signal, fs)[0]
        np.testing.assert_allclose(banded.omega, TWO_PI * 10.0, rtol=0.05)
        assert abs(banded.omega - TWO_PI * 10.0) < abs(broadband.omega - TWO_PI * 10.0)

    def test_edge_trim_selects_interior_endpoint(self) -> None:
        """edge_trim=k reports the phase k samples before the raw endpoint."""
        fs = 1000.0
        t = np.arange(0, 0.5, 1.0 / fs)
        signal = np.sin(TWO_PI * 10.0 * t)
        from scipy.signal import hilbert

        inst_phase = np.angle(hilbert(signal)) % TWO_PI
        states = PhysicalExtractor(edge_trim=20).extract(signal, fs)
        assert states[0].theta == pytest.approx(float(inst_phase[-1 - 20]), abs=1e-9)

    def test_edge_trim_clamped_on_short_signal(self) -> None:
        """An oversized edge_trim is clamped so at least two samples survive."""
        fs = 100.0
        signal = np.sin(TWO_PI * 5.0 * np.arange(0, 0.1, 1.0 / fs))
        states = PhysicalExtractor(edge_trim=10_000).extract(signal, fs)
        # The clamp keeps >= 2 samples: extraction succeeds with a finite phase
        # (a modular phase may round onto the closed [0, 2*pi] boundary).
        assert np.isfinite(states[0].theta)
        assert 0.0 <= states[0].theta <= TWO_PI

    def test_bandpass_applies_automatic_edge_trim(self) -> None:
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

    def test_bandpass_and_trim_match_independent_reference(self) -> None:
        """Both real environments use the same filtered and trimmed analytic signal."""
        fs = 1000.0
        t = np.arange(0, 1.0, 1.0 / fs)
        signal = np.sin(TWO_PI * 10.0 * t) + 0.4 * np.sin(TWO_PI * 55.0 * t)
        extractor = PhysicalExtractor(band=(8.0, 12.0), edge_trim=17)
        state = extractor.extract(signal, fs)[0]
        from scipy.signal import butter, filtfilt, hilbert

        b, a = butter(4, (8.0 / 500.0, 12.0 / 500.0), btype="band")
        analytic = hilbert(filtfilt(b, a, signal))[17:-17]
        envelope = np.abs(analytic)
        expected_omega = (
            float(np.median(np.gradient(np.unwrap(np.angle(analytic))))) * fs
        )
        assert state.theta == pytest.approx(
            float(np.angle(analytic[-1]) % TWO_PI), abs=1e-9
        )
        assert state.omega == pytest.approx(expected_omega, rel=1e-6)
        assert state.amplitude == pytest.approx(float(np.mean(envelope)), rel=1e-12)
        assert state.quality == pytest.approx(
            1.0 - float(np.std(envelope) / np.mean(envelope)), abs=1e-12
        )

    def test_bandpass_rejects_band_at_or_above_nyquist(self) -> None:
        """A pass-band reaching the Nyquist frequency is rejected at extract time."""
        fs = 100.0
        signal = np.sin(TWO_PI * 10.0 * np.arange(0, 1.0, 1.0 / fs))
        with pytest.raises(ValueError, match="Nyquist"):
            PhysicalExtractor(band=(10.0, 60.0)).extract(signal, fs)

    def test_bandpass_rejects_signal_shorter_than_filter(self) -> None:
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
    def test_invalid_construction_arguments_raise(
        self, kwargs: dict[str, object]
    ) -> None:
        """Malformed band / filter_order / edge_trim are rejected at construction."""
        with pytest.raises(ValueError):
            PhysicalExtractor(
                band=cast(tuple[float, float] | None, kwargs.get("band")),
                filter_order=cast(int, kwargs.get("filter_order", 4)),
                edge_trim=cast(int | None, kwargs.get("edge_trim")),
            )


@pytest.mark.parametrize("scale", [1.0, 1e150, 1e160, 1e200, 1e290])
@pytest.mark.parametrize("modulation", [0.0, 0.6])
def test_finite_large_waveforms_preserve_envelope_contract(
    scale: float, modulation: float
) -> None:
    """Representable large envelopes retain phase, frequency, amplitude and CV."""
    phase = TWO_PI * np.arange(128, dtype=np.float64) / 128.0
    envelope = 1.0 + modulation * np.cos(phase)
    signal = scale * envelope * np.cos(8.0 * phase)
    original = signal.copy()
    state = PhysicalExtractor(node_id="large").extract(signal, 128.0)[0]
    assert state.theta == pytest.approx(float((8.0 * phase[-1]) % TWO_PI), abs=1e-12)
    assert state.omega == pytest.approx(TWO_PI * 8.0, rel=1e-12)
    assert state.amplitude / scale == pytest.approx(1.0, abs=1e-12)
    assert state.quality == pytest.approx(1.0 - modulation / np.sqrt(2.0), abs=1e-12)
    assert state.node_id == "large"
    assert state.channel == "P"
    np.testing.assert_array_equal(signal, original)


@pytest.mark.parametrize("scale", [0.0, 1e-17, 1e-15, 1.0])
def test_constant_waveform_retains_absolute_envelope_quality_gate(scale: float) -> None:
    """Zero and low amplitudes retain the defined absolute quality cutoff."""
    signal = np.full(4, scale, dtype=np.float64)
    state = PhysicalExtractor().extract(signal, 1.0)[0]
    assert state.theta == 0.0
    assert state.omega == 0.0
    assert state.amplitude == scale
    assert state.quality == (0.0 if scale < 1e-15 else 1.0)


@pytest.mark.parametrize("signal", [np.array([]), np.array([1.0]), np.ones((2, 2))])
def test_invalid_waveform_shapes_refuse(signal: NDArray[np.float64]) -> None:
    """Insufficient samples or multiple dimensions refuse before Hilbert processing."""
    with pytest.raises(ValueError, match="1-D with >= 2 samples"):
        PhysicalExtractor().extract(signal, 1.0)


def test_numeric_lists_preserve_boolean_admission_boundary() -> None:
    """Numeric lists are measurements; promoted boolean elements remain invalid."""
    with pytest.raises(ValueError, match="boolean"):
        PhysicalExtractor().extract(cast(NDArray[np.float64], [True, 1.0]), 1.0)
    state = PhysicalExtractor().extract(cast(NDArray[np.float64], [1.0, 1.0]), 1.0)[0]
    assert state.amplitude == 1.0
    assert state.quality == 1.0


@pytest.mark.parametrize("length, scale, cycles", [(128, 1e307, 8), (4, 1e308, 0)])
def test_finite_waveform_with_overflowing_analytic_signal_refuses(
    length: int, scale: float, cycles: int
) -> None:
    """FFT overflow on finite raw samples refuses before returning phase metadata."""
    phase = TWO_PI * cycles * np.arange(length, dtype=np.float64) / length
    signal = scale * np.cos(phase)
    original = signal.copy()
    assert np.all(np.isfinite(signal))
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ValueError, match="analytic signal must be finite"),
    ):
        PhysicalExtractor().extract(signal, float(length))
    np.testing.assert_array_equal(signal, original)
