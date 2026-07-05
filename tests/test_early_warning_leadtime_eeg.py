# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — real-EEG early-warning capstone logic tests

"""Tests for the scalp-EEG early-warning capstone adapter on synthetic data.

This capstone is the scalp-EEG *adapter* onto the domain-neutral harness (tested
in ``tests/test_early_warning_domain.py``): the only EEG-specific work is the
signal-processing pipeline that produces the neutral observable bundle, plus the
EDF ingestion and the end-to-end orchestration. Those read a citation-only
corpus, so every path is pinned here on **synthetic arrays** — never the
protected raw recordings. The observable pipeline (band-pass, analytic phase,
phase-consistent decimation), the :class:`EEGPhaseAdapter`, the EDF ingestion (on
a synthetic EDF written in the test), and the end-to-end ``main`` (over synthetic
EDFs) are each exercised directly, so the logic's correctness does not depend on
downloading the CHB-MIT database.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from bench.early_warning_leadtime_eeg import (
    BAND_HZ,
    DETECTORS,
    SAMPLING_RATE_HZ,
    STEP,
    WINDOW,
    EEGPhaseAdapter,
    analytic_phase,
    bandpass,
    decimate_analytic_phase,
    edf_start_datetime,
    eeg_observables,
    load_edf_channels,
    main,
)
from scpn_phase_orchestrator.monitor.early_warning_suite import (
    DomainObservableAdapter,
    SuiteObservables,
)

_TWO_PI = 2.0 * np.pi

# The EDF-ingestion path needs pyedflib, an optional native dependency declared as
# the ``eeg`` extra (``pip install -e .[eeg]``). The pipeline and adapter tests
# never touch it; only the EDF I/O and end-to-end orchestration tests do, so they
# are gated on its presence — the same pattern the suite uses for jax, juliacall,
# and the other optional backends.
_requires_pyedflib = pytest.mark.skipif(
    importlib.util.find_spec("pyedflib") is None,
    reason="pyedflib is an optional dependency (install the 'eeg' extra)",
)


# --------------------------------------------------------------------------- #
# Synthetic fixtures                                                           #
# --------------------------------------------------------------------------- #


def _raw_tone(
    *, n_channels: int, n_samples: int, fs: float, hz: float, seed: int
) -> np.ndarray:
    """Return a multichannel band-limited tone with per-channel phase offsets."""
    rng = np.random.default_rng(seed)
    times = np.arange(n_samples) / fs
    offsets = rng.uniform(-np.pi, np.pi, n_channels)
    raw = np.empty((n_channels, n_samples), dtype=np.float64)
    for channel in range(n_channels):
        raw[channel] = np.sin(_TWO_PI * hz * times + offsets[channel])
    return raw


# --------------------------------------------------------------------------- #
# bandpass / analytic_phase / decimate_analytic_phase                          #
# --------------------------------------------------------------------------- #


def test_bandpass_keeps_in_band_and_rejects_out_of_band() -> None:
    fs = 256.0
    n = 2048
    times = np.arange(n) / fs
    in_band = np.sin(_TWO_PI * 10.0 * times)
    out_band = np.sin(_TWO_PI * 60.0 * times)
    signal = np.vstack([in_band + out_band])
    filtered = bandpass(signal, sampling_rate_hz=fs)
    # The 10 Hz component survives; the 60 Hz component is strongly attenuated.
    residual_out = float(np.std(filtered[0] - in_band))
    assert residual_out < 0.2
    assert float(np.std(filtered[0])) > 0.5


def test_bandpass_promotes_a_single_channel() -> None:
    fs = 256.0
    tone = np.sin(_TWO_PI * 12.0 * np.arange(1024) / fs)
    filtered = bandpass(tone, sampling_rate_hz=fs)
    assert filtered.shape == (1, 1024)


def test_bandpass_rejects_a_band_above_nyquist() -> None:
    with pytest.raises(ValueError, match="Nyquist"):
        bandpass(np.zeros((2, 512)), sampling_rate_hz=64.0, band_hz=(4.0, 40.0))


def test_bandpass_rejects_a_low_edge_at_or_above_the_high_edge() -> None:
    with pytest.raises(ValueError, match="must be below high"):
        bandpass(np.zeros((2, 512)), sampling_rate_hz=256.0, band_hz=(30.0, 30.0))


def test_bandpass_rejects_a_malformed_band() -> None:
    with pytest.raises(ValueError, match="low, high"):
        bandpass(np.zeros((2, 512)), sampling_rate_hz=256.0, band_hz=(4.0, 20.0, 30.0))


def test_analytic_phase_advances_linearly_for_a_pure_tone() -> None:
    fs = 256.0
    hz = 10.0
    tone = np.cos(_TWO_PI * hz * np.arange(1024) / fs)
    phase = analytic_phase(tone)
    increments = np.diff(np.unwrap(phase[0]))
    expected = _TWO_PI * hz / fs
    # The interior increments track the tone's angular step (edges excluded).
    assert np.allclose(increments[50:-50], expected, atol=1.0e-3)


def test_decimate_reduces_length_and_preserves_a_constant_phase() -> None:
    phases = np.zeros((3, 2400), dtype=np.float64)
    decimated = decimate_analytic_phase(phases, factor=8)
    assert decimated.shape == (3, 300)
    assert np.allclose(decimated, 0.0, atol=1.0e-6)


def test_decimate_with_unit_factor_is_the_identity() -> None:
    phases = np.linspace(-np.pi, np.pi, 200).reshape(2, 100)
    decimated = decimate_analytic_phase(phases, factor=1)
    assert np.array_equal(decimated, phases)


def test_decimate_rejects_a_non_positive_factor() -> None:
    with pytest.raises(ValueError, match="factor"):
        decimate_analytic_phase(np.zeros((2, 80)), factor=0)


# --------------------------------------------------------------------------- #
# eeg_observables and the EEGPhaseAdapter                                       #
# --------------------------------------------------------------------------- #


def test_eeg_observables_shapes_rate_and_derived_fields() -> None:
    raw = _raw_tone(n_channels=4, n_samples=2400, fs=256.0, hz=10.0, seed=3)
    observables = eeg_observables(raw)
    assert isinstance(observables, SuiteObservables)
    assert observables.sampling_rate_hz == pytest.approx(32.0)
    assert observables.n_nodes == 4
    assert observables.n_samples == 300
    assert observables.phases.shape == (4, 300)
    assert observables.phase_field.shape == (4, 300)
    assert observables.order_parameter.shape == (300,)
    # The projection is exactly sin of the reconstructed phase.
    assert np.allclose(observables.phase_field, np.sin(observables.phases))
    # The order parameter is a coherence in [0, 1].
    assert np.all(observables.order_parameter >= 0.0)
    assert np.all(observables.order_parameter <= 1.0 + 1.0e-9)


def test_eeg_observables_rejects_a_single_channel() -> None:
    with pytest.raises(ValueError, match="at least two channels"):
        eeg_observables(np.zeros((1, 2400)))


def test_eeg_observables_rejects_a_complex_recording() -> None:
    with pytest.raises(ValueError, match="real-valued"):
        eeg_observables(np.ones((2, 2400), dtype=np.complex128))


def test_eeg_observables_rejects_a_non_finite_recording() -> None:
    raw = _raw_tone(n_channels=2, n_samples=2400, fs=256.0, hz=10.0, seed=4)
    raw[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        eeg_observables(raw)


def test_eeg_phase_adapter_satisfies_the_protocol_and_wraps_the_pipeline() -> None:
    adapter = EEGPhaseAdapter()
    assert isinstance(adapter, DomainObservableAdapter)
    assert adapter.domain == "scalp_eeg"
    raw = _raw_tone(n_channels=3, n_samples=2400, fs=256.0, hz=10.0, seed=5)
    observables = adapter.observables(raw)
    assert isinstance(observables, SuiteObservables)
    assert observables.n_nodes == 3
    assert observables.sampling_rate_hz == pytest.approx(32.0)
    # The adapter is exactly the pipeline function packaged as an adapter.
    direct = eeg_observables(raw)
    assert np.array_equal(observables.phases, direct.phases)


# --------------------------------------------------------------------------- #
# EDF ingestion (synthetic EDF — never the citation-only corpus)               #
# --------------------------------------------------------------------------- #


def _write_edf(
    path: Path,
    *,
    rates: list[float],
    seconds: int = 4,
    data: list[np.ndarray] | None = None,
) -> None:
    """Write a tiny multi-rate EDF for the loader and orchestration tests.

    Without ``data`` each channel is a plain 10 Hz tone; supplying ``data``
    (one array per rate) writes a custom recording, used to inject a rising
    coherence precursor for the end-to-end orchestration test.
    """
    import pyedflib

    writer = pyedflib.EdfWriter(
        str(path), len(rates), file_type=pyedflib.FILETYPE_EDFPLUS
    )
    try:
        headers = [
            {
                "label": f"ch{index}",
                "dimension": "uV",
                "sample_frequency": rate,
                "physical_max": 200.0,
                "physical_min": -200.0,
                "digital_max": 32767,
                "digital_min": -32768,
                "transducer": "",
                "prefilter": "",
            }
            for index, rate in enumerate(rates)
        ]
        writer.setSignalHeaders(headers)
        writer.setStartdatetime(dt.datetime(2009, 1, 1, 12, 0, 0))
        if data is None:
            data = [
                100.0
                * np.sin(_TWO_PI * 10.0 * np.arange(int(rate) * seconds) / float(rate))
                for rate in rates
            ]
        writer.writeSamples(data)
    finally:
        writer.close()


def _rising_coherence_channels(
    *, n_channels: int, fs: float, seconds: int, rise_second: int, seed: int
) -> list[np.ndarray]:
    """Return 10 Hz channels that lock to a common phase from ``rise_second``.

    Before the lock each channel carries an independent phase offset (low
    coherence); after it every channel shares one phase (high coherence) — a
    rising-synchronisation precursor the suite flags ahead of a later onset.
    """
    rng = np.random.default_rng(seed)
    n_samples = int(fs) * seconds
    rise = int(fs) * rise_second
    times = np.arange(n_samples) / fs
    offsets = rng.uniform(-np.pi, np.pi, n_channels)
    channels: list[np.ndarray] = []
    for channel in range(n_channels):
        phase = np.full(n_samples, offsets[channel])
        phase[rise:] = 0.0
        channels.append(100.0 * np.sin(_TWO_PI * 10.0 * times + phase))
    return channels


@_requires_pyedflib
def test_load_edf_channels_selects_the_expected_rate(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.edf"
    _write_edf(path, rates=[256.0, 256.0, 128.0])
    channels = load_edf_channels(path, expected_rate_hz=256.0)
    # Only the two 256 Hz channels are kept; the 128 Hz channel is dropped.
    assert channels.shape == (2, 256 * 4)


@_requires_pyedflib
def test_load_edf_channels_rejects_a_recording_without_the_rate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "off_rate.edf"
    _write_edf(path, rates=[128.0])
    with pytest.raises(ValueError, match="sampled at"):
        load_edf_channels(path, expected_rate_hz=256.0)


@_requires_pyedflib
def test_edf_start_datetime_reads_the_recording_stamp(tmp_path: Path) -> None:
    path = tmp_path / "stamped.edf"
    _write_edf(path, rates=[256.0, 256.0])
    assert edf_start_datetime(path).startswith("2009-01-01T12:00:00")


@_requires_pyedflib
def test_load_then_observables_wire_end_to_end(tmp_path: Path) -> None:
    path = tmp_path / "wire.edf"
    _write_edf(path, rates=[256.0, 256.0, 256.0], seconds=10)
    raw = load_edf_channels(path)
    observables = eeg_observables(raw)
    assert observables.n_nodes == 3
    assert observables.sampling_rate_hz == pytest.approx(32.0)
    assert observables.order_parameter.shape[0] == observables.n_samples


def test_module_constants_match_the_documented_pipeline() -> None:
    assert SAMPLING_RATE_HZ == 256.0
    assert BAND_HZ == (4.0, 30.0)
    assert WINDOW == 128
    assert STEP == 16


# --------------------------------------------------------------------------- #
# validator guards                                                            #
# --------------------------------------------------------------------------- #


def test_bandpass_rejects_boolean_signals() -> None:
    with pytest.raises(ValueError, match="boolean"):
        bandpass(np.ones((2, 512), dtype=bool), sampling_rate_hz=256.0)


def test_bandpass_rejects_non_numeric_signals() -> None:
    with pytest.raises(ValueError, match="real float array"):
        bandpass(np.array([["a", "b", "c", "d"]]), sampling_rate_hz=256.0)


def test_bandpass_rejects_a_three_dimensional_signal() -> None:
    with pytest.raises(ValueError, match="one- or two-dimensional"):
        bandpass(np.zeros((2, 2, 2)), sampling_rate_hz=256.0)


def test_bandpass_rejects_an_empty_signal() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        bandpass(np.zeros((2, 0)), sampling_rate_hz=256.0)


def test_bandpass_rejects_a_non_positive_rate() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        bandpass(np.zeros((2, 512)), sampling_rate_hz=0.0)


def test_bandpass_rejects_a_boolean_rate() -> None:
    with pytest.raises(ValueError, match="positive real"):
        bandpass(np.zeros((2, 512)), sampling_rate_hz=True)


def test_decimate_rejects_a_non_integer_factor() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        decimate_analytic_phase(np.zeros((2, 80)), factor=1.5)


# --------------------------------------------------------------------------- #
# main — end-to-end over synthetic EDFs (never the citation-only corpus)        #
# --------------------------------------------------------------------------- #


@_requires_pyedflib
def test_main_writes_sealed_derived_artefacts(tmp_path: Path) -> None:
    data = tmp_path / "corpus"
    data.mkdir()
    out = tmp_path / "derived"
    rates = [256.0, 256.0, 256.0, 256.0]
    # A short (320-sample, 10 s) segment keeps the synthetic recordings small; the
    # module's own SEGMENT_SAMPLES is used for the real corpus.
    segment_samples = 320
    for record in ("null_a", "null_b"):
        _write_edf(data / f"{record}.edf", rates=rates, seconds=12)
    # The seizure locks into coherence at 6 s with a later onset (11 s), so its
    # pre-onset segment has a clean incoherent baseline and a leading rise.
    _write_edf(
        data / "seizure_x.edf",
        rates=rates,
        data=_rising_coherence_channels(
            n_channels=len(rates), fs=256.0, seconds=12, rise_second=6, seed=2
        ),
    )
    # A seizure that never locks (the coherence rise is placed past the record) is
    # evaluated but leads no detector — a sealed silence, the honest half.
    _write_edf(
        data / "seizure_flat.edf",
        rates=rates,
        data=_rising_coherence_channels(
            n_channels=len(rates), fs=256.0, seconds=12, rise_second=100, seed=3
        ),
    )
    # An early onset (2 s) leaves no room for a clean pre-onset segment.
    _write_edf(data / "seizure_early.edf", rates=rates, seconds=12)

    main(
        data,
        out,
        interictal_records=("null_a", "null_b"),
        seizures={"seizure_x": 11, "seizure_flat": 11, "seizure_early": 2},
        segment_samples=segment_samples,
        baseline_fraction=1.0 / 3.0,
    )

    per_seizure = out / "seizure_x_early_warning_evidence.json"
    aggregate = out / "early_warning_leadtime_eeg_results.json"
    assert per_seizure.exists()
    # The early-onset seizure is excluded, not sealed as a silent null.
    assert not (out / "seizure_early_early_warning_evidence.json").exists()
    assert aggregate.exists()

    sealed = json.loads(per_seizure.read_text(encoding="utf-8"))
    assert sealed["record_id"] == "seizure_x"
    assert set(sealed["detectors"]) == set(DETECTORS)
    for detector in DETECTORS:
        assert sealed["detectors"][detector]["content_hash"]

    # The never-locking seizure is still sealed for every detector; without a
    # genuine coherence rise at least one detector does not lead it, so `main`
    # exercises the honestly-recorded no-lead path too.
    flat = json.loads(
        (out / "seizure_flat_early_warning_evidence.json").read_text(encoding="utf-8")
    )
    assert set(flat["detectors"]) == set(DETECTORS)
    assert any(
        flat["detectors"][detector]["warning_triggered"] is False
        for detector in DETECTORS
    )

    payload = json.loads(aggregate.read_text(encoding="utf-8"))
    assert payload["benchmark"] == "early_warning_leadtime_eeg"
    assert set(payload["matched_false_alarm_thresholds"]) == set(DETECTORS)
    assert payload["interictal_null_records"] == ["null_a", "null_b"]
    assert payload["n_null_trials"] == 2  # one 320-sample trial per 384-sample null
    assert payload["segment_seconds"] == pytest.approx(10.0)
    assert [e["record_id"] for e in payload["excluded_seizures"]] == ["seizure_early"]
    assert payload["verdict"]
