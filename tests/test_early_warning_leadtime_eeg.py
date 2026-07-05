# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — real-EEG early-warning capstone logic tests

"""Tests for the scalp-EEG early-warning lead-time capstone on synthetic data.

The capstone reads a citation-only corpus, so every signal-processing and
evaluation path is pinned here on **synthetic arrays** — never the protected raw
recordings. The observable pipeline (band-pass, analytic phase, phase-consistent
decimation), the matched-false-alarm calibration and lead machinery, the sealed
per-seizure evidence, and the EDF ingestion (on a synthetic EDF written in the
test) are each exercised directly, so the logic's correctness does not depend on
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
    ENSEMBLE_WEIGHTED,
    SAMPLING_RATE_HZ,
    STEP,
    SYNCHRONISATION,
    THRESHOLD_GRID,
    WINDOW,
    DetectorTrajectory,
    EEGObservables,
    SeizureLeadResult,
    analytic_phase,
    bandpass,
    calibrate_detectors,
    calibrate_threshold,
    decimate_analytic_phase,
    detector_trajectories,
    edf_start_datetime,
    eeg_observables,
    evaluate_seizure,
    false_alarm_rate,
    load_edf_channels,
    main,
    seizure_lead_samples,
)
from bench.early_warning_leadtime_eeg import _verdict as verdict
from scpn_phase_orchestrator.assurance.early_warning_evidence import (
    EARLY_WARNING_FLAGGED,
    NO_EARLY_WARNING,
)

_TWO_PI = 2.0 * np.pi

# The EDF-ingestion path needs pyedflib, an optional native dependency declared as
# the ``eeg`` extra (``pip install -e .[eeg]``). The pipeline, calibration, lead,
# sealing, and verdict tests never touch it; only the EDF I/O and end-to-end
# orchestration tests do, so they are gated on its presence — the same pattern the
# suite uses for jax, juliacall, and the other optional backends.
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


def _transition_observables(
    *,
    n_channels: int = 6,
    n_samples: int = 320,
    rise_sample: int = 150,
    fs: float = 32.0,
    seed: int = 0,
) -> EEGObservables:
    """Return observables whose channels lock into coherence from ``rise_sample``.

    The baseline is incoherent (independent random phases) and, from
    ``rise_sample`` on, every channel follows one shared phase with a little
    jitter — a rising-synchronisation precursor that the suite should flag while
    the labelled onset is placed later.
    """
    rng = np.random.default_rng(seed)
    phases = rng.uniform(-np.pi, np.pi, (n_channels, n_samples))
    shared = rng.uniform(-np.pi, np.pi, n_samples)
    jitter = 0.05 * rng.standard_normal((n_channels, n_samples))
    for channel in range(n_channels):
        phases[channel, rise_sample:] = (
            shared[rise_sample:] + jitter[channel, rise_sample:]
        )
    phases = np.arctan2(np.sin(phases), np.cos(phases))
    order = np.abs(np.mean(np.exp(1j * phases), axis=0))
    return EEGObservables(
        phases=np.ascontiguousarray(phases, dtype=np.float64),
        phase_field=np.ascontiguousarray(np.sin(phases), dtype=np.float64),
        order_parameter=np.ascontiguousarray(order, dtype=np.float64),
        sampling_rate_hz=fs,
    )


def _incoherent_observables(
    *, n_channels: int = 6, n_samples: int = 320, fs: float = 32.0, seed: int = 1
) -> EEGObservables:
    """Return a seizure-free (incoherent throughout) interictal-null observable."""
    rng = np.random.default_rng(seed)
    phases = rng.uniform(-np.pi, np.pi, (n_channels, n_samples))
    order = np.abs(np.mean(np.exp(1j * phases), axis=0))
    return EEGObservables(
        phases=np.ascontiguousarray(phases, dtype=np.float64),
        phase_field=np.ascontiguousarray(np.sin(phases), dtype=np.float64),
        order_parameter=np.ascontiguousarray(order, dtype=np.float64),
        sampling_rate_hz=fs,
    )


def _trajectory(
    values: list[float],
    *,
    relative_gate: float = 0.0,
    n_baseline: int = 0,
    relative: list[float] | None = None,
    name: str = "probe",
) -> DetectorTrajectory:
    """Return a detector trajectory with 16-sample hops for the shared alarm rule."""
    score = np.asarray(values, dtype=np.float64)
    gate = (
        np.ones(score.shape[0], dtype=np.float64)
        if relative is None
        else np.asarray(relative, dtype=np.float64)
    )
    return DetectorTrajectory(
        name=name,
        score=score,
        relative=gate,
        relative_gate=relative_gate,
        window_starts=np.arange(score.shape[0], dtype=np.int64) * STEP,
        n_baseline=n_baseline,
    )


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
# eeg_observables                                                              #
# --------------------------------------------------------------------------- #


def test_eeg_observables_shapes_rate_and_derived_fields() -> None:
    raw = _raw_tone(n_channels=4, n_samples=2400, fs=256.0, hz=10.0, seed=3)
    observables = eeg_observables(raw)
    assert observables.sampling_rate_hz == pytest.approx(32.0)
    assert observables.n_channels == 4
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


# --------------------------------------------------------------------------- #
# detector_trajectories                                                        #
# --------------------------------------------------------------------------- #


def test_detector_trajectories_share_one_window_grid() -> None:
    observables = _transition_observables(seed=5)
    trajectories = detector_trajectories(observables)
    assert set(trajectories) == set(DETECTORS)
    reference = trajectories[DETECTORS[0]].window_starts
    for name in DETECTORS:
        assert np.array_equal(trajectories[name].window_starts, reference)
        assert trajectories[name].score.shape == reference.shape


def test_detector_trajectories_flag_a_coherence_rise() -> None:
    observables = _transition_observables(seed=6)
    trajectories = detector_trajectories(observables)
    synchrony = trajectories[SYNCHRONISATION]
    # The post-baseline coherence lock drives the synchrony z-score up.
    post = synchrony.score[synchrony.n_baseline :]
    assert float(post.max()) > 3.0


# --------------------------------------------------------------------------- #
# shared alarm rule, calibration, lead                                         #
# --------------------------------------------------------------------------- #


def test_alarm_fires_on_a_sustained_post_baseline_breach() -> None:
    lead = seizure_lead_samples(
        _trajectory([0.0, 0.0, 5.0, 5.0, 0.0]),
        onset_sample=100,
        threshold=3.0,
        persistence=2,
    )
    assert lead == 100 - STEP * 2  # alarm at window_starts[2] = 32


def test_alarm_requires_the_full_persistence_run() -> None:
    lead = seizure_lead_samples(
        _trajectory([0.0, 5.0, 0.0, 5.0]),
        onset_sample=100,
        threshold=3.0,
        persistence=2,
    )
    assert lead is None


def test_alarm_relative_gate_blocks_a_high_score() -> None:
    trajectory = _trajectory(
        [0.0, 5.0, 5.0], relative=[0.0, 0.01, 0.01], relative_gate=0.5
    )
    assert (
        seizure_lead_samples(trajectory, onset_sample=100, threshold=3.0, persistence=2)
        is None
    )


def test_alarm_ignores_breaches_inside_the_baseline() -> None:
    trajectory = _trajectory([5.0, 5.0, 5.0, 5.0], n_baseline=2)
    lead = seizure_lead_samples(
        trajectory, onset_sample=100, threshold=3.0, persistence=2
    )
    assert lead == 100 - STEP * 2  # first breach at the first post-baseline window


def test_lead_is_none_when_the_alarm_follows_the_onset() -> None:
    lead = seizure_lead_samples(
        _trajectory([0.0, 0.0, 5.0, 5.0]),
        onset_sample=10,  # earlier than the alarm at sample 32
        threshold=3.0,
        persistence=2,
    )
    assert lead is None


def test_false_alarm_rate_counts_alarming_nulls() -> None:
    nulls = [_trajectory([0.0, 5.0, 5.0]) for _ in range(3)]
    nulls += [_trajectory([0.0, 0.0, 0.0]) for _ in range(7)]
    assert false_alarm_rate(nulls, 3.0, persistence=2) == pytest.approx(0.3)


def test_false_alarm_rate_rejects_an_empty_null() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        false_alarm_rate([], 3.0)


def test_calibrate_threshold_picks_the_smallest_meeting_the_target() -> None:
    nulls = [_trajectory([0.0, 5.0, 5.0]) for _ in range(3)]
    nulls += [_trajectory([0.0, 0.0, 0.0]) for _ in range(7)]
    threshold = calibrate_threshold(nulls, target_fa=0.1, persistence=2)
    assert threshold == 5.25
    assert false_alarm_rate(nulls, threshold, persistence=2) == 0.0


def test_calibrate_threshold_falls_back_to_the_largest_grid_value() -> None:
    nulls = [_trajectory([0.0, 100.0, 100.0]) for _ in range(4)]
    assert (
        calibrate_threshold(nulls, target_fa=0.0, persistence=2) == THRESHOLD_GRID[-1]
    )


def test_calibrate_threshold_rejects_an_empty_null() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        calibrate_threshold([])


def test_calibrate_detectors_returns_a_threshold_per_detector() -> None:
    nulls = [_incoherent_observables(seed=seed) for seed in (10, 11, 12)]
    thresholds = calibrate_detectors(nulls, target_fa=0.5)
    assert set(thresholds) == set(DETECTORS)
    for value in thresholds.values():
        assert THRESHOLD_GRID[0] <= value <= THRESHOLD_GRID[-1]


def test_calibrate_detectors_rejects_an_empty_null() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        calibrate_detectors([])


# --------------------------------------------------------------------------- #
# evaluate_seizure — sealing at the calibrated thresholds                      #
# --------------------------------------------------------------------------- #


def _evaluate(thresholds: dict[str, float], *, onset_sample: int) -> SeizureLeadResult:
    observables = _transition_observables(rise_sample=150, n_samples=320, seed=7)
    return evaluate_seizure(
        observables,
        record_id="synthetic_01",
        onset_sample=onset_sample,
        signal_source="synthetic coherence-rise fixture",
        captured_at="2009-01-01T12:00:00",
        thresholds=thresholds,
    )


def test_evaluate_seizure_seals_every_detector_with_a_leading_alarm() -> None:
    thresholds = dict.fromkeys(DETECTORS, 0.0)
    result = _evaluate(thresholds, onset_sample=300)
    assert set(result.evidences) == set(DETECTORS)
    synchrony = result.evidences[SYNCHRONISATION]
    assert synchrony.verdict == EARLY_WARNING_FLAGGED
    assert synchrony.warning_triggered is True
    assert synchrony.lead_is_early is True
    assert synchrony.content_hash  # sealed
    # The synchrony lead is reported in seconds against the 32 Hz analysis rate.
    assert result.lead_seconds()[SYNCHRONISATION] is not None


def test_evaluate_seizure_seals_a_silence_when_no_detector_fires() -> None:
    thresholds = dict.fromkeys(DETECTORS, 1.0e3)
    result = _evaluate(thresholds, onset_sample=300)
    for name in DETECTORS:
        evidence = result.evidences[name]
        assert evidence.verdict == NO_EARLY_WARNING
        assert evidence.warning_triggered is False
        assert result.lead_seconds()[name] is None


def test_seizure_lead_result_audit_record_round_trips() -> None:
    result = _evaluate(dict.fromkeys(DETECTORS, 0.0), onset_sample=300)
    record = result.to_audit_record()
    encoded = json.loads(json.dumps(record))
    assert encoded["record_id"] == "synthetic_01"
    assert encoded["onset_sample"] == 300
    assert set(encoded["detectors"]) == set(DETECTORS)
    assert encoded["detectors"][SYNCHRONISATION]["content_hash"]


# --------------------------------------------------------------------------- #
# verdict                                                                      #
# --------------------------------------------------------------------------- #


def test_verdict_names_a_fusion_lead_when_it_beats_every_member() -> None:
    leads = {
        "critical_slowing_down": [10.0],
        "synchronisation": [20.0],
        "transition_entropy": [15.0],
        ENSEMBLE_WEIGHTED: [40.0],
    }
    assert verdict(leads).startswith("FUSION LEADS")


def test_verdict_reports_no_advantage_when_a_member_matches_the_fusion() -> None:
    leads = {
        "critical_slowing_down": [10.0],
        "synchronisation": [50.0],
        "transition_entropy": [15.0],
        ENSEMBLE_WEIGHTED: [40.0],
    }
    assert verdict(leads).startswith("NO FUSION ADVANTAGE")


def test_verdict_reports_no_fusion_lead_when_the_fusion_is_silent() -> None:
    leads = {
        "critical_slowing_down": [10.0],
        "synchronisation": [20.0],
        "transition_entropy": [15.0],
        ENSEMBLE_WEIGHTED: [],
    }
    assert verdict(leads).startswith("NO FUSION LEAD")


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
    assert observables.n_channels == 3
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
    for record in ("null_a", "null_b"):
        _write_edf(data / f"{record}.edf", rates=rates, seconds=12)
    # The seizure recording locks into coherence at 6 s and is labelled with a
    # later onset (11 s), so the suite raises a leading alarm the run accumulates.
    _write_edf(
        data / "seizure_x.edf",
        rates=rates,
        data=_rising_coherence_channels(
            n_channels=len(rates), fs=256.0, seconds=12, rise_second=6, seed=2
        ),
    )

    main(
        data,
        out,
        interictal_records=("null_a", "null_b"),
        seizures={"seizure_x": 11},
    )

    per_seizure = out / "seizure_x_early_warning_evidence.json"
    aggregate = out / "early_warning_leadtime_eeg_results.json"
    assert per_seizure.exists()
    assert aggregate.exists()

    sealed = json.loads(per_seizure.read_text(encoding="utf-8"))
    assert sealed["record_id"] == "seizure_x"
    assert set(sealed["detectors"]) == set(DETECTORS)
    for detector in DETECTORS:
        assert sealed["detectors"][detector]["content_hash"]

    payload = json.loads(aggregate.read_text(encoding="utf-8"))
    assert payload["benchmark"] == "early_warning_leadtime_eeg"
    assert set(payload["matched_false_alarm_thresholds"]) == set(DETECTORS)
    assert payload["interictal_null_records"] == ["null_a", "null_b"]
    assert payload["verdict"]
