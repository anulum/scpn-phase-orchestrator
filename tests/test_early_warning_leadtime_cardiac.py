# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — real-cardiac early-warning capstone logic tests

"""Tests for the cardiac-ECG early-warning capstone adapter on synthetic data.

This capstone is the cardiac adapter onto the domain-neutral harness (tested in
``tests/test_early_warning_domain.py``): the only cardiac-specific work is the
signal-processing pipeline that produces the neutral bundle, the WFDB ingestion,
and the end-to-end orchestration. Those read a citation-only corpus, so every
path is pinned here on **synthetic arrays and a synthetic WFDB record written in
the test** — never the protected raw recordings. ``cardiac_observables``, the
:class:`CardiacPhaseAdapter`, the WFDB reader and rhythm-annotation parsing, and
the end-to-end ``main`` (over synthetic records) are each exercised directly.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from bench.early_warning_leadtime_cardiac import (
    BAND_HZ,
    DETECTORS,
    SAMPLING_RATE_HZ,
    STEP,
    WINDOW,
    CardiacPhaseAdapter,
    afib_onsets,
    cardiac_observables,
    load_wfdb_leads,
    longest_sinus_span,
    main,
    rhythm_transitions,
)
from scpn_phase_orchestrator.monitor.early_warning_suite import (
    DomainObservableAdapter,
    SuiteObservables,
)

_TWO_PI = 2.0 * np.pi

# The WFDB-ingestion path needs wfdb, an optional dependency declared as the
# ``cardiac`` extra (``pip install -e .[cardiac]``). The pipeline and adapter
# tests never touch it; only the WFDB I/O and end-to-end orchestration tests do,
# so they are gated on its presence — the pattern the suite uses for the other
# optional backends.
_requires_wfdb = pytest.mark.skipif(
    importlib.util.find_spec("wfdb") is None,
    reason="wfdb is an optional dependency (install the 'cardiac' extra)",
)


# --------------------------------------------------------------------------- #
# Synthetic fixtures                                                           #
# --------------------------------------------------------------------------- #


def _two_lead_tone(
    *,
    n_samples: int,
    fs: float = SAMPLING_RATE_HZ,
    hz: float = 10.0,
    offset: float = 0.5,
) -> np.ndarray:
    """Return a two-lead ECG-like tone with a fixed inter-lead phase offset."""
    times = np.arange(n_samples) / fs
    lead0 = np.sin(_TWO_PI * hz * times)
    lead1 = np.sin(_TWO_PI * hz * times + offset)
    return np.vstack([lead0, lead1])


def _converging_two_lead(
    *, n_samples: int, fs: float = SAMPLING_RATE_HZ, hz: float = 10.0
) -> np.ndarray:
    """Return two leads whose inter-lead phase offset shrinks to zero at the end.

    The inter-lead coherence therefore rises toward the recording's end — a
    synthetic rising-synchronisation precursor, so a detector can lead an onset
    placed at the end (the test exercises the leading path, not cardiac
    physiology, whose AF onset is a desynchronisation).
    """
    times = np.arange(n_samples) / fs
    offset = np.linspace(0.5 * np.pi, 0.0, n_samples)
    lead0 = np.sin(_TWO_PI * hz * times)
    lead1 = np.sin(_TWO_PI * hz * times + offset)
    return np.vstack([lead0, lead1])


# --------------------------------------------------------------------------- #
# cardiac_observables and the CardiacPhaseAdapter                              #
# --------------------------------------------------------------------------- #


def test_cardiac_observables_shapes_rate_and_derived_fields() -> None:
    raw = _two_lead_tone(n_samples=5000)
    observables = cardiac_observables(raw)
    assert isinstance(observables, SuiteObservables)
    assert observables.sampling_rate_hz == pytest.approx(50.0)
    assert observables.n_nodes == 2
    assert observables.n_samples == 1000
    assert np.allclose(observables.phase_field, np.sin(observables.phases))
    assert np.all(observables.order_parameter >= 0.0)
    assert np.all(observables.order_parameter <= 1.0 + 1.0e-9)


def test_cardiac_observables_rejects_a_single_lead() -> None:
    with pytest.raises(ValueError, match="at least two leads"):
        cardiac_observables(np.zeros((1, 5000)))


def test_cardiac_observables_rejects_a_complex_recording() -> None:
    with pytest.raises(ValueError, match="real-valued"):
        cardiac_observables(np.ones((2, 5000), dtype=np.complex128))


def test_cardiac_observables_rejects_a_non_finite_recording() -> None:
    raw = _two_lead_tone(n_samples=5000)
    raw[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        cardiac_observables(raw)


def test_cardiac_phase_adapter_satisfies_the_protocol_and_wraps_the_pipeline() -> None:
    adapter = CardiacPhaseAdapter()
    assert isinstance(adapter, DomainObservableAdapter)
    assert adapter.domain == "cardiac_ecg"
    raw = _two_lead_tone(n_samples=5000)
    observables = adapter.observables(raw)
    assert isinstance(observables, SuiteObservables)
    assert observables.n_nodes == 2
    direct = cardiac_observables(raw)
    assert np.array_equal(observables.phases, direct.phases)


def test_module_constants_match_the_documented_pipeline() -> None:
    assert SAMPLING_RATE_HZ == 250.0
    assert BAND_HZ == (5.0, 20.0)
    assert WINDOW == 200
    assert STEP == 25


# --------------------------------------------------------------------------- #
# WFDB ingestion (synthetic WFDB record — never the citation-only corpus)      #
# --------------------------------------------------------------------------- #


def _write_afdb_record(
    directory: Path,
    name: str,
    *,
    leads: np.ndarray,
    transitions: list[tuple[int, str]],
    fs: float = SAMPLING_RATE_HZ,
) -> None:
    """Write a synthetic two-lead WFDB record with a rhythm-annotation stream."""
    import wfdb

    wfdb.wrsamp(
        name,
        fs=fs,
        units=["mV", "mV"],
        sig_name=["ECG1", "ECG2"],
        p_signal=np.ascontiguousarray(leads.T, dtype=np.float64),
        fmt=["16", "16"],
        write_dir=str(directory),
    )
    samples = np.array([sample for sample, _ in transitions], dtype=np.int64)
    wfdb.wrann(
        name,
        "atr",
        sample=samples,
        symbol=["+"] * len(transitions),
        aux_note=[label for _, label in transitions],
        fs=fs,
        write_dir=str(directory),
    )


@_requires_wfdb
def test_load_wfdb_leads_reads_two_leads(tmp_path: Path) -> None:
    _write_afdb_record(
        tmp_path, "rec", leads=_two_lead_tone(n_samples=2000), transitions=[(0, "(N")]
    )
    leads = load_wfdb_leads(tmp_path / "rec")
    assert leads.shape == (2, 2000)
    assert np.all(np.isfinite(leads))


@_requires_wfdb
def test_load_wfdb_leads_zeroes_gap_samples(tmp_path: Path) -> None:
    raw = _two_lead_tone(n_samples=2000)
    raw[0, 5] = np.nan
    _write_afdb_record(tmp_path, "gap", leads=raw, transitions=[(0, "(N")])
    leads = load_wfdb_leads(tmp_path / "gap")
    assert np.all(np.isfinite(leads))


@_requires_wfdb
def test_load_wfdb_leads_rejects_a_wrong_rate(tmp_path: Path) -> None:
    _write_afdb_record(
        tmp_path,
        "slow",
        leads=_two_lead_tone(n_samples=2000, fs=200.0),
        transitions=[(0, "(N")],
        fs=200.0,
    )
    with pytest.raises(ValueError, match="sampled at"):
        load_wfdb_leads(tmp_path / "slow", expected_rate_hz=250.0)


@_requires_wfdb
def test_load_wfdb_leads_rejects_a_single_lead(tmp_path: Path) -> None:
    import wfdb

    times = np.arange(2000) / SAMPLING_RATE_HZ
    wfdb.wrsamp(
        "mono",
        fs=SAMPLING_RATE_HZ,
        units=["mV"],
        sig_name=["ECG1"],
        p_signal=np.sin(_TWO_PI * 10.0 * times).reshape(-1, 1),
        fmt=["16"],
        write_dir=str(tmp_path),
    )
    with pytest.raises(ValueError, match="fewer than two leads"):
        load_wfdb_leads(tmp_path / "mono")


@_requires_wfdb
def test_rhythm_transitions_reads_the_annotation_stream(tmp_path: Path) -> None:
    _write_afdb_record(
        tmp_path,
        "rhythm",
        leads=_two_lead_tone(n_samples=3000),
        transitions=[(0, "(N"), (1500, "(AFIB"), (2200, "(N")],
    )
    transitions = rhythm_transitions(tmp_path / "rhythm")
    assert transitions == [(0, "(N"), (1500, "(AFIB"), (2200, "(N")]


@_requires_wfdb
def test_afib_onsets_keeps_only_clean_baselines(tmp_path: Path) -> None:
    _write_afdb_record(
        tmp_path,
        "clean",
        leads=_two_lead_tone(n_samples=3000),
        transitions=[(0, "(N"), (2500, "(AFIB")],
    )
    onsets = afib_onsets(tmp_path / "clean", min_baseline_native=2000)
    assert onsets == [2500]


@_requires_wfdb
def test_afib_onsets_excludes_an_early_onset(tmp_path: Path) -> None:
    _write_afdb_record(
        tmp_path,
        "early",
        leads=_two_lead_tone(n_samples=3000),
        transitions=[(0, "(N"), (500, "(AFIB")],
    )
    assert afib_onsets(tmp_path / "early", min_baseline_native=2000) == []


@_requires_wfdb
def test_afib_onsets_ignores_an_onset_without_a_sinus_predecessor(
    tmp_path: Path,
) -> None:
    # A record that opens in AF has no preceding sinus baseline at index 0.
    _write_afdb_record(
        tmp_path,
        "opens_af",
        leads=_two_lead_tone(n_samples=3000),
        transitions=[(0, "(AFIB"), (1500, "(N")],
    )
    assert afib_onsets(tmp_path / "opens_af", min_baseline_native=1000) == []


@_requires_wfdb
def test_longest_sinus_span_finds_the_widest_normal_stretch(tmp_path: Path) -> None:
    _write_afdb_record(
        tmp_path,
        "spans",
        leads=_two_lead_tone(n_samples=5000),
        transitions=[(0, "(N"), (500, "(AFIB"), (1000, "(N")],
    )
    # The (N from 1000 to the 5000-sample end (4000) beats the 0–500 stretch.
    assert longest_sinus_span(tmp_path / "spans") == (1000, 5000)


@_requires_wfdb
def test_longest_sinus_span_keeps_the_first_widest_stretch(tmp_path: Path) -> None:
    _write_afdb_record(
        tmp_path,
        "first_widest",
        leads=_two_lead_tone(n_samples=5000),
        transitions=[(0, "(N"), (3000, "(AFIB"), (4000, "(N"), (4500, "(AFIB")],
    )
    # The opening (N (0–3000) is wider than the later 4000–4500 stretch.
    assert longest_sinus_span(tmp_path / "first_widest") == (0, 3000)


@_requires_wfdb
def test_longest_sinus_span_rejects_a_record_without_sinus(tmp_path: Path) -> None:
    _write_afdb_record(
        tmp_path,
        "no_sinus",
        leads=_two_lead_tone(n_samples=3000),
        transitions=[(0, "(AFIB")],
    )
    with pytest.raises(ValueError, match="no sinus rhythm"):
        longest_sinus_span(tmp_path / "no_sinus")


# --------------------------------------------------------------------------- #
# main — end-to-end over synthetic WFDB records (never the citation-only corpus) #
# --------------------------------------------------------------------------- #


@_requires_wfdb
def test_main_writes_sealed_derived_artefacts(tmp_path: Path) -> None:
    data = tmp_path / "corpus"
    data.mkdir()
    out = tmp_path / "derived"
    # A short (400-sample decimated, 2000-sample native) segment keeps the
    # synthetic records small; the module's own SEGMENT_SAMPLES is used for the
    # real corpus.
    segment_samples = 400
    # Two sinus null records (all normal), each long enough for one null trial.
    for record in ("null_a", "null_b"):
        _write_afdb_record(
            data, record, leads=_two_lead_tone(n_samples=2400), transitions=[(0, "(N")]
        )
    # An AF record whose leads lock into coherence toward the onset (leads fire).
    _write_afdb_record(
        data,
        "af_lead",
        leads=_converging_two_lead(n_samples=2500),
        transitions=[(0, "(N"), (2500, "(AFIB")],
    )
    # An AF record with no precursor (a flat inter-lead offset) — no detector leads.
    _write_afdb_record(
        data,
        "af_flat",
        leads=_two_lead_tone(n_samples=2500),
        transitions=[(0, "(N"), (2500, "(AFIB")],
    )
    # An AF record whose onset is too early for a clean pre-onset segment.
    _write_afdb_record(
        data,
        "af_early",
        leads=_two_lead_tone(n_samples=2500),
        transitions=[(0, "(N"), (400, "(AFIB")],
    )

    main(
        data,
        out,
        af_records=("af_lead", "af_flat", "af_early"),
        null_records=("null_a", "null_b"),
        segment_samples=segment_samples,
        baseline_fraction=1.0 / 3.0,
    )

    aggregate = out / "early_warning_leadtime_cardiac_results.json"
    assert aggregate.exists()
    assert (out / "af_lead_early_warning_evidence.json").exists()
    # The early-onset record is excluded, not sealed as a silent null.
    assert not (out / "af_early_early_warning_evidence.json").exists()

    sealed = json.loads(
        (out / "af_lead_early_warning_evidence.json").read_text(encoding="utf-8")
    )
    assert set(sealed["detectors"]) == set(DETECTORS)
    for detector in DETECTORS:
        assert sealed["detectors"][detector]["content_hash"]

    # The flat record is still sealed for every detector; without a genuine
    # precursor at least one detector does not lead it.
    flat = json.loads(
        (out / "af_flat_early_warning_evidence.json").read_text(encoding="utf-8")
    )
    assert set(flat["detectors"]) == set(DETECTORS)
    assert any(
        flat["detectors"][detector]["warning_triggered"] is False
        for detector in DETECTORS
    )

    payload = json.loads(aggregate.read_text(encoding="utf-8"))
    assert payload["benchmark"] == "early_warning_leadtime_cardiac"
    assert set(payload["matched_false_alarm_thresholds"]) == set(DETECTORS)
    assert payload["sinus_null_records"] == ["null_a", "null_b"]
    assert [e["record_id"] for e in payload["excluded_records"]] == ["af_early"]
    assert payload["verdict"]
