# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — palaeoclimate early-warning capstone tests

"""Tests for the palaeoclimate early-warning capstone on synthetic data.

The climate-specific work is the Gaussian detrend (reproducing the ``earlywarnings``
toolbox's ``ksmooth`` scaling), the two-file text ingestion, the record segmentation,
and the end-to-end ``main``. Every path is exercised here on synthetic arrays and
synthetic text files written in the test, so the coverage never depends on the
citation-only palaeoclimate corpus. The pipeline uses only numpy and scipy, so no path
is skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bench.early_warning_leadtime_climate import (
    CLIMATE_RECORDS,
    KSMOOTH_NORMAL_SCALE,
    ClimateProxyAdapter,
    ClimateRecord,
    _segment_record,
    climate_observable,
    gaussian_detrend,
    load_climate_series,
    main,
)
from bench.early_warning_single_series import SingleSeriesObservable

# --------------------------------------------------------------------------- #
# Synthetic fixtures                                                          #
# --------------------------------------------------------------------------- #


def _proxy_series(*, n_samples: int = 1200, seed: int = 0) -> np.ndarray:
    """Return an equidistant proxy whose fluctuation amplitude ramps to its end.

    A slow trend plus fluctuations whose amplitude grows towards the youngest end — a
    critical-slowing-down approach the detector can lead, on an already-equidistant
    series like the toolbox ``_Y1int`` files.
    """
    rng = np.random.default_rng(seed)
    trend = np.linspace(0.0, 5.0, n_samples)
    amp = np.linspace(0.3, 4.0, n_samples)
    return trend + rng.standard_normal(n_samples) * amp


def _write_record(
    data: Path, stem: str, proxy: np.ndarray, *, age_lo: float, age_hi: float
) -> None:
    """Write a synthetic ``_Y1int`` proxy and a ``_Yt`` age axis for a record."""
    np.savetxt(data / f"{stem}_Y1int.txt", proxy)
    # An age axis whose range is [age_lo, age_hi]; its length need not match the proxy.
    np.savetxt(data / f"{stem}_Yt.txt", np.linspace(age_hi, age_lo, len(proxy) + 20))


def _record(record_id: str, stem: str, age_unit_years: float = 1.0) -> ClimateRecord:
    """Return a ClimateRecord pointing at synthetic text files in the data directory."""
    return ClimateRecord(
        record_id=record_id,
        proxy_file=f"{stem}_Y1int.txt",
        time_file=f"{stem}_Yt.txt",
        age_unit_years=age_unit_years,
        proxy_description="synthetic detrended proxy",
        citation="synthetic record",
    )


# --------------------------------------------------------------------------- #
# gaussian_detrend                                                            #
# --------------------------------------------------------------------------- #


def test_gaussian_detrend_removes_a_linear_trend() -> None:
    series = np.linspace(0.0, 100.0, 500) + np.sin(np.arange(500) * 0.3)
    residual = gaussian_detrend(series, bandwidth_percent=10.0)
    assert abs(float(residual.mean())) < 1.0
    assert float(residual.std()) < float(series.std())
    assert residual.shape == series.shape


def test_gaussian_detrend_floors_a_tiny_bandwidth_at_one_sample() -> None:
    # N * pct / 100 rounds below one, so the bandwidth is floored to a single sample.
    residual = gaussian_detrend(np.arange(10.0), bandwidth_percent=1.0)
    assert residual.shape == (10,)


def test_gaussian_detrend_rejects_a_two_dimensional_series() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        gaussian_detrend(np.zeros((2, 5)))


@pytest.mark.parametrize("bandwidth", [0.0, 150.0, np.nan])
def test_gaussian_detrend_rejects_an_out_of_range_bandwidth(bandwidth: float) -> None:
    with pytest.raises(ValueError, match=r"\(0, 100\]"):
        gaussian_detrend(np.arange(20.0), bandwidth_percent=bandwidth)


def test_ksmooth_scale_is_the_r_normal_kernel_constant() -> None:
    # sigma = 0.25 / qnorm(0.75); guards against silent drift of the scaling factor.
    expected = 0.25 / 0.6744897501960817
    assert pytest.approx(expected, rel=1e-6) == KSMOOTH_NORMAL_SCALE


# --------------------------------------------------------------------------- #
# climate_observable / ClimateProxyAdapter                                    #
# --------------------------------------------------------------------------- #


def test_climate_observable_places_the_span_across_the_record() -> None:
    proxy = _proxy_series(n_samples=1001, seed=1)
    observable = climate_observable(proxy, span_years=2000.0)
    assert isinstance(observable, SingleSeriesObservable)
    # 1000 gaps over 2000 yr -> 0.5 samples per year.
    assert observable.sampling_rate_hz == pytest.approx(0.5)
    assert abs(float(observable.series.mean())) < 1.0  # detrended, centred


def test_climate_observable_rejects_a_two_dimensional_proxy() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        climate_observable(np.zeros((2, 5)), span_years=100.0)


def test_climate_observable_rejects_a_too_short_proxy() -> None:
    with pytest.raises(ValueError, match="at least three"):
        climate_observable(np.array([1.0, 2.0]), span_years=100.0)


def test_climate_observable_rejects_a_non_finite_proxy() -> None:
    with pytest.raises(ValueError, match="finite"):
        climate_observable(np.array([1.0, np.nan, 3.0]), span_years=100.0)


@pytest.mark.parametrize("span", [0.0, -5.0, np.inf])
def test_climate_observable_rejects_a_non_positive_span(span: float) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        climate_observable(np.arange(10.0), span_years=span)


def test_adapter_labels_the_domain_and_delegates() -> None:
    adapter = ClimateProxyAdapter()
    assert adapter.domain == "palaeoclimate"
    proxy = _proxy_series(seed=2)
    observable = adapter.observable(proxy, span_years=500.0)
    reference = climate_observable(
        proxy,
        span_years=500.0,
        detrend_bandwidth_percent=adapter.detrend_bandwidth_percent,
    )
    assert np.array_equal(observable.series, reference.series)


# --------------------------------------------------------------------------- #
# load_climate_series                                                         #
# --------------------------------------------------------------------------- #


def test_load_climate_series_reads_the_proxy_and_span(tmp_path: Path) -> None:
    proxy = _proxy_series(n_samples=200, seed=3)
    _write_record(tmp_path, "rec", proxy, age_lo=1000.0, age_hi=3000.0)
    loaded, span = load_climate_series(
        tmp_path / "rec_Y1int.txt", tmp_path / "rec_Yt.txt", age_unit_years=1.0
    )
    assert np.allclose(loaded, proxy)
    assert span == pytest.approx(2000.0)  # (3000 - 1000) yr


def test_load_climate_series_scales_the_age_unit(tmp_path: Path) -> None:
    proxy = _proxy_series(n_samples=200, seed=4)
    _write_record(tmp_path, "ma", proxy, age_lo=32.0, age_hi=40.0)  # Ma
    _, span = load_climate_series(
        tmp_path / "ma_Y1int.txt", tmp_path / "ma_Yt.txt", age_unit_years=1.0e6
    )
    assert span == pytest.approx(8.0e6)  # (40 - 32) Ma -> 8 Myr


def test_load_climate_series_rejects_a_too_short_proxy(tmp_path: Path) -> None:
    np.savetxt(tmp_path / "s_Y1int.txt", np.array([1.0, 2.0]))
    np.savetxt(tmp_path / "s_Yt.txt", np.array([3.0, 2.0, 1.0]))
    with pytest.raises(ValueError, match="at least three"):
        load_climate_series(
            tmp_path / "s_Y1int.txt", tmp_path / "s_Yt.txt", age_unit_years=1.0
        )


def test_load_climate_series_rejects_a_non_positive_age_unit(tmp_path: Path) -> None:
    proxy = _proxy_series(n_samples=50, seed=5)
    _write_record(tmp_path, "u", proxy, age_lo=1.0, age_hi=2.0)
    with pytest.raises(ValueError, match="positive finite"):
        load_climate_series(
            tmp_path / "u_Y1int.txt", tmp_path / "u_Yt.txt", age_unit_years=0.0
        )


# --------------------------------------------------------------------------- #
# ClimateRecord / corpus                                                      #
# --------------------------------------------------------------------------- #


def test_shipped_corpus_holds_the_eight_dakos_records() -> None:
    ids = {record.record_id for record in CLIMATE_RECORDS}
    assert len(CLIMATE_RECORDS) == 8
    assert "younger_dryas_termination" in ids
    assert "eocene_oligocene_greenhouse_end" in ids
    eo = next(
        r for r in CLIMATE_RECORDS if r.record_id == "eocene_oligocene_greenhouse_end"
    )
    assert eo.age_unit_years == pytest.approx(1.0e6)  # ages in Ma


# --------------------------------------------------------------------------- #
# _segment_record                                                             #
# --------------------------------------------------------------------------- #


def test_segment_record_splits_a_long_record(tmp_path: Path) -> None:
    proxy = _proxy_series(n_samples=1200, seed=6)
    _write_record(tmp_path, "long", proxy, age_lo=1000.0, age_hi=3000.0)
    segments = _segment_record(
        _record("long", "long"), ClimateProxyAdapter(), tmp_path, segment_samples=100
    )
    assert segments is not None
    assert segments.transition_segment.n_samples == 100
    assert segments.null_region.n_samples == 1100


def test_segment_record_rejects_a_too_short_record(tmp_path: Path) -> None:
    proxy = _proxy_series(n_samples=150, seed=7)
    _write_record(tmp_path, "short", proxy, age_lo=1000.0, age_hi=1100.0)
    assert (
        _segment_record(
            _record("short", "short"),
            ClimateProxyAdapter(),
            tmp_path,
            segment_samples=100,
        )
        is None
    )


# --------------------------------------------------------------------------- #
# main — end-to-end over a synthetic text corpus                              #
# --------------------------------------------------------------------------- #


def test_main_seals_a_transition_and_writes_the_aggregate(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    _write_record(
        data, "rec_a", _proxy_series(n_samples=1600, seed=8), age_lo=1e4, age_hi=6e4
    )
    out = tmp_path / "out"
    main(data, out, records=[_record("rec_a", "rec_a")], segment_samples=200)

    evidence = json.loads((out / "rec_a_early_warning_evidence.json").read_text())
    assert evidence["record_id"] == "rec_a"
    assert evidence["detector"]["detector"] == "critical_slowing_down"
    assert evidence["detector"]["content_hash"]

    results = json.loads(
        (out / "early_warning_leadtime_climate_results.json").read_text()
    )
    assert results["benchmark"] == "early_warning_leadtime_climate"
    assert results["n_null_trials"] >= 1
    assert "critical_slowing_down" in results["achieved_false_alarm"]
    assert len(results["transitions"]) == 1
    assert "climate transitions" in results["verdict"]
    significance = results["permutation_significance"]
    assert significance["n_transitions"] == 1
    assert 0.0 < significance["p_value"] <= 1.0
    assert significance["seed"] == 0


def test_main_reports_a_too_short_record_as_excluded(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    _write_record(
        data, "long", _proxy_series(n_samples=1600, seed=9), age_lo=1e4, age_hi=6e4
    )
    _write_record(
        data, "short", _proxy_series(n_samples=150, seed=10), age_lo=1e3, age_hi=1.1e3
    )
    out = tmp_path / "out"
    records = [_record("long", "long"), _record("short", "short")]
    main(data, out, records=records, segment_samples=200)

    results = json.loads(
        (out / "early_warning_leadtime_climate_results.json").read_text()
    )
    excluded = {row["record_id"] for row in results["excluded_records"]}
    assert "short" in excluded
    assert len(results["transitions"]) == 1


def test_main_is_byte_reproducible(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    _write_record(
        data, "rec_a", _proxy_series(n_samples=1600, seed=8), age_lo=1e4, age_hi=6e4
    )
    record = _record("rec_a", "rec_a")
    main(data, tmp_path / "out1", records=[record], segment_samples=200)
    main(data, tmp_path / "out2", records=[record], segment_samples=200)
    a = (tmp_path / "out1" / "rec_a_early_warning_evidence.json").read_text()
    b = (tmp_path / "out2" / "rec_a_early_warning_evidence.json").read_text()
    assert a == b
