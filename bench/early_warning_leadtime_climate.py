# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — real palaeoclimate early-warning lead-time capstone

"""Real palaeoclimate early-warning capstone: matched-false-alarm transition lead time.

The fourth domain the early-warning design is proven on, and the first that carries a
single scalar observable rather than a population of coupled oscillators. Where the
scalp-EEG, cardiac-ECG and grid-PMU capstones feed a multi-node
:class:`~scpn_phase_orchestrator.monitor.early_warning_suite.SuiteObservables` bundle
into the multi-node harness, a palaeoclimate proxy record is one time-series
approaching an abrupt transition, so this capstone feeds a
:class:`~bench.early_warning_single_series.SingleSeriesObservable` into the
single-series critical-slowing-down harness (:mod:`bench.early_warning_single_series`).
Rising synchronisation and ordinal-transition entropy have no meaning on one series;
critical slowing down — the rising variance and lag-one autocorrelation ahead of a
bifurcation — is exactly the indicator Dakos et al. 2008 read on these records.

Corpus
------
The eight abrupt-climate-transition proxy records analysed by Dakos et al. 2008 (PNAS
105:14308), taken from the authors' own Early-Warning-Signals Toolbox dataset
repository (``earlywarningtoolbox/datasets``): the Eocene–Oligocene greenhouse end, the
four Vostok deglaciation terminations, the Bølling–Allerød onset, the Younger Dryas
termination, and the North-African desertification. Each ships as an equidistant
interpolated proxy series (the toolbox's ``_Y1int`` file — the very series the paper's
analysis reads) and an age axis (``_Yt``); the record's curated interval ends at the
abrupt transition, so the whole series is the pre-transition approach. The raw files are
**citation-only** — obtained from that public repository and **never redistributed
here**. This module reads them, but its tests exercise every path on **synthetic series
and synthetic text files**, so the coverage never depends on the corpus.

Observable pipeline and its honest limits
-----------------------------------------
Each record is the toolbox's equidistant ``_Y1int`` proxy, Gaussian-detrended the way
the ``earlywarnings`` toolbox detrends (:func:`gaussian_detrend`): the bandwidth is a
percentage of the record length and R's ``ksmooth(kernel = "normal")`` scaling turns it
into a Gaussian ``sigma``, so the residual the SCPN critical-slowing-down monitor reads
is the residual the toolbox forms. Two honest caveats belong on this proof, not buried:

* **A different statistic from Dakos.** Dakos et al. read a *rising Kendall-τ trend* of
  the lag-one autocorrelation and test it against phase-randomised surrogates. The SCPN
  monitor instead raises a robust (median/MAD) z-score alarm of variance-or-lag-one
  against a leading baseline at a matched false-alarm operating point. This capstone
  therefore measures what the *shipped SCPN detector* does on the Dakos records; it is
  **not** a reproduction of the Dakos AR(1)-trend result, and a silence here is not a
  refutation of their finding.
* **A within-record null.** With one series per transition there is no separate
  no-transition recording, so the matched-false-alarm null is the stable pre-approach
  interval of each record, pooled across the corpus. That interval carries the record's
  own baseline variability, making the calibration conservative — reported as-is.

Each transition is scored on the fixed pre-onset segment ending at the record's final
(youngest) sample, where the curated transition sits; the matched false-alarm threshold
is calibrated on non-overlapping null trials cut from the pooled stable pre-approach
intervals; every alarm or silence is sealed into a claim-bounded
:class:`~scpn_phase_orchestrator.assurance.early_warning_evidence.EarlyWarningEvidence`.
A record too short to yield both a full pre-onset segment and one preceding null trial
is reported and excluded, never counted as a silent miss. A label-permutation
significance test then asks whether the lead count beats the matched false-alarm rate or
is what chance gives, and its p-value is recorded in the aggregate. Only the derived,
sealed artefacts are committed; the raw proxy series never is.

References
----------
* Dakos, Scheffer, van Nes, Brovkin, Petoukhov & Held 2008, *PNAS* 105:14308 — slowing
  down as an early warning signal for abrupt climate change (the eight records).
* Dakos & Lahti, *earlywarnings* R package / ``earlywarningtoolbox/datasets`` — the
  reference toolbox whose Gaussian-``ksmooth`` detrending this pipeline reproduces and
  whose interpolated series it reads.
* Scheffer et al. 2009, *Nature* 461:53 — generic early-warning signals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d

from bench.early_warning_domain import DetectorTrajectory, permutation_significance
from bench.early_warning_single_series import (
    DETECTOR,
    DETECTOR_MULTISCALE,
    SingleSeriesObservable,
    calibrate_single_series,
    critical_slowing_down_trajectory,
    evaluate_single_series,
    null_series_trials,
    single_series_verdict,
    slice_series,
)

if TYPE_CHECKING:  # pragma: no cover - import only for static typing
    from collections.abc import Sequence

FloatArray = NDArray[np.float64]

#: R ``ksmooth(kernel = "normal")`` scales its ``bandwidth`` so the kernel's quartiles
#: sit at ±0.25·bandwidth; for a Gaussian that is ``sigma = 0.25 / qnorm(0.75) · bw``.
#: This is the exact factor that turns the toolbox bandwidth into a Gaussian sigma.
KSMOOTH_NORMAL_SCALE = 0.3706506

#: Gaussian detrend bandwidth as a percentage of the record length — the toolbox
#: expresses its Gaussian ``ksmooth`` bandwidth this way.
DETREND_BANDWIDTH_PERCENT = 10.0

#: Analysis window length in samples for the slowing-down sweep.
WINDOW = 30
#: Hop between consecutive windows in samples.
STEP = 3
#: Pre-onset analysis-segment length in samples — the transition window scored for a
#: lead, and the null-trial length.
SEGMENT_SAMPLES = 60
#: Leading baseline fraction of the analysis segment (its stable early half).
SEGMENT_BASELINE_FRACTION = 0.5
#: Target false-alarm rate the detector is calibrated to on the pre-approach null.
TARGET_FALSE_ALARM = 0.10

__all__ = [
    "DETREND_BANDWIDTH_PERCENT",
    "KSMOOTH_NORMAL_SCALE",
    "SEGMENT_BASELINE_FRACTION",
    "SEGMENT_SAMPLES",
    "STEP",
    "TARGET_FALSE_ALARM",
    "WINDOW",
    "ClimateProxyAdapter",
    "ClimateRecord",
    "climate_observable",
    "gaussian_detrend",
    "load_climate_series",
    "main",
]


# --------------------------------------------------------------------------- #
# Observable pipeline (pure — fully exercised on synthetic arrays)             #
# --------------------------------------------------------------------------- #


def gaussian_detrend(
    series: FloatArray, *, bandwidth_percent: float = DETREND_BANDWIDTH_PERCENT
) -> FloatArray:
    """Return the residual after subtracting a Gaussian-``ksmooth`` trend.

    Reproduces the ``earlywarnings`` toolbox's Gaussian detrending: the bandwidth is
    ``round(N · bandwidth_percent / 100)`` samples, and R's ``ksmooth(kernel =
    "normal")`` scales that bandwidth to a Gaussian ``sigma = KSMOOTH_NORMAL_SCALE ·
    bandwidth`` (its kernel quartiles sit at ±0.25·bandwidth). Subtracting the smoothed
    trend leaves the fluctuations whose variance and autocorrelation the slowing-down
    monitor reads.

    Parameters
    ----------
    series : FloatArray
        The equally-sampled proxy series, shape ``(N,)``.
    bandwidth_percent : float
        Gaussian bandwidth as a percentage of ``N``; must be in ``(0, 100]``.

    Returns
    -------
    FloatArray
        The detrended residual, same shape as ``series``.

    Raises
    ------
    ValueError
        If ``series`` is not one-dimensional or ``bandwidth_percent`` is out of range.
    """
    values = np.asarray(series, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("series must be one-dimensional")
    percent = float(bandwidth_percent)
    if not np.isfinite(percent) or percent <= 0.0 or percent > 100.0:
        raise ValueError("bandwidth_percent must lie in (0, 100]")
    n_samples = values.shape[0]
    bandwidth = max(1, round(n_samples * percent / 100.0))
    sigma = KSMOOTH_NORMAL_SCALE * bandwidth
    trend = gaussian_filter1d(values, sigma=sigma, mode="nearest")
    return np.ascontiguousarray(values - trend)


def climate_observable(
    proxy: FloatArray,
    *,
    span_years: float,
    detrend_bandwidth_percent: float = DETREND_BANDWIDTH_PERCENT,
) -> SingleSeriesObservable:
    """Detrend an equidistant proxy series into a single-series observable.

    The climate-specific half of the capstone. The proxy is the toolbox's equidistant
    ``_Y1int`` series — already linearly interpolated to equal spacing over its age span
    — so the only preprocessing here is the Gaussian detrend (:func:`gaussian_detrend`).
    The residual is wrapped as a
    :class:`~bench.early_warning_single_series.SingleSeriesObservable` whose sampling
    rate places ``span_years`` across the record, so a sample lead reads out in years.

    Parameters
    ----------
    proxy : FloatArray
        The equidistant interpolated proxy, shape ``(N,)`` with ``N >= 3``.
    span_years : float
        Calendar-time span of the record in years (youngest to oldest sample); must be
        positive.
    detrend_bandwidth_percent : float
        Gaussian detrend bandwidth as a percentage of ``N``.

    Returns
    -------
    SingleSeriesObservable
        The detrended residual at ``(N - 1) / span_years`` samples per year, so a lead
        of ``k`` samples is ``k · span_years / (N - 1)`` years.

    Raises
    ------
    ValueError
        If ``proxy`` is malformed or ``span_years`` is not positive.
    """
    values = np.asarray(proxy, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("proxy must be one-dimensional")
    if values.shape[0] < 3:
        raise ValueError("proxy must have at least three samples")
    if not np.all(np.isfinite(values)):
        raise ValueError("proxy must be finite")
    span = float(span_years)
    if not np.isfinite(span) or span <= 0.0:
        raise ValueError("span_years must be a positive finite number")
    residual = gaussian_detrend(values, bandwidth_percent=detrend_bandwidth_percent)
    samples_per_year = (values.shape[0] - 1) / span
    return SingleSeriesObservable(series=residual, sampling_rate_hz=samples_per_year)


@dataclass(frozen=True)
class ClimateProxyAdapter:
    """The palaeoclimate bridge from an equidistant proxy record to an observable.

    A ``DomainObservableAdapter`` packaging the Gaussian detrend, so the single-series
    harness screens a palaeoclimate proxy exactly as its multi-node sibling screens
    scalp EEG, ECG or grid signals through their adapters.

    Attributes
    ----------
    detrend_bandwidth_percent : float
        Gaussian detrend bandwidth as a percentage of the record length.
    """

    detrend_bandwidth_percent: float = DETREND_BANDWIDTH_PERCENT

    @property
    def domain(self) -> str:
        """Return the domain label ``palaeoclimate``."""
        return "palaeoclimate"

    def observable(
        self, proxy: FloatArray, *, span_years: float
    ) -> SingleSeriesObservable:
        """Return the detrended single-series observable for one proxy record.

        Parameters
        ----------
        proxy : FloatArray
            The equidistant interpolated proxy, shape ``(N,)``.
        span_years : float
            Calendar-time span of the record in years.

        Returns
        -------
        SingleSeriesObservable
            The detrended residual the single-series harness reads.
        """
        return climate_observable(
            proxy,
            span_years=span_years,
            detrend_bandwidth_percent=self.detrend_bandwidth_percent,
        )


# --------------------------------------------------------------------------- #
# Text ingestion (thin — touches the citation-only corpus, tested on           #
# synthetic text files, never on the redistributed proxy series)               #
# --------------------------------------------------------------------------- #


def load_climate_series(
    proxy_path: str | Path,
    time_path: str | Path,
    *,
    age_unit_years: float,
) -> tuple[FloatArray, float]:
    """Read an equidistant proxy and its age span from the toolbox text files.

    Parameters
    ----------
    proxy_path : str or Path
        Path to the equidistant interpolated proxy file (one value per line — the
        toolbox ``_Y1int``).
    time_path : str or Path
        Path to the age axis file (one age per line — the toolbox ``_Yt``); only its
        range is used, so it need not be the same length as the proxy.
    age_unit_years : float
        Multiplier converting the age axis to years (``1e6`` for ages in Ma, ``1e3`` for
        kyr, ``1`` for years); must be positive.

    Returns
    -------
    tuple[FloatArray, float]
        The proxy values and the record's calendar-time span in years.

    Raises
    ------
    ValueError
        If the proxy is malformed or ``age_unit_years`` is not positive.
    """
    proxy = np.loadtxt(proxy_path, dtype=np.float64)
    age = np.loadtxt(time_path, dtype=np.float64)
    if proxy.ndim != 1 or proxy.shape[0] < 3:
        raise ValueError(f"{proxy_path} must hold at least three proxy values")
    unit = float(age_unit_years)
    if not np.isfinite(unit) or unit <= 0.0:
        raise ValueError("age_unit_years must be a positive finite number")
    span_years = (float(np.max(age)) - float(np.min(age))) * unit
    return np.ascontiguousarray(proxy), span_years


@dataclass(frozen=True)
class ClimateRecord:
    """One abrupt-transition proxy record and how to read it.

    Attributes
    ----------
    record_id : str
        Corpus label, e.g. ``younger_dryas_termination``.
    proxy_file : str
        File name of the equidistant proxy (``_Y1int``), under the data directory.
    time_file : str
        File name of the age axis (``_Yt``), under the data directory.
    age_unit_years : float
        Multiplier converting the age axis to years (``1e6`` for Ma, ``1e3`` for kyr).
    proxy_description : str
        Human-readable description of the proxy, sealed into the record's evidence.
    citation : str
        Original data source citation, carried into the sealed provenance.
    """

    record_id: str
    proxy_file: str
    time_file: str
    age_unit_years: float
    proxy_description: str
    citation: str


#: The eight Dakos et al. 2008 records, from ``earlywarningtoolbox/datasets``.
CLIMATE_RECORDS: tuple[ClimateRecord, ...] = (
    ClimateRecord(
        record_id="eocene_oligocene_greenhouse_end",
        proxy_file="Eo_Gl_Y1int.txt",
        time_file="Eo_Gl_Yt.txt",
        age_unit_years=1.0e6,
        proxy_description=(
            "benthic foraminiferal δ¹⁸O across the Eocene–Oligocene greenhouse-to-"
            "icehouse transition (~34 Ma; detrended residual of the toolbox _Y1int)"
        ),
        citation=(
            "Eocene–Oligocene benthic δ¹⁸O (earlywarningtoolbox/datasets Eo_Gl, via "
            "NOAA WDC Paleoclimatology); analysed in Dakos et al. 2008, PNAS 105:14308"
        ),
    ),
    ClimateRecord(
        record_id="glaciation_I_termination",
        proxy_file="Vostok1deut_Y1int.txt",
        time_file="Vostok1deut_Yt.txt",
        age_unit_years=1.0,
        proxy_description=(
            "Vostok ice-core deuterium (δD) across the termination of glaciation I "
            "(detrended residual of the toolbox _Y1int)"
        ),
        citation=(
            "Vostok δD, glaciation I (earlywarningtoolbox/datasets Vostok1deut; "
            "Petit et al. 1999); analysed in Dakos et al. 2008, PNAS 105:14308"
        ),
    ),
    ClimateRecord(
        record_id="glaciation_II_termination",
        proxy_file="Vostok2deut_Y1int.txt",
        time_file="Vostok2deut_Yt.txt",
        age_unit_years=1.0,
        proxy_description=(
            "Vostok ice-core deuterium (δD) across the termination of glaciation II "
            "(detrended residual of the toolbox _Y1int)"
        ),
        citation=(
            "Vostok δD, glaciation II (earlywarningtoolbox/datasets Vostok2deut; "
            "Petit et al. 1999); analysed in Dakos et al. 2008, PNAS 105:14308"
        ),
    ),
    ClimateRecord(
        record_id="glaciation_III_termination",
        proxy_file="Vostok3deut_Y1int.txt",
        time_file="Vostok3deut_Yt.txt",
        age_unit_years=1.0,
        proxy_description=(
            "Vostok ice-core deuterium (δD) across the termination of glaciation III "
            "(detrended residual of the toolbox _Y1int)"
        ),
        citation=(
            "Vostok δD, glaciation III (earlywarningtoolbox/datasets Vostok3deut; "
            "Petit et al. 1999); analysed in Dakos et al. 2008, PNAS 105:14308"
        ),
    ),
    ClimateRecord(
        record_id="glaciation_IV_termination",
        proxy_file="Vostok4deut_Y1int.txt",
        time_file="Vostok4deut_Yt.txt",
        age_unit_years=1.0,
        proxy_description=(
            "Vostok ice-core deuterium (δD) across the termination of glaciation IV "
            "(detrended residual of the toolbox _Y1int)"
        ),
        citation=(
            "Vostok δD, glaciation IV (earlywarningtoolbox/datasets Vostok4deut; "
            "Petit et al. 1999); analysed in Dakos et al. 2008, PNAS 105:14308"
        ),
    ),
    ClimateRecord(
        record_id="bolling_allerod_onset",
        proxy_file="GBA_temp_Y1int.txt",
        time_file="GBA_temp_Yt.txt",
        age_unit_years=1.0e3,
        proxy_description=(
            "Greenland ice-core δ¹⁸O across the Bølling–Allerød warming onset "
            "(~14.6 ka; detrended residual of the toolbox _Y1int)"
        ),
        citation=(
            "Greenland ice δ¹⁸O, Bølling–Allerød (earlywarningtoolbox/datasets "
            "GBA_temp, via NOAA WDC Paleo); Dakos et al. 2008, PNAS 105:14308"
        ),
    ),
    ClimateRecord(
        record_id="younger_dryas_termination",
        proxy_file="YD2PB_grayscale_Y1int.txt",
        time_file="YD2PB_grayscale_Yt.txt",
        age_unit_years=1.0,
        proxy_description=(
            "Cariaco Basin sediment greyscale reflectance across the Younger Dryas to "
            "Preboreal termination (~11.5 ka; detrended residual of the toolbox _Y1int)"
        ),
        citation=(
            "Cariaco greyscale, Younger Dryas (earlywarningtoolbox/datasets "
            "YD2PB_grayscale; Hughen et al.); Dakos et al. 2008, PNAS 105:14308"
        ),
    ),
    ClimateRecord(
        record_id="north_africa_desertification",
        proxy_file="terrigenous_Y1int.txt",
        time_file="terrigenous_Yt.txt",
        age_unit_years=1.0,
        proxy_description=(
            "ODP 658C terrigenous dust percentage across the North-African "
            "desertification (~5.5 ka; detrended residual of the toolbox _Y1int)"
        ),
        citation=(
            "ODP 658C terrigenous dust (earlywarningtoolbox/datasets terrigenous; "
            "deMenocal et al. 2000, LDEO); Dakos et al. 2008, PNAS 105:14308"
        ),
    ),
)


# --------------------------------------------------------------------------- #
# Orchestration (I/O shell over the tested logic)                             #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _RecordSegments:
    """One record's detrended observable split into transition and null regions."""

    record: ClimateRecord
    transition_segment: SingleSeriesObservable
    null_region: SingleSeriesObservable


def _segment_record(
    record: ClimateRecord,
    adapter: ClimateProxyAdapter,
    data: Path,
    *,
    segment_samples: int,
) -> _RecordSegments | None:
    """Split a record into a pre-onset transition segment and a stable null region.

    The record's final (youngest) sample is the transition onset, so the pre-onset
    segment is its last ``segment_samples`` and the null region is everything before it.
    Returns ``None`` (reported, not counted) when the record is too short to yield both
    a full segment and at least one preceding null trial.
    """
    proxy, span_years = load_climate_series(
        data / record.proxy_file,
        data / record.time_file,
        age_unit_years=record.age_unit_years,
    )
    observable = adapter.observable(proxy, span_years=span_years)
    n_samples = observable.n_samples
    transition_start = n_samples - segment_samples
    if transition_start < segment_samples:
        return None
    return _RecordSegments(
        record=record,
        transition_segment=slice_series(
            observable, start=transition_start, stop=n_samples
        ),
        null_region=slice_series(observable, start=0, stop=transition_start),
    )


def main(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    records: Sequence[ClimateRecord] = CLIMATE_RECORDS,
    segment_samples: int = SEGMENT_SAMPLES,
    baseline_fraction: float = SEGMENT_BASELINE_FRACTION,
    multiscale: bool = False,
) -> None:
    """Run the capstone over the palaeoclimate corpus and write the sealed artefacts.

    Reads each proxy record from ``data_dir`` (raw toolbox text, citation-only), splits
    it into a pre-onset transition segment and a stable pre-approach null region,
    calibrates the slowing-down detector to a matched false-alarm rate on the pooled
    null trials, seals each record's alarm for its transition, and writes the sealed
    records plus an aggregate results JSON to ``output_dir``. Only the derived, sealed
    artefacts are written; the raw proxy series is never copied.

    Parameters
    ----------
    data_dir : str or Path
        Directory holding the ``_Y1int``/``_Yt`` text files named in ``records``.
    output_dir : str or Path
        Directory the sealed derived artefacts are written to.
    records : sequence of ClimateRecord
        The corpus to evaluate.
    segment_samples : int
        Pre-onset analysis-segment length in samples; also the null trial length.
    baseline_fraction : float
        Leading baseline fraction of each segment.
    multiscale : bool
        If True, evaluate the multi-scale CSD detector.
    """
    data = Path(data_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    adapter = ClimateProxyAdapter()

    segmented: list[_RecordSegments] = []
    excluded: list[dict[str, object]] = []
    for record in records:
        segments = _segment_record(
            record, adapter, data, segment_samples=segment_samples
        )
        if segments is None:
            excluded.append(
                {"record_id": record.record_id, "reason": "record too short"}
            )
            continue
        segmented.append(segments)

    null_trials = null_series_trials(
        [segments.null_region for segments in segmented],
        segment_samples=segment_samples,
    )
    calibration = calibrate_single_series(
        null_trials,
        target_fa=TARGET_FALSE_ALARM,
        window=WINDOW,
        step=STEP,
        baseline_fraction=baseline_fraction,
        multiscale=multiscale,
    )
    detector = DETECTOR_MULTISCALE if multiscale else DETECTOR
    threshold = calibration.thresholds[detector]

    leads: dict[str, float | None] = {}
    transition_records: list[dict[str, object]] = []
    for segments in segmented:
        record = segments.record
        result = evaluate_single_series(
            segments.transition_segment,
            record_id=record.record_id,
            onset_sample=segment_samples,
            signal_source=(
                f"{record.citation} / {segment_samples}-sample pre-onset segment "
                "(detrended residual)"
            ),
            captured_at=f"Dakos2008/{record.record_id}",
            threshold=threshold,
            observable_description=record.proxy_description,
            window=WINDOW,
            step=STEP,
            baseline_fraction=baseline_fraction,
            multiscale=multiscale,
        )
        (out / f"{record.record_id}_early_warning_evidence.json").write_text(
            json.dumps(result.to_audit_record(), indent=2) + "\n", encoding="utf-8"
        )
        leads[record.record_id] = result.lead_seconds()
        transition_records.append(
            {
                "record_id": record.record_id,
                "lead_years": result.lead_seconds(),
            }
        )

    def _trajectory(observable: SingleSeriesObservable) -> DetectorTrajectory:
        return critical_slowing_down_trajectory(
            observable,
            window=WINDOW,
            step=STEP,
            baseline_fraction=baseline_fraction,
            multiscale=multiscale,
        )

    significance = permutation_significance(
        [_trajectory(segments.transition_segment) for segments in segmented],
        [_trajectory(trial) for trial in null_trials],
        threshold=threshold,
    )

    result_filename = (
        "early_warning_leadtime_climate_multiscale_results.json"
        if multiscale
        else "early_warning_leadtime_climate_results.json"
    )
    payload = {
        "benchmark": "early_warning_leadtime_climate",
        "corpus": "Dakos et al. 2008 palaeoclimate abrupt-transition records",
        "multiscale": multiscale,
        "detrend_bandwidth_percent": DETREND_BANDWIDTH_PERCENT,
        "window": WINDOW,
        "step": STEP,
        "segment_samples": segment_samples,
        "baseline_fraction": baseline_fraction,
        "target_false_alarm": TARGET_FALSE_ALARM,
        "n_null_trials": len(null_trials),
        "matched_false_alarm_threshold": threshold,
        "achieved_false_alarm": calibration.achieved_false_alarm,
        "transitions": transition_records,
        "excluded_records": excluded,
        "permutation_significance": significance.to_audit_record(),
        "verdict": single_series_verdict(
            leads,
            len(transition_records),
            noun="climate transitions",
            singular="climate transition",
            lead_unit="yr",
        ),
    }
    (out / result_filename).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(payload["verdict"])
    print(
        f"permutation p-value {significance.p_value:.3f} "
        f"({significance.observed_led}/{significance.n_transitions} led vs "
        f"{significance.expected_led:.2f} expected by chance)"
    )
    print(
        f"{len(null_trials)} null trials; {len(excluded)} records excluded (too short)"
    )
    print(f"results written to {out}")


if __name__ == "__main__":  # pragma: no cover - CLI shell over the tested logic
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_dir", help="directory holding the toolbox proxy text files"
    )
    parser.add_argument("output_dir", help="directory for the sealed derived output")
    parser.add_argument(
        "--multiscale",
        action="store_true",
        help="evaluate the multi-scale CSD detector",
    )
    arguments = parser.parse_args()
    main(arguments.data_dir, arguments.output_dir, multiscale=arguments.multiscale)
