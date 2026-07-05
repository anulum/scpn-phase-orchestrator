# Dakos 2008 palaeoclimate — real single-series early-warning evidence

This directory holds the **fourth domain** the SCPN Phase Orchestrator's early-warning
design was proven on, and the **first single-series** one. The scalp-EEG, cardiac-ECG
and grid-PMU capstones screen a *population* of coupled oscillators; a palaeoclimate
proxy record is one scalar time-series approaching an abrupt transition, so rising
synchronisation and ordinal-transition entropy do not apply. The **critical
slowing-down** monitor does — the rising variance and lag-one autocorrelation ahead of
a bifurcation — and it is screened through the single-series harness, with each alarm
or silence sealed into a hash-addressed, claim-bounded `EarlyWarningEvidence` record.

It exists to show the design extends past coupled-oscillator domains to the classic
single-series tipping-point setting of Dakos et al. 2008, and to record plainly what
the commodity detector does — and does **not** — show on those records.

## The result, stated honestly

At a matched false-alarm rate (≤ 10 % over 50 pre-approach null trials pooled across the
corpus), **critical slowing down leads one of the six evaluated transitions**:

| Record | Transition | Led | Lead |
|--------|-----------|:---:|-----:|
| eocene_oligocene_greenhouse_end | Eocene–Oligocene greenhouse end (~34 Ma) | ✓ | ≈ 722 kyr |
| glaciation_I_termination | End of glaciation I | — | — |
| glaciation_II_termination | End of glaciation II | — | — |
| glaciation_III_termination | End of glaciation III | — | — |
| bolling_allerod_onset | Bølling–Allerød onset (~14.6 ka) | — | — |
| younger_dryas_termination | Younger Dryas termination (~11.5 ka) | — | — |

Two further records — the glaciation IV termination and the North-African
desertification — are **excluded**, not counted as silent misses: their interpolated
series (100 and 30 samples) are too short to yield both a 60-sample pre-onset segment
and a preceding null trial.

Only the longest record, the Eocene–Oligocene greenhouse end, is led; the other five —
including the Younger Dryas termination — are **sealed silences**. A lead on one record
is evidence, not a robust precursor. The deliverable is the **auditable, reproducible,
claim-bounded sealed evidence** — including the sealed silences — not the lead.

### Two honest caveats specific to this domain

- **A different statistic from Dakos.** Dakos et al. 2008 read a *rising Kendall-τ
  trend* of the lag-one autocorrelation and test it against phase-randomised surrogates,
  and report a positive early-warning signal on these records. The SCPN monitor instead
  raises a robust (median/MAD) z-score alarm of variance-or-lag-one against a leading
  baseline **at a matched false-alarm operating point**. This capstone measures what the
  *shipped SCPN detector* does on the Dakos records; a silence here is **not** a
  refutation of the Dakos AR(1)-trend finding — it is a different detector, statistic and
  operating point.
- **A within-record null.** With one series per transition there is no separate
  no-transition recording, so the matched-false-alarm null is the stable pre-approach
  interval of each record, pooled across the corpus. That interval carries the record's
  own baseline variability, making the calibration conservative — reported as-is.

## The source data (not included here)

The raw proxy series are **citation-only** and are **not redistributed** in this
repository. Obtain them from the authors' own Early-Warning-Signals Toolbox dataset
repository:

- Repository: `earlywarningtoolbox/datasets`
  (<https://github.com/earlywarningtoolbox/datasets>), which states it holds the
  "climate data as used in Dakos et al PNAS 2008". Each record ships as
  `<name>_Y1int.txt` (the equidistant interpolated proxy the paper's analysis reads) and
  `<name>_Yt.txt` (the age axis).
- Cite the study and the underlying archives:
  - V. Dakos, M. Scheffer, E. H. van Nes, V. Brovkin, V. Petoukhov, H. Held, *Slowing
    down as an early warning signal for abrupt climate change*, PNAS 105(38):14308–14312,
    2008.
  - Underlying sources (per the paper's Data Sources): NOAA World Data Center for
    Paleoclimatology (benthic δ¹⁸O, Vostok deuterium — Petit et al. 1999, Greenland
    δ¹⁸O, Cariaco greyscale — Hughen et al.) and LDEO Columbia (ODP 658C terrigenous
    dust — deMenocal et al. 2000).

The evaluated records are `Eo_Gl`, `Vostok1deut`, `Vostok2deut`, `Vostok3deut`,
`GBA_temp`, `YD2PB_grayscale`; `Vostok4deut` and `terrigenous` are excluded as too
short. Place the `*_Y1int.txt` / `*_Yt.txt` files in one directory.

## Methodology

- **Observable pipeline.** Each record is the toolbox's equidistant `_Y1int` proxy —
  already linearly interpolated to equal spacing over its age span, the series the
  paper's analysis reads — Gaussian-detrended the way the `earlywarnings` toolbox
  detrends: the bandwidth is `round(N × 10 / 100)` samples and R's
  `ksmooth(kernel = "normal")` scaling turns it into a Gaussian
  `sigma = 0.3706506 × bandwidth`. The detrended residual is the single observable the
  critical-slowing-down monitor reads. Window 30, step 3 samples.
- **Fixed pre-onset segment.** The record's curated interval ends at the abrupt
  transition, so each transition is scored on the last 60 samples: a 30-sample leading
  baseline then a 30-sample detection horizon, with the onset at the final sample. Any
  alarm is a genuine pre-onset lead, read out in years through the record's own
  sample-to-year spacing.
- **Matched false alarm.** The stable pre-approach interval of each record (everything
  before its pre-onset segment) is cut into non-overlapping 60-sample null trials (50
  trials pooled across the corpus); the detector's threshold is set continuously to the
  tightest value holding the trial false-alarm rate at or below 10 % (the quantile of the
  null alarm scores, no grid ceiling). The calibrated robust-z threshold was ≈ 7.9 and
  the achieved false-alarm rate is recorded in the aggregate.
- **Sealing.** Each evaluated transition yields one `EarlyWarningEvidence` record,
  including a sealed silence where the detector did not fire. Each record's
  `content_hash` is a canonical-JSON SHA-256, the same seal the assurance-case bundle and
  the other real-data capstones use.

## Reproducing the sealed evidence

With the toolbox text files in a directory `DATA/`, the shipped module regenerates the
committed artefact (the pipeline and detector are deterministic — no randomness — so a
fresh run reproduces the sealed records byte for byte):

```bash
python bench/early_warning_leadtime_climate.py DATA OUT
```

`OUT/` then contains the six `<record>_early_warning_evidence.json` records and
`early_warning_leadtime_climate_results.json`, identical to the files committed here. The
raw proxy series is read but never copied; only the derived sealed records are.

## Scope and limits (what this is not)

- **Review-only, offline.** The `EarlyWarningEvidence` disclaimer applies: this is a
  technical evidence-mapping artefact, not a climatological, operational, or safety
  decision, nor a certification. It never actuates.
- **Six transitions, one series each.** It shows the shipped critical-slowing-down
  detector on six Dakos records through the single-series harness; it is not a
  reproduction of the Dakos AR(1)-trend analysis, nor a palaeoclimate-prediction
  benchmark.
- **Sparse detection.** One of six evaluated transitions is led at matched false alarm.
  The honest reading is that detection is a commodity, which is exactly why the auditable
  sealed evidence — including the sealed silences — is the deliverable, and why the same
  conclusion holds across four independent domains.
