<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# Real-Data Case Study: ISO-NE Forced Oscillation

This page runs the shipped SCPN Phase Orchestrator screening chain on a **real,
public, documented** power-system disturbance and shows exactly what it produced.
It is the first entry in an external-validation track whose premise is that
evidence on real external data counts for more than any number of synthetic
fixtures: a documented outcome an outsider can reproduce and check, not a claim.

The sealed output, its provenance, and an integrity test are committed under
`examples/real_data/iso_ne_case1/` so the result is inspectable directly from the
repository.

## The disturbance

The capture is case 1 of the University of Tennessee, Knoxville forced-oscillation
test-case library — an ISO New England recording with a documented sustained
oscillation near 0.27 Hz. It is an IEEE-format multi-header phasor concentrator
export: a label row, a quantity-type row (`T` for time, `F` for frequency,
`VM`/`VA`/`IM`/`IA` for the phasor channels), a unit row, a secondary-label row,
then one time column and five channels per unit — 35 frequency channels over
5400 samples at roughly 30 Hz.

The raw capture is **citation-only and is not redistributed here.** Obtain it from
the dataset page (<https://web.eecs.utk.edu/~kaisun/Oscillation/actualcases.html>,
`ISO-NE-case1.zip`) and cite S. Maslennikov, B. Wang, Q. Zhang, F. Ma, X. Luo,
K. Sun, E. Litvinov, *A test cases library for methods locating the sources of
sustained oscillations*, IEEE PES General Meeting, 2016. The processed
`ISO-NE_case1.csv` had SHA-256
`ca5001bb64cfecced20ea71a6a007a5db8ad96acdcfa13cb021358f0f2575de0`.

## The pipeline

Two shipped commands take the raw multi-header export to sealed review evidence.
Both stages are deterministic, so the committed artefact reproduces byte for byte.

```bash
# 1. Adapt the IEEE multi-header export to the screener's two-column input.
#    Selects the dropout-free, in-band frequency channel with the largest swing.
spo pmu-ieee-adapt ISO-NE_case1.csv iso_ne_case1_frequency_Sub9Ln20.csv \
  --nominal-frequency-hz 60.0 --plausible-band-hz 2.0

# 2. Screen the derived series for PRC oscillation review evidence.
spo pmu-ringdown iso_ne_case1_frequency_Sub9Ln20.csv \
  --event-id iso-ne-forced-oscillation-case1 \
  --captured-at 1900-01-01T00:00:00Z \
  --signal-source "ISO-NE Oscillation dataset case1 (UTK) / Sub:9:Ln:20 frequency" \
  --nominal-frequency-hz 60.0 --detrend mean --analysis-rate-hz 5.0 \
  --output pmu_ringdown_prc_evidence.json
```

The adapter selects channel `Sub:9:Ln:20`. The screener mean-detrends the
frequency deviation (removing the operating-point offset that would otherwise be
fit as a spurious 0 Hz mode) and decimates the 5400-sample, ~30 Hz capture to
5 Hz (900 samples) before the matrix-pencil estimator — both are necessary on
this real capture, where the full rate exceeds the estimator's practical size and
the un-detrended offset dominates.

## What it found

| Mode | Frequency (Hz) | Damping ratio | Family | Flagged |
|------|----------------|---------------|--------|---------|
| 1 | 0.0000 | +1.0000 | aperiodic | no |
| 2 | 0.2753 | −0.0028 | inter-area | **yes** (undamped) |
| 3 | 0.2851 | +0.0026 | inter-area | **yes** (poorly damped) |
| 4 | 0.0196 | +0.0543 | inter-area | no |
| 5 | 0.0000 | −1.0000 | aperiodic | **yes** (undamped) |

The two inter-area modes at 0.2753 / 0.2851 Hz are the documented ~0.27 Hz forced
oscillation, recovered with essentially zero damping — a sustained, not decaying,
oscillation — and flagged. Verdict: `flagged_for_review`, `flagged_count = 3`.

This result is not free: the shipped screener originally missed this mode despite
passing its whole synthetic suite at full coverage, and recovering it required
fixing three real-data defects (operating-point offset, decimal-rounded
timestamps, and estimator sizing). That the failure was invisible to synthetic
fixtures is the point of validating on real data.

## Provenance and integrity

The sealed `pmu_ringdown_prc_evidence.json` records only analytical results —
the modal estimates, their damping, the flag decision — and provenance digests;
it contains no raw time series, so it is a research result about the data rather
than a redistribution of it. The derived single-channel series has SHA-256
`2ed93167ced93d75d61fef5dc9fb7a878ceb631d74a76fa352e046f64e32f915`
(recorded as `source_sha256`), and the sealed record's `content_hash` is
`44b2aa69b5575cfb84dc5b18741a55af904c04a07358c28256623b09c6bd6b2f`.

`tests/test_iso_ne_case1_real_evidence.py` guards the committed artefact without
the raw data: it recomputes the top-level and nested content seals, pins the
derived-series digest, and asserts the documented inter-area mode is flagged.

## Scope and limits

- **Review-only, offline.** The claim boundary is
  `review_only_offline_no_live_actuation`. The chain screens measured modal
  damping for NERC PRC-028-1 / PRC-030-1 review workflows; it is not a conformity
  assessment, a certification of compliance, or legal advice, and it never
  actuates. Conformance is for a qualified assessor to determine.
- **A single-evidence artefact, not a full assessor bundle.** The three-role
  `spo power-grid-prc-bundle` binds dVOC damping, PMU ringdown, and IBR
  ride-through evidence. This capture is a frequency ringdown only, so no real
  companion dVOC or IBR record exists for it; the honest artefact is the PMU
  ringdown evidence on its own rather than a bundle padded with synthetic
  companions.
- **One event.** It shows the shipped chain recovers a documented real
  oscillation; it is not a statistical evaluation across many disturbances.
