# ISO-NE case 1 — real forced-oscillation PRC evidence

This directory holds the first **non-synthetic** sealed artefact the shipped
SCPN Phase Orchestrator chain produced: a documented real ISO-NE forced
oscillation (near 0.27 Hz) screened through `spo pmu-ieee-adapt` and then
`spo pmu-ringdown`. Everything the pipeline emitted for this run — the estimated
modes, their damping, the flag decision, and the provenance digests — is sealed
in `pmu_ringdown_prc_evidence.json` and guarded by
`tests/test_iso_ne_case1_real_evidence.py`.

It exists so an outsider can inspect a real output without taking our word for it,
and to record honestly what the review-only screener does and does **not** claim.

## The source data (not included here)

The raw phasor-measurement capture is **citation-only** and is **not
redistributed** in this repository. Obtain it directly from the dataset authors:

- Dataset: *A test cases library for methods locating the sources of sustained
  oscillations* — the University of Tennessee, Knoxville oscillation test-case
  library.
- Page: <https://web.eecs.utk.edu/~kaisun/Oscillation/actualcases.html>
  (download `ISO-NE-case1.zip`, which contains `ISO-NE_case1.csv`).
- Cite as requested on the dataset page: S. Maslennikov, B. Wang, Q. Zhang,
  F. Ma, X. Luo, K. Sun, E. Litvinov, *A test cases library for methods locating
  the sources of sustained oscillations*, IEEE PES General Meeting, 2016.

The raw `ISO-NE_case1.csv` we processed had SHA-256
`ca5001bb64cfecced20ea71a6a007a5db8ad96acdcfa13cb021358f0f2575de0`; verifying
this digest confirms you have the identical source file.

It is an IEEE-format multi-header phasor concentrator export: a label row, a
quantity-type row (`T` for time, `F` for frequency, `VM`/`VA`/`IM`/`IA` for the
phasor channels), a unit row, a secondary-label row, then one time column and
five channels per unit — 35 frequency channels over 5400 samples at ~30 Hz.

## Reproducing the sealed evidence

With the raw CSV in the working directory, the two shipped commands regenerate
the committed artefact byte for byte (both stages are deterministic):

```bash
# 1. Adapt the IEEE multi-header export to the screener's two-column input.
#    Selects the dropout-free in-band channel with the largest swing: Sub:9:Ln:20.
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

The derived single-channel series has SHA-256
`2ed93167ced93d75d61fef5dc9fb7a878ceb631d74a76fa352e046f64e32f915`, recorded as
`source_sha256` in the evidence. The sealed record's `content_hash` is
`44b2aa69b5575cfb84dc5b18741a55af904c04a07358c28256623b09c6bd6b2f`.

The `captured_at` value is the raw file's own placeholder epoch — the public
dataset anonymises the physical capture instant to `1900-01-01`, and that
placeholder is carried forward rather than invented.

The screener decimates the 5400-sample, ~30 Hz capture to 5 Hz (900 samples)
before the matrix-pencil estimator, and mean-detrends the frequency deviation to
remove the operating-point offset. Both are necessary on this real capture: the
full rate exceeds the estimator's practical size, and the un-detrended offset
otherwise dominates the estimate as a spurious 0 Hz mode.

## What the screener found

| Mode | Frequency (Hz) | Damping ratio | Family | Flagged |
|------|----------------|---------------|--------|---------|
| 1 | 0.0000 | +1.0000 | aperiodic | no |
| 2 | 0.2753 | −0.0028 | inter-area | **yes** (undamped) |
| 3 | 0.2851 | +0.0026 | inter-area | **yes** (poorly damped) |
| 4 | 0.0196 | +0.0543 | inter-area | no |
| 5 | 0.0000 | −1.0000 | aperiodic | **yes** (undamped) |

Verdict: `flagged_for_review`, `flagged_count = 3`. The two inter-area modes at
0.2753 / 0.2851 Hz are the documented ~0.27 Hz forced oscillation, recovered with
essentially zero damping (a sustained, not decaying, oscillation) and flagged.

## Scope and limits (what this is not)

- **Review-only, offline.** The claim boundary is
  `review_only_offline_no_live_actuation`. This screens measured modal damping
  for NERC PRC-028-1 / PRC-030-1 review workflows; it is not a conformity
  assessment, a certification of compliance, or legal advice, and it never
  actuates. Conformance is for a qualified assessor to determine.
- **A single-evidence artefact, not a full assessor bundle.** The three-role
  `spo power-grid-prc-bundle` binds dVOC damping, PMU ringdown, and IBR
  ride-through evidence. This dataset is a frequency ringdown only, so a real
  companion dVOC or IBR record does not exist for it; padding the bundle with
  synthetic companions would misrepresent it, so the honest artefact here is the
  PMU ringdown evidence on its own.
- **One capture.** It shows the shipped chain recovers a documented real
  oscillation; it is not a statistical evaluation across many events.
