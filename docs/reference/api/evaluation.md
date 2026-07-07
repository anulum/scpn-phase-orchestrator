# Honest Early-Warning Auditor

`scpn_phase_orchestrator.evaluation` scores whether an early-warning detector has
**real skill**, not just a high detection rate. A detector that alarms on 90 % of
pre-transition events is worthless if it also alarms on 90 % of transition-free
nulls; the honest question is whether it alarms on events *more often than its own
false-alarm rate explains*. This subsystem answers that question for **any**
detector — the SCPN suite, an AR(1)/Kendall-τ trend baseline, or a black-box deep
classifier — because it reads only the per-segment score each one emits.

Why this exists: the early-warning-signals literature has repeatedly found that
most indicators sit at chance once a matched false-alarm rate and a null model are
imposed (Boettiger & Hastings 2012 asked for exactly this discipline; O'Brien &
Clements 2023 confirmed the null result on real data). A packaged, detector-neutral
harness that enforces that discipline — and seals the verdict — did not exist. That
harness is the asset here, above any one detector.

## What an audit reports

`audit_detector` takes two arrays of per-segment scores — one on genuine
pre-transition **event** segments, one on transition-free **null** segments — and
returns a [`DetectorAudit`][scpn_phase_orchestrator.evaluation.auditor.DetectorAudit]:

- **`matched_threshold`** — the alarm threshold calibrated *from the null scores*
  so at most a target fraction of nulls alarm. It is placed just above an order
  statistic of the nulls, never on a fixed grid, so a detector whose nulls need a
  high gate is not silently clipped. `-inf` means the target permitted every null
  to alarm (the gate is fully open).
- **`achieved_false_alarm`** — the false-alarm rate the threshold actually held,
  reported alongside the target so a detector held below (or, in a degenerate
  corpus, unable to hold) the target is visible rather than assumed.
- **`detection_rate`** — the fraction of event segments that alarmed at that
  threshold.
- **`p_value`** (the headline) — the one-sided **label-permutation** p-value: under
  the exchangeability null that events and nulls are interchangeable, the fraction
  of random relabellings whose event-slot alarm count reaches the observed count,
  add-one corrected so it is never zero. A small p-value means the events alarmed
  more than the matched false alarm on this corpus.

A convention flag `beats_chance` (`p_value < alpha`) is offered for convenience;
the p-value itself is the honest quantity and is always reported. Higher score
means more evidence of a transition — orient a falling statistic by negating it.

Convention `alpha` defaults to `0.05`. `audit_scoring_detector` is the convenience
wrapper for a detector still expressed as a callable on a raw window: it applies
the scoring function to each event and null series, then defers to `audit_detector`.

## A worked audit distinguishes skill from no skill

`bench/honest_auditor_worked_example.py` audits two detectors on one synthetic
corpus of AR(1) event windows (rising autocorrelation — the textbook
critical-slowing-down signature) and white-noise nulls. Both arms are zero-mean
with the same marginal variance, so the window mean is genuinely uninformative and
serves as an honest negative control:

```console
$ python -m bench.honest_auditor_worked_example
lag1-autocorrelation   target_fa=0.10 achieved_fa=0.100 detect=1.000 p=9.999e-05 beats_chance=True  hash=d97f55721c09
window-mean-control    target_fa=0.10 achieved_fa=0.050 detect=0.050 p=0.6215    beats_chance=False hash=52242831ed23
```

The lag-1 autocorrelation detector — a real critical-slowing signal — beats chance
decisively; the window-mean control lands at chance. Both verdicts are read from
scores alone, so a competitor's classifier would be judged on identical footing.

## A head-to-head between two real detectors

`bench/auditor_detector_head_to_head.py` goes further: it audits the *real* SCPN
modal envelope-growth detector
([`modal_growth_score`](monitor_grid_modal_growth.md)) against a published
competitor — the Dakos et al. 2008 AR(1)/Kendall-τ rising-autocorrelation trend —
on two synthetic-but-honest regimes (a growing oscillatory mode, and a monotone
rising-autocorrelation slowdown). With clearly skilful detectors the permutation
p-value saturates (both beat chance), so the auditor's **detection rate at the
matched false alarm** is the discriminator, and it is regime-dependent:

```console
$ python -m bench.auditor_detector_head_to_head
oscillatory  scpn-modal-growth    achieved_fa=0.100 detect=1.000 p=9.999e-05 beats_chance=True
oscillatory  ar1-kendall-tau      achieved_fa=0.100 detect=0.775 p=9.999e-05 beats_chance=True
monotone     scpn-modal-growth    achieved_fa=0.100 detect=0.550 p=9.999e-05 beats_chance=True
monotone     ar1-kendall-tau      achieved_fa=0.100 detect=0.775 p=9.999e-05 beats_chance=True
```

The envelope-growth detector leads on the oscillatory regime, the AR(1) competitor
leads on the monotone one — the eigenvalue-regime-map finding, adjudicated without
bias by one matched-false-alarm + permutation test. Real field data would replace
the synthetic corpora without changing the auditor. This is the integration proof:
the productised auditor plugs into actual detector code, not just toy scorers.

## Sealing an audit

`seal_detector_audit` binds a verdict to its corpus provenance — an identifier and
a caller-supplied capture timestamp — under a SHA-256 over its canonical JSON,
reusing the same hashing path as the [assurance bundle](assurance.md). Recomputing
the hash from the recorded fields detects any later edit, so a published audit
verdict cannot be quietly altered. A fully open (`-inf`) threshold is serialised to
the string `"-inf"` so the record stays strict JSON. The sealed record carries an
explicit disclaimer: an audit measures skill on *the supplied corpus only* and is
not a certification of field performance.

## Auditing from the command line

`spo audit-detector` runs the same audit without writing Python, so a detector's
skill can be checked from a scores file. The file is a JSON object with
`event_scores` and `null_scores` arrays of per-segment scores (higher means more
evidence of a transition) and an optional `detector_name`:

```bash
spo audit-detector scores.json \
  --target-false-alarm 0.10 \
  --corpus-id grid-2026 \
  --captured-at 2026-07-07T15:00:00+02:00
```

Without `--corpus-id`/`--captured-at` the command prints the bare verdict; supply
both (they must be given together) to seal it into a hash-addressed record.
`--output` also writes the JSON to a file. Score entries must be finite numbers —
a missing key, an empty list, or a non-numeric or non-finite entry is an error,
never a silently dropped score. The command reads a local file and prints JSON; it
never actuates, signs, or reaches the network.

## Skill primitives

For callers composing their own harness, the detector-agnostic primitives are
public: `calibrate_score_threshold` (matched-false-alarm calibration),
`matched_false_alarm_rate`, `permutation_significance_from_alarms` (the
exchangeability test), and `surrogate_rank_pvalue` (the single-statistic
counterpart against a surrogate ensemble).

::: scpn_phase_orchestrator.evaluation.auditor

::: scpn_phase_orchestrator.evaluation.skill

::: scpn_phase_orchestrator.evaluation.record
