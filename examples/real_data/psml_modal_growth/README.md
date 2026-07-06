# PSML grid modal growth — the flagship domain-specific head-to-head

This directory holds the programme's **flagship** result: on the real PSML 23-bus
power-system corpus, a **domain-specific** early-warning detector that clears the bar
where the generic suite is at chance, certified by the same matched-false-alarm moat.

The four-domain study showed the generic early-warning family (critical slowing down,
synchronisation, ordinal entropy, their fusion) at chance at a matched false alarm, and
the seizure-specific spectral detector (`../chb01_seizures/`, `bench/seizure_detector.py`)
confirmed a domain-specific detector does **not** automatically win — on a murky preictal
scalp EEG it too is at chance. This is the counterpoint. A growing power-grid oscillation
is a *deterministic, detectable* instability signature: when a disturbance leaves a mode
under-damped, the cross-bus voltage-deviation envelope grows exponentially, and the real
part `σ` of the mode's eigenvalue — the canonical wide-area-monitoring early-warning
quantity — is positive. The detector estimates it directly and, at its validated operating
point, leads far more instability transitions than any generic member.

## The comparison is fair and non-circular

* **Identical split.** Both detectors read the *same* two-second pre-onset voltage
  segments of the *same* scenarios, calibrated to the *same* 10 % matched false alarm on
  the *same* damped null segments. Only the detector differs — the modal detector reads
  the raw cross-bus voltage deviation, the generic suite reads the cross-bus Kuramoto
  observables — never the data or the operating point.
* **Non-circular labels.** A transition is any **generator-trip** scenario, a null any
  **damped bus-fault or branch-trip** scenario: the label is the *disturbance type*, a
  physical annotation independent of the growth statistic the modal detector measures. A
  growing-oscillation detector cannot be scored against a label defined by growth.
* **Disclosed data-quality gate.** Three scenarios carry a non-monotonic time column (a
  non-physical, negative inferred sampling rate) and are dropped; the count is recorded in
  the sealed payload, never silently.

## The result, stated honestly

On the clean 90-transition / 88-null disturbance-type split at a matched 10 % false alarm
(achieved 9.1 %), each pre-onset segment is reduced to the growth rate `σ` of its cross-bus
deviation envelope and tested by the shared label-permutation core:

| Detector | Transitions led | Permutation p |
|----------|:---------------:|:-------------:|
| generic `critical_slowing_down` | 13 / 90 | 0.185 |
| generic `ensemble_weighted` | 11 / 90 | 0.324 |
| generic `transition_entropy` | 8 / 90 | 0.629 |
| generic `synchronisation` | 3 / 90 | 0.976 |
| **modal envelope-growth (focal, recency-weighted)** | **36 / 90** | **0.0001** |

The domain-specific modal detector leads **36 of 90** transitions (40 %) at p = 0.0001,
beating every generic member — the best of which (critical slowing down, 13/90, p = 0.185)
is at chance. This is the flagship claim: the framework, the honest matched-false-alarm
moat, **and** the domain-specific detector that clears the bar where generic early warning
cannot.

## The operating point was validated, not tuned to the answer

The `"focal"` aggregation (the most unstable bus's growth rate, un-diluted) and the
recency weighting (later samples, nearer the disturbance where instability has accelerated,
count for more) are physically motivated and were chosen on an **even-index development
half** and validated on the **odd-index held-out half** before this full-corpus run. On
the held-out half the detector leads **24 / 45** transitions (53 %) at p = 0.0002 — the
unbiased estimate, sealed in the artefact as `held_out_validation`. Every variant tried was
disclosed: the whole-network unweighted growth rate (13/90 full corpus), matrix-pencil
modal damping (global/PCA-dominant-mode), and a per-bus Hilbert-envelope slope were all at
or near chance on the short two-second windows, so the direct recency-weighted per-bus
log-envelope slope is the retained detector. A recency-ratio sensitivity sweep (2–8) beats
the unweighted per-bus detector across the whole range, so the choice is robust.

## Reproduce

```bash
PYTHONPATH=src:. python -m bench.grid_modal_head_to_head \
    <dir-with-PSML-scenarios> examples/real_data/psml_modal_growth
```

where `<dir-with-PSML-scenarios>` holds the citation-only PSML scenario folders (each with
`trans.csv` and `info.csv`). The pipeline is deterministic (seed 0), so the sealed
`content_hash`
(`bc6895879088b31b763c566aa315f0edcfc842f537e81102cbd5706cf3ef7bf2`) is reproduced. The
integrity test `tests/test_psml_modal_head_to_head.py` recomputes the hash **from the
committed payload alone** (no raw re-run, so no cross-platform float drift) and pins it. No
raw PSML data is committed.

## The streaming operating point — the honest live-deployment reality

The head-to-head above is a *per-window* result: given the pre-onset window, the detector
leads 40% of transitions at a matched 10% per-window false alarm. Deploying it on a **live
stream** is a stricter problem, and `grid_modal_stream_operating_point.json` measures how
much stricter, honestly, on the same corpus.

The gap is physical: a damped bus-fault or branch-trip produces a **short transient growth**
window that the continuous monitor scores and alarms on, so scoring every window over a
whole stream compounds the false alarm far above the per-window rate. At the certified
per-window threshold the stream is useless — it leads 82/90 transitions but false-alarms on
**73%** of damped scenarios. Recalibrating for the stream (a persistence debounce plus an
**exponential-fit-quality gate** that rejects a fault's step-like transient — a fault fits an
exponential poorly, a genuine instability well) recovers a usable operating point:

| Operating point | Leads | False alarm |
|-----------------|:-----:|:-----------:|
| offline per-window (the head-to-head above) | 36/90 (40%) | 9% per-window |
| naive stream at the per-window threshold | 82/90 | **73% stream** |
| **streaming, fit-quality-gated, matched stream FA** | **11/45 held-out (24%)** | **10% stream** |

At a matched stream false alarm the fit-quality gate (window 2 s, persistence 2) leads
**24%** of held-out transitions with a median lead of ~0.5 s, holding the false alarm at
target where the plain focal rate drifts to 18%. This is well below the per-window 40%: the
offline certification is **necessary but not sufficient** for a stream, because
distinguishing a sustained instability from a damped fault online requires observing the
damping sign over several cycles — a physical lead-time / discrimination limit. The operating
point was chosen on a development half and reported on a held-out half; every window / step /
persistence / feature configuration searched is sealed in the artefact, so the honest
conclusion is auditable.

```bash
PYTHONPATH=src:. python -m bench.grid_modal_stream_operating_point \
    <dir-with-PSML-scenarios> \
    examples/real_data/psml_modal_growth/grid_modal_head_to_head.json \
    examples/real_data/psml_modal_growth
```

reproduces the sealed `content_hash`
(`3e2d74b7970d9f4aa78cda431f90d0bb4c6dc9cadd9366e45c79474f2d31f8f2`); the integrity test
`tests/test_psml_stream_operating_point.py` recomputes it from the committed payload alone
and pins it.

## The advisory — from a stream alarm to an operator decision, honestly

`grid_early_warning_advisory.json` closes the loop end-to-end on real data: the certified
streaming monitor (built straight from the operating-point artefact above) is replayed over
a real generator-trip scenario (`Natural Oscillation/row_108`), and the first stream alarm
is turned into a claim-bounded, review-only operator advisory
(`assurance.grid_early_warning_advisory`). The sealed record surfaces the growth rate
σ = 1.47 (crossing the certified threshold 1.20) on the most-unstable bus, the full
operating point, and — as first-class fields — the **honest recall (11/45 ≈ 24 %)** and the
matched stream false alarm (≈ 10 %) read from the operating-point artefact.

It is honest by construction. The advisory carries `non_actuating = true` and
`actuating = false`: it never actuates, it informs a human review. It is sealed **without**
a ground-truth onset — the real live-deployment case, where an operator receives the alarm
and investigates without knowing when (or whether) an onset follows — so no lead is claimed.
The sealed recall makes the limit explicit: an advisory is a reason to inspect, and the
*absence* of an advisory is not evidence of stability. Regenerate it (the raw voltages are
read but never redistributed; only the derived advisory is committed):

```bash
python bench/grid_advisory_example.py DATA \
  examples/real_data/psml_modal_growth/grid_modal_stream_operating_point.json \
  examples/real_data/psml_modal_growth/grid_early_warning_advisory.json
```

with `DATA` the PSML `Millisecond-level PMU Measurements` directory. The integrity test
`tests/test_grid_advisory_example_evidence.py` recomputes the seal from the committed
payload alone (np.polyfit BLAS drift makes a fresh pipeline run reproduce σ only to
floating-point tolerance, so the seal — not the re-run — is what is pinned).

## References

* Zheng, C. et al. 2021. *PSML: A Multi-scale Time-series Dataset for Machine Learning in
  Decarbonized Energy Grids.* — the 23-bus millisecond-level PMU corpus with
  disturbance-type annotations.
* Kundur, P. 1994. *Power System Stability and Control.* — small-signal (modal) stability:
  a mode's eigenvalue real part is its growth rate, the sign of instability.
