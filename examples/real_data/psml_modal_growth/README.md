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

## References

* Zheng, C. et al. 2021. *PSML: A Multi-scale Time-series Dataset for Machine Learning in
  Decarbonized Energy Grids.* — the 23-bus millisecond-level PMU corpus with
  disturbance-type annotations.
* Kundur, P. 1994. *Power System Stability and Control.* — small-signal (modal) stability:
  a mode's eigenvalue real part is its growth rate, the sign of instability.
