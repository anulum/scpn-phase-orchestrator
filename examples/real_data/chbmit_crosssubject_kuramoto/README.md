<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->

# CHB-MIT cross-subject Kuramoto generalisation audit

Sealed output of `bench/chbmit_crosssubject_validation.py` — a leave-one-subject-out
test of whether the global top-k PLV Kuramoto detector generalises beyond the
single subject (`chb01`) on which it was originally tuned.

For each held-out subject, `k` is calibrated **only on the other four subjects**,
then top-k PLV and the mean-R baseline are scored on the held-out subject's
pre-ictal windows versus its own interictal null epochs (seizure band 4–30 Hz,
30 s epochs, 5-minute pre-ictal window, matched 10 % false-alarm rate).

## Result

| Held-out | Pre-ictal epochs | Calibrated `k` | top-k PLV AUC | mean-R AUC |
|----------|-----------------:|---------------:|--------------:|-----------:|
| chb01 | 63 | 23 | 0.908 | 0.908 |
| chb02 | 18 | 15 | 0.114 | 0.275 |
| chb03 | 63 | 23 | 0.309 | 0.309 |
| chb04 | 27 | 23 | 0.460 | 0.348 |
| chb05 | 45 | 23 | 0.707 | 0.707 |
| **mean** | | | **0.499** | **0.509** |

Out-of-sample, top-k PLV beats mean-R on **1 / 5** subjects, both average ≈ 0.50
AUC (chance), and the calibrated `k` collapses to 23 (all channels = mean-R) on
four of five subjects. **The detector does not generalise across subjects**; the
`chb01` advantage was `k`-tuning on the evaluation subject, and the underlying
pre-ictal coherence signal is itself subject-specific (strong on `chb01`, weak on
`chb05`, at or below chance on `chb02`/`chb03`/`chb04`).

## Files

- `chbmit_crosssubject_kuramoto.json` — the sealed per-subject and aggregate record.

Raw CHB-MIT EDFs are citation-only (PhysioNet CHB-MIT Scalp EEG Database 1.0.0)
and are not redistributed; only this derived record is committed. See
`docs/studies/chbmit_multichannel_kuramoto.md` for the full discussion.
