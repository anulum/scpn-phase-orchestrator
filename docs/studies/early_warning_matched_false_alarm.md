<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 | Contact: www.anulum.li | protoscience@anulum.li -->

# Generic early-warning detection is at chance across five modalities; a domain-specific detector clears the bar where the signature is deterministic

**Miroslav Šotek** (ANULUM / Fortis Studio) · ORCID 0009-0009-3560-0851 · protoscience@anulum.li

*A same-protocol replication across brain, heart, power-grid and palaeoclimate data, with an AR(1)-Kendall-τ head-to-head against Dakos et al. 2008, a dynamical-network-biomarker extension to single-cell and bulk cancer/injury transcriptomics, and a domain-specific head-to-head in which a power-grid modal-growth detector beats the whole generic suite on a real corpus.*

---

## Abstract

Generic early-warning signals (EWS) — the rising variance and lag-one autocorrelation of *critical slowing down*, and related synchronisation and ordinal-entropy indicators — are widely reported to precede abrupt transitions in the brain, the heart, power systems and the climate. Most of that evidence rests on a **retrospective, per-record** test: a rising trend measured over one pre-transition window and compared to surrogates of the *same* record. We ask a different, operational question: **at a fixed false-alarm budget, does an early-warning detector fire on transitions more often than on no-transition controls?** We build one domain-adaptable detector suite and one matched-false-alarm evaluation harness, apply them unchanged (through a per-domain adapter) to four independent labelled corpora — scalp-EEG seizures, cardiac atrial-fibrillation onsets, power-grid growing oscillations, and palaeoclimate abrupt transitions — and score every result with a label-permutation significance test. Across all four domains, **no detector reaches significance** (every best-member p ≥ 0.05): the sparse detection observed is what the matched false-alarm rate produces by chance. Running the canonical literature detector — the Dakos et al. 2008 AR(1)-Kendall-τ trend — through the **same** protocol on the **same** segments confirms it: on its own palaeoclimate records it leads 0 of 6 transitions (p = 1.00), and it beats chance in none of the four domains, though on scalp EEG it is the strongest signal anywhere (3 of 6, p = 0.067). We conclude that the operational bar — matched false alarm plus a significance test — is stricter than the retrospective per-record test the literature uses, that generic EWS *detection* behaves as a commodity at this bar, and that the defensible deliverable is not a lead but the **auditable, hash-sealed, byte-reproducible evidence** — including the sealed silences — that the protocol produces. Extending the same honesty to a **molecular fifth modality** — dynamical-network-biomarker (DNB) early warning of cell-fate and disease transitions, where the statistic is cross-sample rather than sliding-window — reaches the same conclusion: the celebrated single-cell transition index of Mojtahedi et al. 2016 clears a matched operating point on only 1 of 3 leukaemic lineages (permutation p = 0.27), and the canonical GSE2565 phosgene-injury DNB benchmark does **not** beat a *selection-controlled* surrogate null (surrogate-rank p = 0.39) that re-selects the biomarker module on each shuffled surrogate — showing the apparent DNB rise is largely a selection artefact. The same protocol also separates commodity detection from genuine skill. Where a domain carries a *deterministic, detectable* instability signature, a **domain-specific** detector clears the operational bar decisively: on the PSML power grid, a modal envelope-growth detector — the exponential growth rate σ of the most unstable bus's cross-bus voltage deviation, the canonical wide-area-monitoring instability quantity — leads 36 of 90 growing-instability transitions (permutation p = 0.0001) where every generic member is at chance (best 13 of 90, p = 0.18), on the identical non-circular disturbance-type split at a matched false alarm, and a held-out half confirms it (24 of 45, p = 0.0002). A domain-specific detector does not automatically win: a spectral detector on the murky preictal scalp EEG is itself at chance. It is the matched-false-alarm moat that certifies which — commodity where the signature is absent, genuinely skilled where it is present. We state the limitations plainly: the corpora are small (3–90 transitions; the bulk DNB case has one exposed arm), so the tests have low power, and "at chance" bounds the *demonstrated* skill rather than proving early warning impossible.

---

## 1. Background and question

The theory of critical transitions predicts that a dynamical system approaching a bifurcation recovers more slowly from perturbations, which shows up as a rising variance and a rising lag-one autocorrelation of an observable — *critical slowing down* (Scheffer et al. 2009). The same idea motivates rising-synchronisation and ordinal-transition-entropy indicators. These generic early-warning signals have been reported before epileptic seizures, cardiac arrhythmias, power-system instability, ecological collapse and abrupt climate change.

Almost all of that evidence answers a **retrospective, per-record** question: *within one pre-transition window, is there a statistically significant rising trend of the indicator, relative to surrogate time series of that same record?* (e.g. the Kendall-τ trend test of Dakos et al. 2008). This question is valuable but it is not the question an operator or clinician faces. The operational question is: *if I set an alarm threshold that fires on at most a fixed fraction of no-transition situations, does the detector then fire on genuine transitions more often than that fraction?* — a **matched-false-alarm** question, followed by *is the transition hit-rate above chance?*

This study builds the machinery to answer the operational question honestly, applies it across four independent physical domains with one shared suite, and runs the canonical literature detector through the identical protocol as a same-segments head-to-head.

## 2. Methods

### 2.1 The detector suite

One suite of three passive detectors reads a neutral observable bundle and emits per-window scores:

- **Critical slowing down** — the larger of the robust (median/MAD) z-scores of the windowed variance and lag-one autocorrelation of the observable, against a leading baseline.
- **Rising synchronisation** — the robust z-score of the Kuramoto order parameter of a population of phase oscillators.
- **Ordinal-transition entropy** — the robust z-score of the (negated) ordinal-pattern transition entropy of the phase field.
- **Weighted fusion** — the weighted mean of the three members.

The single-series domains (palaeoclimate) carry one scalar observable, so only critical slowing down applies; the multi-node domains (EEG, ECG, grid) carry a population of oscillators and use all four.

### 2.2 The matched-false-alarm harness

For each domain: a per-domain adapter turns the raw recording into the neutral observable bundle. Each transition is scored on a **fixed pre-onset segment ending at the annotated onset**, so every window is pre-onset and any alarm is a genuine lead. A **no-transition null** — a stable, transition-free stretch — is cut into non-overlapping trials of the same length. Each detector's alarm threshold is set **continuously** to the tightest value holding the trial false-alarm rate at or below a target of 10 % (the quantile of the null alarm scores, with no grid ceiling, so a variance-heavy detector is never silently clipped). A transition is *led* when the detector alarms within its pre-onset segment. The achieved false-alarm rate is recorded alongside every threshold.

### 2.3 The permutation significance test

A sealed detection count leaves one question open: does the detector lead transitions *more than the matched false-alarm rate explains*? We answer it with a **label-permutation (exchangeability) test**. Each segment — transition or null — alarms or not at the calibrated threshold. Under the null hypothesis that transition segments are indistinguishable from nulls, which of the pooled segments carry the "transition" label is exchangeable. We draw 10 000 random transition-sized subsets of the pooled alarm outcomes (fixed seed, so the p-value is byte-reproducible), build the null distribution of the lead count, and report the one-sided p-value with an add-one correction. A small p-value means the detector beats the matched false-alarm rate.

### 2.4 The competitor head-to-head

The canonical literature detector is Dakos et al. 2008: the **Kendall rank correlation τ of the windowed lag-one autocorrelation against time** — a rising-AR(1) trend. We implement it as a per-segment score, calibrate a matched-false-alarm τ threshold on the null segments exactly as above, and run the **same** permutation test on the resulting alarms. Because the two detectors read the same one-dimensional signal (the detrended proxy residual for palaeoclimate, the cross-node Kuramoto order parameter for the multi-node domains), the head-to-head is a same-segments, same-budget, same-test comparison in which only the detector differs.

### 2.5 Corpora (citation-only)

| Domain | Corpus | Transitions | Null |
|--------|--------|:-----------:|------|
| Brain (scalp EEG) | CHB-MIT, subject chb01 (Shoeb 2009, PhysioNet) | 6 seizures | interictal records |
| Heart (ECG) | MIT-BIH AFDB (Moody & Mark 1983, PhysioNet) | 6 AF onsets | sinus stretches |
| Grid (PMU) | PSML 23-bus co-simulation (Zheng et al. 2021) | 12 growing oscillations | damped disturbances |
| Palaeoclimate | Dakos et al. 2008 records (earlywarningtoolbox/datasets) | 6 of 8 evaluated | stable pre-approach intervals |

All raw data are public and cited; none is redistributed. Only derived, hash-sealed, byte-reproducible evidence records are committed. Each detector's alarm — or silence — is sealed into a content-addressed (canonical-JSON SHA-256) `EarlyWarningEvidence` record, so the result, including every sealed silence, is auditable and reproduces bit-for-bit from a fresh run.

### 2.6 The molecular fifth modality (DNB)

Dynamical-network-biomarker early warning (Chen et al. 2012) is a *different modality*: few timepoints, many genes, and a **cross-sample** statistic computed over the sample population at each timepoint, so the sliding-window suite does not apply. Its index rises to a peak at the tipping point. Two published index forms are implemented and each is run through a modality-appropriate honest test that keeps the same philosophy — a controlled null and a permutation/rank p-value:

- **Single-cell (Mojtahedi et al. 2016).** The transition index is the ratio of mean signed gene–gene to mean signed cell–cell correlation over the curated panel (which *is* the critical module, so no selection). We take the per-lineage index trajectory and its bootstrap standard error **from the published supplement** (not re-derived from the confounded raw qPCR); the pre-transition rising limb (Days 0, 1, 3) is scored by its least-squares slope; the matched-false-alarm null is a **temporally-shuffled surrogate** resampled from the published mean and SE; and the three lineages' alarms are tested by the same label-permutation core as the physical domains.
- **Bulk (GSE2565 phosgene lung injury).** Here the module must be *selected*, which introduces **selection freedom**: the composite index `SD_in · PCC_in / PCC_out` is evaluated on a module chosen to peak at the transition, so a rise is partly guaranteed by construction. The only honest null **re-selects the entire module on each surrogate**: it shuffles the timepoint labels across the arm's samples and reruns module selection from scratch, so the surrogate has the same selection freedom as the real analysis. The observed rising slope is ranked against these selection-controlled surrogates for a one-sided p-value, and also thresholded to a matched false alarm.

Both DNB paths seal a hash-addressed, byte-reproducible record exactly as the physical domains do.

### 2.7 Domain-specific detectors

The generic suite reads a modality-neutral observable bundle, which is what makes the four-domain replication possible — but it is also why it is a commodity. A **domain-specific** detector instead reads the raw physical quantity whose growth *is* the instability, and is tested through the *same* matched-false-alarm and permutation moat, so its p-value is directly comparable to the generic suite on the same segments. Two are built, one for a domain with a deterministic signature and one for a murky domain, precisely to show the moat certifies the difference rather than the detector always winning:

- **Power-grid modal envelope-growth.** When a disturbance leaves an electromechanical mode under-damped, the amplitude of the cross-bus voltage deviation grows exponentially — the real part σ of the mode's eigenvalue is positive, the canonical wide-area-monitoring early-warning quantity (Kundur 1994). The detector estimates σ directly as the slope of the log deviation envelope. Two orthogonal, physically-motivated refinements are added and selected on a development half only: a **focal** aggregation takes the growth rate of the *most unstable bus* (instability is localised to a mode/bus cluster, so the per-bus maximum, un-diluted, is the signal; the matched-false-alarm threshold is calibrated on that same per-bus maximum over the nulls, absorbing the multiplicity), and a **recency weighting** up-weights the later samples (a real instability accelerates toward the disturbance). Every variant tried is disclosed in §3.5.
- **Scalp-EEG spectral rise.** The seizure-prediction literature reports a rising high-frequency to low-frequency band-power ratio before onset (e.g. Mormann et al. 2007). The detector takes the beta(13–30 Hz)/delta(0.5–4 Hz) band-power ratio per channel, its Kendall-τ rising trend over the pre-onset segment, with the same whole-head and focal aggregations. Its bands are fixed a-priori from the literature, not tuned.

The grid labels are **non-circular**: a transition is any generator-trip scenario and a null any damped bus-fault or branch-trip scenario — the label is the disturbance *type*, a physical annotation independent of the growth statistic the detector measures, so a growing-oscillation detector is never scored against a growth-defined label. Three PSML scenarios carry a non-monotonic time column (a non-physical, negative inferred sampling rate) and are dropped, the count recorded in the sealed payload. The operating point (focal aggregation, recency weighting) is chosen on the even-index development half and validated on the odd-index held-out half before the full-corpus comparison, so no operating point is tuned to the reported answer.

## 3. Results

### 3.1 SCPN suite across four domains

At a matched false-alarm rate (target 10 %), detection is sparse everywhere, and **no detector reaches significance in any domain**:

| Domain | Best member | Led / N | Permutation p |
|--------|-------------|:-------:|:-------------:|
| Brain (EEG) | critical slowing down | 2 / 6 | 0.215 |
| Heart (ECG) | synchronisation | 2 / 6 | 0.219 |
| Grid (PMU) | critical slowing down | 3 / 12 | 0.195 |
| Palaeoclimate | critical slowing down | 1 / 6 | 0.514 |

The fusion never leads more transitions than its best single member, so there is no robust fusion advantage. The observed lead counts are within what the matched false-alarm rate produces by chance (expected counts 0.6–1.7 across the domains).

### 3.2 Head-to-head: SCPN vs Dakos AR(1)-Kendall-τ

Running the canonical Dakos detector through the identical protocol on the identical segments:

| Domain | SCPN best member (led/N, p) | Dakos AR(1)-Kendall-τ (led/N, p) |
|--------|:---------------------------:|:--------------------------------:|
| Palaeoclimate | CSD 1/6, p = 0.514 | 0/6, **p = 1.000** |
| Grid | CSD 3/12, p = 0.195 | 1/12, **p = 0.713** |
| Heart | sync 2/6, p = 0.219 | 0/6, **p = 1.000** |
| Brain (EEG) | CSD 2/6, p = 0.215 | **3/6, p = 0.067** |

On its own palaeoclimate records — the data Dakos et al. analysed — the AR(1)-Kendall-τ detector, at a matched operating point, leads **zero** of six transitions (p = 1.00): it does not beat chance, and it does no better than the SCPN suite. The same holds for grid and heart. The one exception is scalp EEG, where the rising-AR(1) trend is the **strongest signal anywhere in the study** (3 of 6, p = 0.067) — marginally short of significance, and better than the SCPN detectors on that domain.

### 3.3 The honest reading

The result is not "nothing works". It is: **at a matched operating point and a permutation significance test, nothing reaches significance at these corpus sizes** — neither the SCPN suite nor the canonical literature detector, in any of the four domains. The literature's positive EWS results come from a *retrospective per-record* significance test that this operational protocol is stricter than. The one place a detector comes close — the AR(1) trend on scalp EEG — is consistent with the seizure-prediction literature's use of autocorrelation features and warrants a larger, higher-power EEG corpus.

### 3.4 The molecular fifth domain (DNB)

The same conclusion holds when the honesty is carried to a molecular modality:

| Case | Test | Result | p |
|------|------|--------|:-:|
| Single-cell (Mojtahedi 2016) | matched-FA + permutation, 3 lineages | 1 / 3 lineages clear the 10 % operating point | 0.266 |
| Bulk (GSE2565 phosgene) | selection-controlled surrogate rank, 1 exposed arm | exposed-arm rise does not beat reselecting surrogates | 0.385 |

The single-cell transition index rises 2.8-fold toward the erythroid bifurcation — an unmistakable effect size — yet at its published four-timepoint resolution only the strongest of three lineages clears a matched false-alarm operating point, and the corpus-level permutation test is not significant (p = 0.27); the binding constraint is the temporal resolution, not the effect size. The bulk GSE2565 case is sharper: the phosgene-exposed arm's apparent DNB rise (slope 1.15) sits *below* the 90th percentile of surrogates that were allowed to **re-select the biomarker module on shuffled time** (surrogate-rank p = 0.385), and it barely exceeds the air control's own selected-module rise (p = 0.414). Once the null is granted the same selection freedom as the analysis, the celebrated bulk DNB rise is largely a **selection artefact**. Both molecular cases are hash-sealed and byte-reproducible.

### 3.5 Domain-specific detectors: a real signature beats the bar

The generic result is that *modality-neutral* detection is a commodity. The complementary result is that a *domain-specific* detector, tested through the identical moat, beats it decisively — but only where the domain carries a deterministic signature.

On the PSML power grid, the non-circular disturbance-type split gives **90 growing-instability (generator-trip) transitions and 88 damped (bus-fault / branch-trip) nulls** — a larger and label-independent corpus than the growth-selected 12 of §3.1. Both the modal detector and the generic suite read the *same* two-second pre-onset segments of the *same* scenarios, calibrated to the *same* 10 % matched false alarm on the *same* nulls (achieved 9.1 %); only the detector differs. At its validated operating point (focal aggregation, recency weighting):

| Detector | Transitions led / 90 | Permutation p |
|----------|:--------------------:|:-------------:|
| generic critical slowing down | 13 / 90 | 0.185 |
| generic weighted ensemble | 11 / 90 | 0.324 |
| generic ordinal-transition entropy | 8 / 90 | 0.629 |
| generic synchronisation | 3 / 90 | 0.976 |
| **modal envelope-growth (focal, recency)** | **36 / 90** | **0.0001** |

The domain-specific detector leads **36 of 90** transitions (40 %) at p = 0.0001, beating every generic member — the best of which (critical slowing down, 13/90, p = 0.19) is at chance. The **held-out half** confirms it: 24 of 45 transitions (53 %) at p = 0.0002, the unbiased estimate, sealed in the artefact.

The winning operating point was reached by an explicit, disclosed variant search on the development half (choice made on development, reported on held-out — no head-to-head data touched during selection):

| Variant | Development led / 45, p | Held-out led / 45, p |
|---------|:-----------------------:|:--------------------:|
| whole-network log-slope σ | 11, 0.051 | 14, 0.009 |
| per-bus maximum σ (focal) | 14, 0.008 | 22, 0.0001 |
| matrix-pencil, PCA-dominant mode | 10, 0.072 | 3, 0.80 |
| matrix-pencil, per-bus maximum | 8, 0.189 | 6, 0.39 |
| Hilbert-envelope σ, per-bus | 2, 0.90 | 3, 0.80 |
| **recency-weighted focal σ** | **16, 0.002** | **24, 0.0002** |

The gold-standard wide-area-monitoring estimator — **matrix-pencil modal damping** — recovers a planted σ exactly on synthetic data, yet on the short two-second, 238 Hz windows it is at chance (held-out p = 0.39–0.80); the robust direct log-envelope slope wins on real noisy segments. A recency-ratio sensitivity sweep (2–8) beats the unweighted per-bus detector across the whole range, so the choice is robust, not knife-edge.

The counterpoint confirms this is not detector-worship. The **scalp-EEG spectral** detector — a beta/delta band-power rise, a-priori bands, tested through the same moat on the chb01 seizures — is itself at chance (1 of 6 led, p ≈ 0.56, in both whole-head and focal aggregations): the preictal state is murky and the corpus small, so a domain-specific detector on a domain without a clean signature fares no better than the generic suite. **The moat is what certifies the difference** between a genuine, physically-grounded early warning (grid modal growth) and a plausible but empty one (preictal spectral rise) — the same honest test that flags the commodity detectors as at chance flags the domain-specific one as genuinely skilled.

## 4. Discussion

**Detection is a commodity; the moat is the evidence.** Across four independent physical domains — and a fifth, molecular one — generic early-warning *detection* at an honest operating point is sparse and, by a permutation or selection-controlled test, at chance. This is not a defect of one suite: the canonical Dakos detector fares no better on its own data, and the celebrated single-cell and bulk DNB benchmarks do not clear a modality-appropriate honest null either. What is *not* a commodity is the auditable, reproducible, claim-bounded envelope the protocol produces — a matched-false-alarm operating point, a permutation p-value, and a hash-sealed `EarlyWarningEvidence` record for every transition, **including the sealed silences**. A positive early-warning claim should be required to clear this operational bar; most published EWS results have only cleared the retrospective per-record one.

**A domain-specific detector wins where the signature is deterministic — and the moat certifies which.** The commodity result is about *modality-neutral* detectors. When the domain carries a physically deterministic, directly-measurable instability — a growing electromechanical mode, whose exponential envelope *is* the eigenvalue crossing into instability — a detector that reads that quantity beats the whole generic suite decisively (grid modal growth, 36/90, p = 0.0001, versus every generic member at chance). But the same detector *idea* on a murky domain (the preictal spectral rise) is itself at chance. So the deliverable is not "build domain-specific detectors" as a slogan — a domain-specific detector does not automatically win — but the **matched-false-alarm moat that certifies the difference**: the identical honest test that flags commodity detection as at chance also licenses a genuinely-skilled detector where a real signature exists, and refuses to license a plausible-but-empty one where it does not. Value is unlocked per-domain, and only when the physics carries a detectable signal that clears the operational bar.

**Selection freedom is a hidden operating cost.** The bulk-DNB case adds a specific lesson: when the detector *chooses* its own module (or feature set) to peak at the transition, an honest null must be granted the same choice. A surrogate that re-selects from scratch on shuffled time absorbs most of the apparent signal — so a fair null for a self-selecting detector is not an afterthought but the whole test.

**The evaluation gap is the finding.** The difference between "there is a detectable rising trend in this record's approach" (retrospective, per-record, vs surrogates) and "the detector fires on transitions more than on controls at a fixed false-alarm budget" (operational, matched, significance-tested) is exactly the gap between the literature's positive results and this study's null. That gap — not any single detector — is what the four-domain replication and the Dakos head-to-head expose.

## 5. Limitations

Stated plainly, because they bound the claim:

- **Small corpora, low power.** With 6–12 transitions per physical domain — and only 3 lineages (single-cell) or 1 exposed arm (bulk) in the molecular one — the significance tests have limited power; "at chance" bounds the *demonstrated* skill and does **not** prove early warning is impossible. A modestly skilled detector could remain non-significant at this n.
- **One competitor.** Only the Dakos AR(1)-Kendall-τ detector was run head-to-head on the physical domains; modern approaches (e.g. deep-learning tipping-point predictors) were not tested and could behave differently.
- **Single-subject EEG; within-record climate null.** The EEG corpus is one subject (chb01); the palaeoclimate null is the stable pre-approach interval of each record, which carries that record's own variability and so is conservative.
- **Parameterisation.** One reasonable choice of window, step, baseline and target false alarm was used per domain; a sweep was not performed.
- **DNB caveats.** The single-cell index is taken from the published supplement (its bootstrap SE drives the surrogate), not re-derived from the confounded raw qPCR; the bulk case has one exposed arm, so it is a single-transition surrogate test, not a corpus, and its module search is a greedy reimplementation of the DNB selection rather than the authors' exact procedure. These bound the molecular result to "no significance at this resolution / under a selection-controlled null", not a refutation of the DNB method.
- **Domain-specific operating point.** The grid modal detector's operating point (focal aggregation, recency weighting) was chosen by a variant search on a development half and reported on a held-out half, so the held-out lead count (24/45, p = 0.0002) is the unbiased estimate and the full-corpus count (36/90) is the fitted-model deployment figure; both are sealed. The corpus is one power-system dataset at one sampling rate with two-second windows, so the strong grid result bounds *demonstrated* skill on PSML growing-oscillation transitions, not a universal grid claim. The scalp-EEG spectral counterpoint is one subject (chb01, six seizures), so its at-chance result is low-power like the generic EEG result.

## 6. Reproduction

Every result regenerates deterministically (no randomness in the pipelines; the permutation test uses a fixed seed). With each raw corpus in a directory:

```bash
python bench/early_warning_leadtime_eeg.py     DATA OUT   # brain
python bench/early_warning_leadtime_cardiac.py DATA OUT   # heart
python bench/early_warning_leadtime_grid.py    DATA OUT   # grid
python bench/early_warning_leadtime_climate.py DATA OUT   # palaeoclimate
python bench/head_to_head_ar1_kendall.py OUT \
    --climate-dir C --grid-dir G --cardiac-dir H --eeg-dir E   # head-to-head
python -m bench.early_warning_dnb          OUT               # single-cell DNB (embedded summary)
python -m bench.early_warning_dnb_bulk     DATA OUT          # bulk GSE2565 DNB
python -m bench.grid_modal_head_to_head    PSML OUT          # grid modal vs generic head-to-head
```

The single-cell DNB capstone reads only the embedded published summary, so it needs no external data; the bulk capstone reads the citation-only GSE2565 files; the grid modal head-to-head reads the citation-only PSML scenarios. The sealed evidence, aggregate results (with the permutation block), the head-to-head comparisons, and the sealed grid modal-vs-generic result (`examples/real_data/psml_modal_growth/`) are committed under `examples/real_data/`. A fresh run reproduces every `content_hash` bit-for-bit; the integrity tests recompute each sealed hash from the committed payload alone.

## 7. Data availability

Raw corpora are public and cited, not redistributed:

- CHB-MIT Scalp EEG Database — <https://physionet.org/content/chbmit/>
- MIT-BIH Atrial Fibrillation Database — <https://physionet.org/content/afdb/>
- PSML power-system dataset (Zheng et al. 2021) — the 23-bus Millisecond-level PMU measurements.
- Dakos et al. 2008 palaeoclimate records — <https://github.com/earlywarningtoolbox/datasets>.
- Mojtahedi et al. 2016 single-cell transition index — Table S2 of the paper (index + bootstrap SE per lineage per day).
- GSE2565 phosgene lung-injury expression — <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE2565>.

## References

- Scheffer M, Bascompte J, Brock WA, et al. Early-warning signals for critical transitions. *Nature* 461:53 (2009).
- Dakos V, Scheffer M, van Nes EH, Brovkin V, Petoukhov V, Held H. Slowing down as an early warning signal for abrupt climate change. *PNAS* 105(38):14308 (2008).
- Dakos V, Carpenter SR, Brock WA, et al. Methods for detecting early warning signals of critical transitions in time series. *PLoS ONE* 7(7):e41010 (2012).
- Shoeb A. Application of machine learning to epileptic seizure onset detection and treatment. PhD thesis, MIT (2009); CHB-MIT via PhysioNet.
- Moody GB, Mark RG. A new method for detecting atrial fibrillation using R-R intervals. *Computers in Cardiology* 10:227 (1983).
- Goldberger AL, Amaral LAN, Glass L, et al. PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101(23):e215 (2000).
- Zheng X, et al. A multi-scale time-series dataset with benchmark for machine learning in electricity (PSML) (2021).
- Kundur P. *Power System Stability and Control.* McGraw-Hill (1994) — small-signal (modal) stability; a mode's eigenvalue real part is its growth rate.
- Mormann F, Andrzejak RG, Elger CE, Lehnertz K. Seizure prediction: the long and winding road. *Brain* 130(2):314 (2007).
- Chen L, Liu R, Liu Z-P, Li M, Aihara K. Detecting early-warning signals for sudden deterioration of complex diseases by dynamical network biomarkers. *Scientific Reports* 2:342 (2012).
- Mojtahedi M, Skupin A, Zhou J, et al. Cell fate decision as high-dimensional critical state transition. *PLoS Biology* 14(12):e2000640 (2016).
- Sciuto AM, Phillips CS, Orzolek LD, Hensley JL, Chang L-Y, Nadadur SS. Genomic analysis of murine pulmonary tissue following carbonyl chloride inhalation. *Chemical Research in Toxicology* 18(11):1654 (2005); GSE2565.
