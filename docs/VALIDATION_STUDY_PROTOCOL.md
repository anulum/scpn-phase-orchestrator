<!--
SPDX-License-Identifier: AGPL-3.0-or-later
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# Protocol for Validating Oscillator-Based Auditory Entrainment

**Status: PROTOCOL ONLY — no data have been collected and no results are reported.**

This document defines the design, procedures, and statistical plan for
a future clinical validation study. It is pre-registered before any
data collection begins.

---

## 1. Objective

Test whether the SCPN Phase Orchestrator's Kuramoto-based auditory
stimulation produces genuine neural entrainment (phase-locking between
stimulus and cortical oscillations) as opposed to mere evoked responses.

### 1.1 Primary Hypothesis

Oscillator-driven auditory entrainment produces significantly higher
Inter-Trial Phase Coherence (ITPC) at the stimulus frequency compared
to amplitude-matched noise control.

### 1.2 Secondary Hypotheses

- ITPC persists after a brief stimulus pause (≥500 ms), distinguishing
  entrainment from evoked response (the "persistence test").
- Entrainment is frequency-specific: ITPC increase at the target
  frequency exceeds ITPC at neighbouring ±2 Hz frequencies.

---

## 2. Study Design

**Type:** Double-blind, within-subject crossover.

**Arms:**
1. **Active:** Kuramoto-optimised auditory stimulation at target frequency
   (individually calibrated, typically 10 Hz alpha or 40 Hz gamma).
2. **Control:** Amplitude-matched pink noise with no periodic structure.

**Crossover:** Each participant completes both arms, separated by a
≥48-hour washout period. Arm order is counterbalanced (block
randomisation, block size = 4).

**Blinding:** Participants and the EEG analyst are blinded to condition.
The stimulation software assigns condition labels (A/B) that are
unblinded only after statistical analysis is locked.

---

## 3. Participants

**Sample size:** N ≥ 60 (see §7 Power Analysis).

**Inclusion criteria:**
- Age 18–65 years.
- Normal hearing (pure-tone audiometry ≤25 dB HL, 250–8000 Hz).
- No history of epilepsy or photosensitive seizures.
- Able to provide informed consent.

**Exclusion criteria:**
- Current psychoactive medication affecting EEG.
- History of neurological disorder.
- Non-removable ferromagnetic implants (if MEG is used).
- Participation in another neurostimulation study within 30 days.

---

## 4. Procedure

### 4.1 Session Timeline (per arm)

| Time (min) | Activity |
|-----------|----------|
| 0–10      | Electrode application, impedance check (<10 kΩ) |
| 10–15     | Resting-state baseline (eyes open, eyes closed) |
| 15–30     | Stimulation block 1 (5 min stim + 2 min pause + 5 min stim) |
| 30–35     | Rest break |
| 35–50     | Stimulation block 2 (same structure as block 1) |
| 50–55     | Post-stimulation resting state |
| 55–60     | Adverse event questionnaire, electrode removal |

### 4.2 Stimulation Parameters

- Carrier: amplitude-modulated tone, modulation depth 100%.
- Target frequency: individually determined from resting-state peak alpha
  (or fixed 40 Hz for gamma protocol).
- Sound level: 65 dB SPL, delivered via insert earphones.
- The 2-minute pause within each block is the critical window for the
  persistence test (§1.2).

### 4.3 EEG Recording

- System: 64-channel, 1000 Hz sampling rate.
- Reference: average reference (online: Cz; offline: re-referenced).
- Electrodes of interest: Fz, Cz, Pz, Oz (midline), plus bilateral
  temporal (T7/T8) for auditory cortex.

---

## 5. Outcome Measures

### 5.1 Primary Outcome

**ITPC change from baseline** at the stimulus frequency, averaged
across midline electrodes (Fz, Cz, Pz), computed as:

    ΔITPC = ITPC_stim − ITPC_baseline

ITPC is defined as |mean(exp(i·θ_k))| across trials k at each
time-frequency bin (Lachaux et al. 1999), extracted via Morlet wavelet
(width = 7 cycles).

### 5.2 Secondary Outcomes

1. **Persistence ITPC:** ITPC computed during the 2-minute pause
   window, compared to pre-pause and post-pause segments.
2. **Frequency specificity:** ITPC at target frequency vs. ITPC at
   target ± 2 Hz.
3. **Subjective ratings:** Visual Analogue Scale (VAS) for
   relaxation, focus, and comfort (0–100 mm).

---

## 6. Ethics and Safety

### 6.1 IRB Approval

The study requires approval from an accredited Institutional Review
Board (or national equivalent Ethics Committee) before recruitment.
The protocol, informed consent form, and recruitment materials must
all be approved.

### 6.2 Informed Consent

Written informed consent obtained from all participants, covering:
- Purpose, procedures, and duration.
- Risks: potential mild discomfort from earphones, fatigue, and
  (very rare) auditory-evoked dizziness.
- Right to withdraw at any time without consequence.
- Data handling and anonymisation procedures.

### 6.3 Adverse Event Monitoring

- Participants complete a symptom checklist before and after each session.
- Monitored symptoms: headache, tinnitus, dizziness, nausea, fatigue.
- Stopping rule: if ≥3 participants in any arm report a grade ≥2
  adverse event (moderate, interferes with daily activities), the
  Data Safety Monitoring Board reviews before continuing.
- Serious adverse events reported to the IRB within 24 hours.

### 6.4 Data Protection

- All EEG data pseudonymised at acquisition (participant ID only).
- Linking table stored separately, encrypted, accessible only to PI.
- Compliant with GDPR (if EU) or equivalent local regulation.

---

## 7. Statistical Plan

### 7.1 Primary Analysis

Mixed-effects linear model:

    ΔITPC ~ condition + period + sequence + (1|participant)

- `condition`: active vs. control (fixed effect of interest).
- `period`: first vs. second session (fixed, controls learning/fatigue).
- `sequence`: AB vs. BA (fixed, controls carryover).
- `(1|participant)`: random intercept.

Significance threshold: α = 0.05 (two-sided). No multiplicity
adjustment for the primary outcome (single pre-specified contrast).

### 7.2 Secondary Analyses

- Persistence test: paired t-test on ITPC during pause vs. baseline,
  Bonferroni-corrected for 2 comparisons (α = 0.025).
- Frequency specificity: repeated-measures ANOVA across 3 frequency bins
  (target, target+2, target−2), Greenhouse-Geisser correction.
- Effect sizes: Cohen's d with 95% CI for all contrasts.

### 7.3 Power Analysis

- Expected effect size: d = 0.45 (medium), based on Zoefel et al. 2018
  and Notbohm et al. 2016 entrainment studies.
- Power: 0.80. Alpha: 0.05 (two-sided).
- Design: crossover (within-subject), correlation between arms ρ = 0.5.
- Required N: ~52 (paired test). Recruiting N = 60 to allow for ~15%
  dropout/artifact rejection.

### 7.4 Missing Data

- Epochs with >100 μV amplitude or >3 SD from channel mean are rejected.
- Participants with <50% usable epochs in any condition are excluded
  from primary analysis and reported separately.
- Sensitivity analysis: multiple imputation under MAR assumption.

---

## 8. Budget Estimate

| Item | Unit Cost | Quantity | Total |
|------|-----------|----------|-------|
| EEG lab rental | $80/hr | 120 hr (60 participants × 2 hr) | $9,600 |
| Participant compensation | $50/session | 120 sessions | $6,000 |
| EEG consumables (gel, caps) | $15/session | 120 | $1,800 |
| Insert earphones (single-use tips) | $5/session | 120 | $600 |
| Audiometry screening | $30/participant | 60 | $1,800 |
| Research assistant (recruitment, data collection) | $25/hr | 400 hr | $10,000 |
| EEG analysis software licence | — | 1 year | $2,000 |
| Statistical consultation | — | 20 hr @ $150/hr | $3,000 |
| IRB fees | — | 1 | $2,000 |
| Contingency (15%) | — | — | $5,520 |
| **Total** | | | **~$42,320** |

Note: if MEG is used instead of EEG, add ~$30,000 for scanner time
(bringing total to ~$72,000). The $70–80K estimate in project planning
assumes MEG as an option.

---

## 9. Timeline

| Phase | Duration | Milestone |
|-------|----------|-----------|
| IRB submission + approval | 2–3 months | Protocol approved |
| Recruitment + screening | 2 months | 60 enrolled |
| Data collection | 3–4 months | 120 sessions complete |
| Preprocessing + analysis | 2 months | Statistical report |
| Manuscript preparation | 2 months | Submitted for review |
| **Total** | **~12 months** | |

---

## 10. References

- Lachaux, J.-P., et al. (1999). Measuring phase synchrony in brain
  signals. *Human Brain Mapping*, 8(4), 194–208.
- Notbohm, A., Kurths, J., & Herrmann, C. S. (2016). Modification of
  brain oscillations via rhythmic sensory stimulation. *Frontiers in
  Human Neuroscience*, 10, 246.
- Zoefel, B., ten Oever, S., & Sack, A. T. (2018). The involvement of
  endogenous neural oscillations in the processing of rhythmic input.
  *Frontiers in Neuroscience*, 12, 834.
- Rechtschaffen, A., & Kales, A. (1968). *A manual of standardized
  terminology, techniques and scoring system for sleep stages of human
  subjects*. US Government Printing Office.
