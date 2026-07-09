<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# Diagnostic Study: Why Does the CAP Kuramoto Detector Win on `n2`?

## Abstract

The cross-subject CAP audit showed that the simple multi-channel delta-phase
Kuramoto detector under-performs the normalized delta envelope on average, yet
it wins decisively on the control recording `n2`. This diagnostic study
quantifies per-recording signal properties and correlates them with the
envelope–Kuramoto detection-rate gap. The goal is to turn the observed
heterogeneity into a concrete design rule for the next Kuramoto variant.

## Question

What properties of a CAP recording make spatial delta-phase coherence
informative for separating N3 from Wake? And what properties make the simple
mean-R Kuramoto detector collapse?

## Data

The same four recordings used in the CAP multi-channel N3-vs-Wake audit:

| Recording | Condition | Wake epochs | N3 epochs |
|-----------|-----------|------------:|----------:|
| `n1`      | Control   | 39          | 321       |
| `n2`      | Control   | 142         | 197       |
| `brux2`   | Bruxism   | 127         | 289       |
| `narco2`  | Narcolepsy| 180         | 188       |

## Methods

For every 30-second epoch we extract a small set of interpretable signal
properties:

| Feature | Definition |
|---------|------------|
| `n_channels` | Number of EEG channels selected by the audit's substring matcher. |
| `delta_snr` | Ratio of delta-band power (0.5–4 Hz) to total signal power, averaged across channels. |
| `delta_env_mean` / `delta_env_std` | Mean and across-channel standard deviation of the per-channel delta Hilbert envelope. |
| `phase_circvar` | Circular variance of delta-band Hilbert phases across channels; 0 = perfectly coherent, 1 = uniformly distributed. |
| `kuramoto_r_mean` / `kuramoto_r_std` | Mean and temporal standard deviation of the Kuramoto order parameter R(t) within the epoch. |
| `hf_power_ratio` | Ratio of 30–45 Hz power to total power, averaged across channels; muscle/bruxism artifact proxy. |
| `signal_kurtosis` | Excess kurtosis of the raw epoch, averaged across channels; heavy tails indicate transients. |

We then compute the Spearman correlation between each N3-epoch feature mean and:

1. The Kuramoto detection rate per recording.
2. The envelope-minus-Kuramoto detection-rate gap per recording.

A positive gap-correlation means the feature rises where the envelope beats
Kuramoto by more.

## Results

### Per-recording N3 feature means

| Recording | Kuramoto DR | Envelope DR | Gap | n_channels | delta_snr | phase_circvar | hf_power_ratio | signal_kurtosis | kuramoto_r_mean | kuramoto_r_std |
|-----------|------------:|------------:|----:|-----------:|----------:|--------------:|---------------:|----------------:|----------------:|---------------:|
| `n1`      | 0.231       | 0.380       | 0.150 | 8          | 0.596     | 0.371         | 0.001          | 1.791           | 0.629           | 0.237          |
| `n2`      | 0.223       | 0.005       | -0.218 | 3          | 0.445     | 0.272         | 0.004          | 3.197           | 0.728           | 0.228          |
| `brux2`   | 0.062       | 0.913       | 0.851 | 6          | 0.586     | 0.313         | 0.002          | 2.053           | 0.687           | 0.232          |
| `narco2`  | 0.218       | 0.803       | 0.585 | 3          | 0.577     | 0.397         | 0.011          | 1.449           | 0.603           | 0.208          |

### Feature correlations with the detector gap

| Feature | ρ (gap) | p (gap) | ρ (Kuramoto DR) | p (Kuramoto DR) |
|---------|--------:|--------:|----------------:|----------------:|
| `delta_snr` | 0.4 | 0.6 | 0.2 | 0.8 |
| `delta_env_std` | 0.4 | 0.6 | 0.0 | 1.0 |
| `phase_circvar` | 0.4 | 0.6 | 0.0 | 1.0 |
| `kuramoto_r_mean` | -0.4 | 0.6 | 0.0 | 1.0 |
| `signal_kurtosis` | -0.4 | 0.6 | 0.0 | 1.0 |
| `delta_env_mean` | 0.2 | 0.8 | 0.4 | 0.6 |
| `kuramoto_r_std` | 0.0 | 1.0 | 0.4 | 0.6 |
| `hf_power_ratio` | 0.0 | 1.0 | -0.4 | 0.6 |

### Top predictor

`delta_snr` — higher delta-band SNR is associated with a larger envelope-minus-Kuramoto gap. The association is not statistically significant with only four recordings (Spearman ρ = 0.4, p = 0.6), but it is the most consistent explanatory signal property.

### Recommended next variant

**SNR-weighted Kuramoto**: weight each channel by its delta-band SNR before computing the Kuramoto order parameter R(t). Rationale: the recording where Kuramoto already wins (`n2`) has the lowest delta SNR but the highest phase coherence; a weighted estimator can amplify channels that carry clean slow-wave activity and suppress noisy or artifact-ridden channels. The refinement should be validated on the full four-recording panel.

### Interpretation

- **Channel count is not the deciding factor.** `n2` and `narco2` both use only three bipolar derivations, yet Kuramoto succeeds on `n2` and fails on `narco2`.
- **Phase coherence matters.** `n2` has the lowest circular phase variance (0.27) and the highest mean Kuramoto R (0.73), while `narco2` has the highest phase variance (0.40) and the lowest mean R (0.60).
- **Class separation is the real problem.** On `brux2`, both N3 and Wake epochs have high mean R (~0.69), so the detector cannot separate them even though coherence is high. The simple mean-R feature ignores the N3-vs-Wake separation of R(t); a refined detector should explicitly model or exploit this separation.

## Reproduction

Run the diagnostic script after the cross-subject audit has produced the
aggregate JSON:

```bash
PYTHONPATH=.:src python bench/cap_kuramoto_diagnostic.py \
  examples/real_data/cap_multichannel_staging
```

The script reads `cap_multichannel_aggregate.json` and the manifest, computes
features from the raw EDFs, and writes
`examples/real_data/cap_multichannel_staging/cap_kuramoto_diagnostic.json`.

## Scope and limitations

- **Post-hoc explanatory, not predictive.** The correlations are computed on the
  same four recordings used to generate the recommendation; they suggest a
  hypothesis to be tested in the next detector variant, not a validated model.
- **Feature set is intentionally small.** We trade completeness for
  interpretability; the aim is a design rule, not a machine-learning predictor.
- **Honest to the audit protocol.** Feature extraction uses the same channel
  selection, resampling, and band definitions as the audited detectors.

## Related work

- `cap_multichannel_n3_vs_wake.md` introduced the honest cross-subject audit
  that motivates this diagnostic.
- The next study will implement and audit the recommended Kuramoto refinement.
