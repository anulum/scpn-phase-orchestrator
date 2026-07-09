# Cross-Domain Detector Meta-Analysis Report

**Generated:** 2026-07-09 22:41 UTC

This report is produced automatically from the committed detector-evidence aggregates under ``examples/real_data/*/``. It normalises each detector's performance, ranks detectors within every domain, and derives a ranked backlog of refinement candidates.

## Data sources

| Domain | Source file | Schema |
| --- | --- | --- |
| afdb_atrial_fibrillation | `early_warning_leadtime_cardiac_results.json` | early-warning lead-time |
| afdb_atrial_fibrillation_multiscale | `early_warning_leadtime_cardiac_multiscale_results.json` | early-warning lead-time |
| cap_kuramoto_variants | `cap_kuramoto_variants.json` | early-warning lead-time |
| cap_multichannel_staging | `cap_multichannel_aggregate.json` | honest-audit aggregate |
| chb01_seizures | `early_warning_leadtime_eeg_results.json` | early-warning lead-time |
| chb01_seizures_multiscale | `early_warning_leadtime_eeg_multiscale_results.json` | early-warning lead-time |
| csd_variant_synthetic | `csd_variant_synthetic_results.json` | early-warning lead-time |
| dakos_climate_transitions | `early_warning_leadtime_climate_results.json` | early-warning lead-time |
| dakos_climate_transitions_multiscale | `early_warning_leadtime_climate_multiscale_results.json` | early-warning lead-time |
| psml_grid_oscillation | `early_warning_leadtime_grid_results.json` | early-warning lead-time |
| psml_grid_oscillation_multiscale | `early_warning_leadtime_grid_multiscale_results.json` | early-warning lead-time |
| synthetic_honest_audit_demo | `synthetic_honest_audit_demo.json` | honest-audit aggregate |

## Per-domain rankings

### afdb_atrial_fibrillation

| Rank | Detector | Detection rate | p-value | Beats chance |
| --- | --- | --- | --- | --- |
| 1 | `ensemble_weighted` | 33.3% | 2.193e-01 | False |
| 1 | `synchronisation` | 33.3% | 2.193e-01 | False |
| 3 | `critical_slowing_down` | 16.7% | 5.627e-01 | False |
| 4 | `transition_entropy` | 0.0% | 1.000e+00 | False |

### afdb_atrial_fibrillation_multiscale

| Rank | Detector | Detection rate | p-value | Beats chance |
| --- | --- | --- | --- | --- |
| 1 | `critical_slowing_down_multiscale` | 33.3% | 2.193e-01 | False |
| 1 | `ensemble_weighted` | 33.3% | 2.193e-01 | False |
| 1 | `synchronisation` | 33.3% | 2.193e-01 | False |
| 4 | `critical_slowing_down` | 16.7% | 5.627e-01 | False |
| 5 | `transition_entropy` | 0.0% | 1.000e+00 | False |

### cap_kuramoto_variants

| Rank | Detector | Detection rate | p-value | Beats chance |
| --- | --- | --- | --- | --- |
| 1 | `coherent_sustained_kuramoto` | 55.2% | 1.112e-03 | True |
| 2 | `normalized_delta_envelope` | 52.5% | 1.000e-03 | True |
| 3 | `amplitude_gated_delta_kuramoto` | 45.7% | 1.309e-03 | True |
| 4 | `adaptive_channel_kuramoto` | 20.4% | 6.570e-03 | True |
| 5 | `multi_channel_delta_kuramoto` | 18.4% | 1.486e-02 | True |
| 6 | `snr_weighted_delta_kuramoto` | 17.5% | 1.614e-02 | True |
| 7 | `sustained_delta_kuramoto` | 17.3% | 2.320e-02 | True |

### cap_multichannel_staging

| Rank | Detector | Detection rate | p-value | Beats chance |
| --- | --- | --- | --- | --- |
| 1 | `normalized_delta_envelope` | 52.5% | 1.000e-03 | True |
| 2 | `multi_channel_delta_kuramoto` | 18.4% | 1.486e-02 | True |
| 3 | `snr_weighted_delta_kuramoto` | 17.5% | 1.614e-02 | True |

### chb01_seizures

| Rank | Detector | Detection rate | p-value | Beats chance |
| --- | --- | --- | --- | --- |
| 1 | `critical_slowing_down` | 33.3% | 2.155e-01 | False |
| 2 | `ensemble_weighted` | 16.7% | 5.582e-01 | False |
| 3 | `synchronisation` | 16.7% | 5.598e-01 | False |
| 4 | `transition_entropy` | 0.0% | 1.000e+00 | False |

### chb01_seizures_multiscale

| Rank | Detector | Detection rate | p-value | Beats chance |
| --- | --- | --- | --- | --- |
| 1 | `critical_slowing_down_multiscale` | 33.3% | 2.154e-01 | False |
| 2 | `critical_slowing_down` | 33.3% | 2.155e-01 | False |
| 3 | `ensemble_weighted` | 16.7% | 5.582e-01 | False |
| 4 | `synchronisation` | 16.7% | 5.598e-01 | False |
| 5 | `transition_entropy` | 0.0% | 1.000e+00 | False |

### csd_variant_synthetic

| Rank | Detector | Detection rate | p-value | Beats chance |
| --- | --- | --- | --- | --- |
| 1 | `critical_slowing_down_multiscale` | 50.0% | 4.055e-03 | True |
| 2 | `critical_slowing_down_surrogate` | 31.2% | 2.078e-02 | True |
| 3 | `critical_slowing_down_baseline` | 25.0% | 1.473e-01 | True |

### dakos_climate_transitions

| Rank | Detector | Detection rate | p-value | Beats chance |
| --- | --- | --- | --- | --- |
| 1 | `critical_slowing_down` | 16.7% | 5.144e-01 | False |

### dakos_climate_transitions_multiscale

| Rank | Detector | Detection rate | p-value | Beats chance |
| --- | --- | --- | --- | --- |
| 1 | `critical_slowing_down_multiscale` | 0.0% | 1.000e+00 | False |

### psml_grid_oscillation

| Rank | Detector | Detection rate | p-value | Beats chance |
| --- | --- | --- | --- | --- |
| 1 | `critical_slowing_down` | 25.0% | 1.953e-01 | False |
| 2 | `transition_entropy` | 16.7% | 4.045e-01 | False |
| 3 | `ensemble_weighted` | 16.7% | 4.073e-01 | False |
| 4 | `synchronisation` | 0.0% | 1.000e+00 | False |

### psml_grid_oscillation_multiscale

| Rank | Detector | Detection rate | p-value | Beats chance |
| --- | --- | --- | --- | --- |
| 1 | `critical_slowing_down_multiscale` | 33.3% | 7.859e-02 | False |
| 2 | `critical_slowing_down` | 25.0% | 1.953e-01 | False |
| 3 | `transition_entropy` | 16.7% | 4.045e-01 | False |
| 4 | `ensemble_weighted` | 16.7% | 4.073e-01 | False |
| 5 | `synchronisation` | 0.0% | 1.000e+00 | False |

### synthetic_honest_audit_demo

| Rank | Detector | Detection rate | p-value | Beats chance |
| --- | --- | --- | --- | --- |
| 1 | `lag1_autocorrelation` | 100.0% | 1.000e-04 | True |
| 2 | `window_mean_control` | 0.0% | 1.000e+00 | False |

## Cross-domain overall ranking

| Rank | Detector | Mean rank | Domains present | Wins | Domain wins |
| --- | --- | --- | --- | --- | --- |
| 1 | `critical_slowing_down_multiscale` | 1.00 | 5 | 5 | afdb_atrial_fibrillation_multiscale (1), chb01_seizures_multiscale (1), csd_variant_synthetic (1), dakos_climate_transitions_multiscale (1), psml_grid_oscillation_multiscale (1) |
| 2 | `coherent_sustained_kuramoto` | 1.00 | 1 | 1 | cap_kuramoto_variants (1) |
| 2 | `lag1_autocorrelation` | 1.00 | 1 | 1 | synthetic_honest_audit_demo (1) |
| 4 | `normalized_delta_envelope` | 1.50 | 2 | 1 | cap_multichannel_staging (1) |
| 5 | `critical_slowing_down` | 2.00 | 7 | 3 | chb01_seizures (1), dakos_climate_transitions (1), psml_grid_oscillation (1) |
| 6 | `critical_slowing_down_surrogate` | 2.00 | 1 | 0 | — |
| 6 | `window_mean_control` | 2.00 | 1 | 0 | — |
| 8 | `ensemble_weighted` | 2.33 | 6 | 2 | afdb_atrial_fibrillation (1), afdb_atrial_fibrillation_multiscale (1) |
| 9 | `synchronisation` | 3.00 | 6 | 2 | afdb_atrial_fibrillation (1), afdb_atrial_fibrillation_multiscale (1) |
| 10 | `amplitude_gated_delta_kuramoto` | 3.00 | 1 | 0 | — |
| 10 | `critical_slowing_down_baseline` | 3.00 | 1 | 0 | — |
| 12 | `multi_channel_delta_kuramoto` | 3.50 | 2 | 0 | — |
| 13 | `transition_entropy` | 3.83 | 6 | 0 | — |
| 14 | `adaptive_channel_kuramoto` | 4.00 | 1 | 0 | — |
| 15 | `snr_weighted_delta_kuramoto` | 4.50 | 2 | 0 | — |
| 16 | `sustained_delta_kuramoto` | 7.00 | 1 | 0 | — |

## Cross-domain patterns

Detectors that appear in more than one domain, sorted by mean rank:

* **`critical_slowing_down_multiscale`** — mean rank 1.00, present in 5 domain(s), wins 5: afdb_atrial_fibrillation_multiscale (1), chb01_seizures_multiscale (1), csd_variant_synthetic (1), dakos_climate_transitions_multiscale (1), psml_grid_oscillation_multiscale (1).
* **`normalized_delta_envelope`** — mean rank 1.50, present in 2 domain(s), wins 1: cap_kuramoto_variants (2), cap_multichannel_staging (1).
* **`critical_slowing_down`** — mean rank 2.00, present in 7 domain(s), wins 3: afdb_atrial_fibrillation (3), afdb_atrial_fibrillation_multiscale (4), chb01_seizures (1), chb01_seizures_multiscale (2), dakos_climate_transitions (1), psml_grid_oscillation (1), psml_grid_oscillation_multiscale (2).
* **`ensemble_weighted`** — mean rank 2.33, present in 6 domain(s), wins 2: afdb_atrial_fibrillation (1), afdb_atrial_fibrillation_multiscale (1), chb01_seizures (2), chb01_seizures_multiscale (3), psml_grid_oscillation (3), psml_grid_oscillation_multiscale (4).
* **`synchronisation`** — mean rank 3.00, present in 6 domain(s), wins 2: afdb_atrial_fibrillation (1), afdb_atrial_fibrillation_multiscale (1), chb01_seizures (3), chb01_seizures_multiscale (4), psml_grid_oscillation (4), psml_grid_oscillation_multiscale (5).
* **`multi_channel_delta_kuramoto`** — mean rank 3.50, present in 2 domain(s), wins 0: cap_kuramoto_variants (5), cap_multichannel_staging (2).
* **`transition_entropy`** — mean rank 3.83, present in 6 domain(s), wins 0: afdb_atrial_fibrillation (4), afdb_atrial_fibrillation_multiscale (5), chb01_seizures (4), chb01_seizures_multiscale (5), psml_grid_oscillation (2), psml_grid_oscillation_multiscale (3).
* **`snr_weighted_delta_kuramoto`** — mean rank 4.50, present in 2 domain(s), wins 0: cap_kuramoto_variants (6), cap_multichannel_staging (3).

## Ranked refinement backlog

1. **Advance the `critical_slowing_down_multiscale` variant** — it wins in 5 early-warning domain(s) (afdb_atrial_fibrillation_multiscale, chb01_seizures_multiscale, csd_variant_synthetic, dakos_climate_transitions_multiscale, psml_grid_oscillation_multiscale) and has the best mean rank (1.00) among detectors present in multiple domains. Extend it to the remaining real-data domains (EEG, cardiac) and compare it head-to-head with the baseline CSD on every corpus.
2. **Protect and productise `normalized_delta_envelope` for CAP sleep staging** — it dominates its domain (mean rank 1.50) and should become the default reference detector there.
3. **Protect and productise `lag1_autocorrelation` for synthetic critical-slowing-down corpus** — it dominates its domain (mean rank 1.00) and should become the default reference detector there.
4. **Deprioritise SNR-weighted Kuramoto** — in the CAP multichannel panel it does not outperform the unweighted multi-channel Kuramoto detector. Reallocate effort toward channel-selection or coupling-structure variants rather than a raw SNR weighting.
5. **Audit the `ensemble_weighted` fusion rule** — it is present in 6 early-warning domains but rarely wins (mean rank 2.33). Investigate whether the current weighting is dominated by a single indicator and whether a learned combination would help.
6. **Study `coherent_sustained_kuramoto` transferability** — it wins in its single domain but has not been tested elsewhere; run the same pipeline on a second domain to judge generality.
7. **Re-evaluate `critical_slowing_down_surrogate`** — it appears in only one domain and never wins; consider whether the feature is under-powered or simply unsuited to that data regime.
8. **Re-evaluate `window_mean_control`** — it appears in only one domain and never wins; consider whether the feature is under-powered or simply unsuited to that data regime.
9. **Re-evaluate `amplitude_gated_delta_kuramoto`** — it appears in only one domain and never wins; consider whether the feature is under-powered or simply unsuited to that data regime.
10. **Re-evaluate `critical_slowing_down_baseline`** — it appears in only one domain and never wins; consider whether the feature is under-powered or simply unsuited to that data regime.
11. **Re-evaluate `adaptive_channel_kuramoto`** — it appears in only one domain and never wins; consider whether the feature is under-powered or simply unsuited to that data regime.
12. **Re-evaluate `sustained_delta_kuramoto`** — it appears in only one domain and never wins; consider whether the feature is under-powered or simply unsuited to that data regime.

## Notes

* Detection rate for early-warning aggregates is approximated by ``observed_led / n_transitions`` — the fraction of transitions for which the detector produced a statistically meaningful lead.
* A detector is marked as *beating chance* when its reported p-value is below 0.05; honest-audit aggregates additionally report the committed ``fraction_beats_chance`` value.
* The CAP multichannel finding that **SNR-weighted Kuramoto did not improve** over the simple mean-R Kuramoto detector is carried forward explicitly; further investment in that exact spatial-R feature is not supported by the current evidence.
