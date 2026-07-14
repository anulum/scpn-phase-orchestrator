<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->

# Monitor — Adaptive Multi-Channel Kuramoto

The `monitor.adaptive_kuramoto` module provides a robust, quality-weighted
multi-channel Kuramoto order-parameter detector. It was introduced to address
the heterogeneity seen in the CAP Sleep Database multi-channel N3-vs-Wake
audit, where the simple mean-R Kuramoto detector wins on some recordings and
collapses on others.

## Core idea

For each channel and each epoch the detector computes a data-driven quality
weight:

- **Reward** delta-band SNR (channels that actually carry slow-wave activity).
- **Penalise** excess kurtosis (a proxy for transients, muscle artefacts, and
  other non-oscillatory bursts).

The weighted Kuramoto order parameter

$$
R(t) = \frac{\bigl| \sum_c w_c(t) \, e^{i \phi_c(t)} \bigr|}{\sum_c w_c(t)}
$$

is then pooled per epoch with the **median** rather than the mean, making the
score less sensitive to brief artefacts.

## API

```python
from scpn_phase_orchestrator.monitor.adaptive_kuramoto import (
    compute_adaptive_kuramoto_scores,
    compute_channel_quality_weights,
    compute_weighted_kuramoto_r,
)
```

### `compute_adaptive_kuramoto_scores`

```python
def compute_adaptive_kuramoto_scores(
    data: NDArray[np.float64],
    fs: float,
    band_hz: tuple[float, float] = (0.5, 4.0),
    epoch_seconds: float = 30.0,
    kurtosis_penalty_scale: float = 0.2,
    score_precision: int = 6,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

Returns per-epoch scores and per-channel/per-epoch weights for a multi-channel
signal. `data` must have shape `(n_channels, n_samples)` with `n_channels >= 2`
and at least one full epoch of samples.

### `compute_channel_quality_weights`

```python
def compute_channel_quality_weights(
    data: NDArray[np.float64],
    fs: float,
    band_hz: tuple[float, float] = (0.5, 4.0),
    epoch_seconds: float = 30.0,
    kurtosis_penalty_scale: float = 0.2,
) -> NDArray[np.float64]: ...
```

Returns weights of shape `(n_channels, n_epochs)` that sum to one per epoch.

### `compute_weighted_kuramoto_r`

```python
def compute_weighted_kuramoto_r(
    phases: NDArray[np.float64],
    weights: NDArray[np.float64],
    epoch_seconds: float,
    fs: float,
) -> NDArray[np.float64]: ...
```

Computes the median-pooled weighted Kuramoto R from pre-computed phases.

## Design rationale

The CAP diagnostic study (`docs/studies/cap_kuramoto_diagnostic.md`) found that
the simple mean-R detector collapses when N3 and Wake epochs have similar mean
phase coherence (e.g. `brux2`), and that SNR-weighting alone does not fully fix
the problem. The adaptive module adds two mechanisms:

1. **Channel-quality weighting** goes beyond SNR by also rejecting channels
   with high excess kurtosis, a cheap but effective artefact detector.
2. **Robust temporal pooling** replaces the epoch mean with the median, so a
   single high-R artefact within an epoch cannot dominate the score.

## Honesty boundaries

- The detector is **not** a clinical sleep-staging product. It is a research
  detector audited under the matched-false-alarm protocol in
  `bench/cap_multichannel_n3_vs_wake.py`.
- Channel weights are fit **per recording**, so cross-recording generalisation
  must be evaluated empirically; the module makes no claim of universal
  superiority over the delta envelope.

## Tests

`tests/test_adaptive_kuramoto.py` covers:

- Quality weights reward high-SNR channels.
- Scores are higher for coherent epochs than incoherent epochs.
- Input validation rejects single-channel and too-short signals.
- Output shapes and bounds are correct.

## Related

- [CAP multi-channel N3 vs Wake audit](../../studies/cap_multichannel_n3_vs_wake.md)
- [CAP Kuramoto diagnostic study](../../studies/cap_kuramoto_diagnostic.md)
- [Monitor overview](monitor.md)

::: scpn_phase_orchestrator.monitor.adaptive_kuramoto
