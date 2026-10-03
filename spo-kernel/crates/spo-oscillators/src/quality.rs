// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Phase quality scorer

/// Aggregate quality scoring and collapse detection.
#[derive(Clone, Debug)]
pub struct PhaseQualityScorer {
    pub collapse_threshold: f64,
    pub min_quality: f64,
}

impl Default for PhaseQualityScorer {
    fn default() -> Self {
        Self {
            collapse_threshold: 0.1,
            min_quality: 0.3,
        }
    }
}

impl PhaseQualityScorer {
    /// Amplitude-weighted mean of finite pairs, with overflow-safe weight scaling.
    ///
    /// # Arguments
    ///
    /// * `qualities` - Per-oscillator quality values, clamped to [0, 1].
    /// * `amplitudes` - Amplitude weights, floored at 1e-12.
    ///
    /// # Returns
    ///
    /// The mean of the matching prefix, skipping any pair containing a
    /// nonfinite measurement. Empty/no usable pairs or nonfinite configured
    /// thresholds return zero. Weights are divided by their finite maximum
    /// before summation; an unscaled total above f64::MAX remains admissible.
    #[must_use]
    pub fn score(&self, qualities: &[f64], amplitudes: &[f64]) -> f64 {
        if qualities.is_empty() {
            return 0.0;
        }
        if !self.collapse_threshold.is_finite() || !self.min_quality.is_finite() {
            return 0.0;
        }
        let pairs = qualities
            .iter()
            .zip(amplitudes)
            .filter(|(q, amp)| q.is_finite() && amp.is_finite());
        let scale = pairs
            .clone()
            .map(|(_, amp)| amp.max(1e-12))
            .fold(0.0, f64::max);
        if scale == 0.0 {
            return 0.0;
        }
        let (wsum, total_w) = pairs.fold((0.0, 0.0), |(ws, tw), (&q, amp)| {
            let q = q.clamp(0.0, 1.0);
            let w = amp.max(1e-12) / scale;
            (ws + q * w, tw + w)
        });
        wsum / total_w
    }

    /// True if quality is below threshold for the majority of states.
    #[must_use]
    pub fn is_collapsed(&self, qualities: &[f64]) -> bool {
        if qualities.is_empty() {
            return true;
        }
        if !self.collapse_threshold.is_finite() {
            return true;
        }
        let below = qualities
            .iter()
            .filter(|&&q| !q.is_finite() || q < self.collapse_threshold)
            .count();
        below > qualities.len() / 2
    }

    /// Weight array: qualities above min_quality pass through, others zeroed.
    #[must_use]
    pub fn downweight_mask(&self, qualities: &[f64]) -> Vec<f64> {
        if !self.min_quality.is_finite() {
            return vec![0.0; qualities.len()];
        }
        qualities
            .iter()
            .map(|&q| {
                if q.is_finite() && q >= self.min_quality {
                    q.clamp(0.0, 1.0)
                } else {
                    0.0
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_uniform() {
        let s = PhaseQualityScorer::default();
        let q = vec![0.8, 0.8, 0.8];
        let a = vec![1.0, 1.0, 1.0];
        assert!((s.score(&q, &a) - 0.8).abs() < 1e-12);
    }

    #[test]
    fn score_weighted() {
        let s = PhaseQualityScorer::default();
        let q = vec![1.0, 0.0];
        let a = vec![2.0, 1.0];
        // (1.0*2 + 0.0*1) / (2+1) = 2/3
        assert!((s.score(&q, &a) - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn score_empty() {
        assert_eq!(PhaseQualityScorer::default().score(&[], &[]), 0.0);
    }

    #[test]
    fn collapse_all_low() {
        let s = PhaseQualityScorer::default();
        assert!(s.is_collapsed(&[0.01, 0.02, 0.03]));
    }

    #[test]
    fn collapse_all_high() {
        let s = PhaseQualityScorer::default();
        assert!(!s.is_collapsed(&[0.8, 0.9, 0.7]));
    }

    #[test]
    fn collapse_empty() {
        assert!(PhaseQualityScorer::default().is_collapsed(&[]));
    }

    #[test]
    fn downweight_mask_filters() {
        let s = PhaseQualityScorer::default();
        let mask = s.downweight_mask(&[0.1, 0.5, 0.3, 0.9]);
        assert_eq!(mask[0], 0.0);
        assert_eq!(mask[1], 0.5);
        assert_eq!(mask[2], 0.3);
        assert_eq!(mask[3], 0.9);
    }

    #[test]
    fn score_ignores_non_finite_inputs() {
        let s = PhaseQualityScorer::default();
        let q = vec![0.8, f64::NAN, 0.2];
        let a = vec![1.0, 1.0, f64::INFINITY];
        assert!((s.score(&q, &a) - 0.8).abs() < 1e-12);
    }

    #[test]
    fn non_finite_thresholds_fail_closed() {
        let s = PhaseQualityScorer {
            collapse_threshold: f64::NAN,
            min_quality: f64::INFINITY,
        };
        assert_eq!(s.score(&[0.5], &[1.0]), 0.0);
        assert!(s.is_collapsed(&[0.9, 0.8]));
        assert_eq!(s.downweight_mask(&[0.9, 0.8]), vec![0.0, 0.0]);
    }

    #[test]
    fn finite_weights_above_representable_total_preserve_mean() {
        let scorer = PhaseQualityScorer::default();
        for amplitudes in [[1e308, 1e308], [f64::MAX, f64::MAX]] {
            assert!((scorer.score(&[0.8, 0.2], &amplitudes) - 0.5).abs() < 1e-15);
            assert_eq!(scorer.score(&[1.0, 1.0], &amplitudes), 1.0);
            assert_eq!(scorer.score(&[0.0, 0.0], &amplitudes), 0.0);
        }
        let score = scorer.score(&[0.9, 0.1, 0.5], &[1e308, 5e307, 2.5e307]);
        assert!((score - 4.3 / 7.0).abs() < 1e-15);
    }

    #[test]
    fn matching_prefix_and_amplitude_floor_remain_defined() {
        let scorer = PhaseQualityScorer::default();
        assert_eq!(scorer.score(&[0.8, 0.2], &[0.0, -f64::MAX]), 0.5);
        assert_eq!(scorer.score(&[0.8, 0.2], &[1e308]), 0.8);
        assert_eq!(scorer.score(&[0.8], &[1e308, f64::NAN]), 0.8);
        assert_eq!(scorer.score(&[0.8], &[]), 0.0);
        assert_eq!(scorer.score(&[f64::NAN], &[1.0]), 0.0);
        assert_eq!(scorer.score(&[0.0, 1.0], &[1e308, 1.0]), 1e-308);
    }

    #[test]
    fn configured_thresholds_keep_strict_majority_and_boundary_mask() {
        let scorer = PhaseQualityScorer {
            collapse_threshold: 0.7,
            min_quality: 0.4,
        };
        assert!(scorer.is_collapsed(&[0.1, 0.6, 0.9]));
        assert!(!scorer.is_collapsed(&[0.1, 0.9]));
        assert!(!scorer.is_collapsed(&[0.7]));
        assert_eq!(
            scorer.downweight_mask(&[0.1, 0.4, 0.6, 1.5, f64::NAN]),
            vec![0.0, 0.4, 0.6, 1.0, 0.0]
        );
    }
}
