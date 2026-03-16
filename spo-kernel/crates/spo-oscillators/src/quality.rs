// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
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
    /// Weighted average quality. `qualities` and `amplitudes` must be same length.
    ///
    /// # Arguments
    ///
    /// * `qualities` - Per-oscillator quality values in [0, 1].
    /// * `amplitudes` - Per-oscillator amplitudes used as weights.
    #[must_use]
    pub fn score(&self, qualities: &[f64], amplitudes: &[f64]) -> f64 {
        if qualities.is_empty() {
            return 0.0;
        }
        let n = qualities.len().min(amplitudes.len());
        let (wsum, total_w) = (0..n).fold((0.0, 0.0), |(ws, tw), i| {
            let w = amplitudes[i].max(1e-12);
            (ws + qualities[i] * w, tw + w)
        });
        if total_w <= 0.0 {
            return 0.0;
        }
        wsum / total_w
    }

    /// True if quality is below threshold for the majority of states.
    #[must_use]
    pub fn is_collapsed(&self, qualities: &[f64]) -> bool {
        if qualities.is_empty() {
            return true;
        }
        let below = qualities
            .iter()
            .filter(|&&q| q < self.collapse_threshold)
            .count();
        below > qualities.len() / 2
    }

    /// Weight array: qualities above min_quality pass through, others zeroed.
    #[must_use]
    pub fn downweight_mask(&self, qualities: &[f64]) -> Vec<f64> {
        qualities
            .iter()
            .map(|&q| if q >= self.min_quality { q } else { 0.0 })
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
}
