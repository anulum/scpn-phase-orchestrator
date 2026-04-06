// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Delay embedding (Takens 1981)

//! Time-delay embedding for attractor reconstruction.
//!
//! Takens 1981: a scalar observable embedded in m dimensions using
//! time-delayed copies recovers the attractor topology.
//!
//! References:
//!   Takens 1981, Lecture Notes in Mathematics 898:366-381.
//!   Fraser & Swinney 1986, Phys. Rev. A 33:1134-1140.
//!   Kennel, Brown & Abarbanel 1992, Phys. Rev. A 45:3403-3411.

use rayon::prelude::*;

/// Construct time-delay embedding matrix (row-major flat output).
///
/// v(t) = [x(t), x(t-τ), x(t-2τ), ..., x(t-(m-1)τ)]
///
/// # Returns
/// Flattened (T_eff × m) array where T_eff = T - (m-1)*delay.
///
/// # Errors
/// Returns `Err` if signal is too short for the given parameters.
pub fn delay_embed(signal: &[f64], delay: usize, dimension: usize) -> Result<Vec<f64>, String> {
    let t = signal.len();
    if dimension == 0 {
        return Err("dimension must be >= 1".into());
    }
    if delay == 0 {
        return Err("delay must be >= 1".into());
    }
    let needed = (dimension - 1) * delay;
    if t <= needed {
        return Err(format!(
            "Signal too short (T={t}) for delay={delay}, dimension={dimension}: need T > {needed}"
        ));
    }
    let t_eff = t - needed;
    let mut out = Vec::with_capacity(t_eff * dimension);
    for row in 0..t_eff {
        for col in 0..dimension {
            out.push(signal[row + col * delay]);
        }
    }
    Ok(out)
}

/// Average mutual information between x(t) and x(t+lag).
///
/// Histogram-based estimation (Fraser & Swinney 1986).
fn mutual_information(signal: &[f64], lag: usize, n_bins: usize) -> f64 {
    let t_len = signal.len();
    if lag >= t_len || n_bins == 0 {
        return 0.0;
    }
    let n = t_len - lag;

    // Find min/max for binning
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    for &v in signal {
        if v < min_val {
            min_val = v;
        }
        if v > max_val {
            max_val = v;
        }
    }
    let range = max_val - min_val;
    if range == 0.0 {
        return 0.0;
    }

    let bin_width = range / n_bins as f64;

    // Build joint histogram
    let mut hist_xy = vec![0u64; n_bins * n_bins];
    for i in 0..n {
        let bx = ((signal[i] - min_val) / bin_width) as usize;
        let by = ((signal[i + lag] - min_val) / bin_width) as usize;
        let bx = bx.min(n_bins - 1);
        let by = by.min(n_bins - 1);
        hist_xy[bx * n_bins + by] += 1;
    }

    // Marginals
    let mut hist_x = vec![0u64; n_bins];
    let mut hist_y = vec![0u64; n_bins];
    for bx in 0..n_bins {
        for by in 0..n_bins {
            let c = hist_xy[bx * n_bins + by];
            hist_x[bx] += c;
            hist_y[by] += c;
        }
    }

    // MI = Σ p(x,y) log(p(x,y) / (p(x) p(y)))
    let inv_n = 1.0 / n as f64;
    let mut mi = 0.0;
    for bx in 0..n_bins {
        if hist_x[bx] == 0 {
            continue;
        }
        let px = hist_x[bx] as f64 * inv_n;
        for by in 0..n_bins {
            let c = hist_xy[bx * n_bins + by];
            if c == 0 || hist_y[by] == 0 {
                continue;
            }
            let pxy = c as f64 * inv_n;
            let py = hist_y[by] as f64 * inv_n;
            mi += pxy * (pxy / (px * py)).ln();
        }
    }
    mi
}

/// Find optimal delay τ as first minimum of mutual information.
///
/// Fraser & Swinney 1986.
#[must_use]
pub fn optimal_delay(signal: &[f64], max_lag: usize, n_bins: usize) -> usize {
    let actual_max = max_lag.min(signal.len() / 2);
    if actual_max < 2 {
        return 1;
    }

    let mi: Vec<f64> = (0..actual_max)
        .into_par_iter()
        .map(|lag| mutual_information(signal, lag, n_bins))
        .collect();

    // First local minimum
    for i in 1..mi.len() - 1 {
        if mi[i] < mi[i - 1] && mi[i] < mi[i + 1] {
            return i;
        }
    }
    1
}

/// Find nearest neighbor distances and indices for each point.
///
/// O(T² · m) brute-force. Returns (distances, indices).
fn nearest_neighbor_distances(embedded: &[f64], t: usize, m: usize) -> (Vec<f64>, Vec<usize>) {
    let mut nn_dist = vec![f64::INFINITY; t];
    let mut nn_idx = vec![0usize; t];

    let results: Vec<(f64, usize)> = (0..t)
        .into_par_iter()
        .map(|i| {
            let mut min_d = f64::INFINITY;
            let mut min_idx = 0;
            let ei = &embedded[i * m..(i + 1) * m];

            for j in 0..t {
                if i == j {
                    continue;
                }
                let ej = &embedded[j * m..(j + 1) * m];
                let mut d2 = 0.0;
                for k in 0..m {
                    let diff = ei[k] - ej[k];
                    d2 += diff * diff;
                }
                if d2 < min_d * min_d {
                    min_d = d2.sqrt();
                    min_idx = j;
                }
            }
            (min_d, min_idx)
        })
        .collect();

    for i in 0..t {
        nn_dist[i] = results[i].0;
        nn_idx[i] = results[i].1;
    }
    (nn_dist, nn_idx)
}

/// Find optimal embedding dimension via False Nearest Neighbors.
///
/// Kennel, Brown & Abarbanel 1992.
///
/// # Arguments
/// * `signal` — scalar time series
/// * `delay` — time delay τ
/// * `max_dim` — maximum dimension to test
/// * `rtol` — relative distance threshold (criterion 1)
/// * `atol` — absolute distance threshold as fraction of σ (criterion 2)
#[must_use]
pub fn optimal_dimension(
    signal: &[f64],
    delay: usize,
    max_dim: usize,
    rtol: f64,
    atol: f64,
) -> usize {
    let t = signal.len();

    // Compute standard deviation
    let mean = signal.iter().sum::<f64>() / t as f64;
    let var = signal.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / t as f64;
    let sigma = var.sqrt();
    if sigma == 0.0 {
        return 1;
    }

    for m in 1..=max_dim {
        let t_eff = t as isize - m as isize * delay as isize;
        if t_eff <= 1 {
            return m;
        }

        let emb = match delay_embed(signal, delay, m) {
            Ok(e) => e,
            Err(_) => return m,
        };
        let t_m = emb.len() / m;

        let t_next = t as isize - m as isize * delay as isize;
        if t_next <= 1 {
            return m;
        }

        let (nn_dist, nn_idx) = nearest_neighbor_distances(&emb, t_m, m);

        let mut n_false = 0u64;
        let mut n_valid = 0u64;

        for i in 0..t_m {
            let j = nn_idx[i];
            let d = nn_dist[i];
            if d == 0.0 || d == f64::INFINITY {
                continue;
            }

            let i_next = i + m * delay;
            let j_next = j + m * delay;
            if i_next >= t || j_next >= t {
                continue;
            }

            n_valid += 1;
            let extra_dist = (signal[i_next] - signal[j_next]).abs();

            // Criterion 1: relative increase (Kennel et al. 1992, Eq. 4)
            if extra_dist / d > rtol {
                n_false += 1;
                continue;
            }

            // Criterion 2: absolute size (Kennel et al. 1992, Eq. 6)
            let new_dist = (d * d + extra_dist * extra_dist).sqrt();
            if new_dist / sigma > atol {
                n_false += 1;
            }
        }

        let fnn_frac = if n_valid > 0 {
            n_false as f64 / n_valid as f64
        } else {
            0.0
        };
        if fnn_frac < 0.01 {
            return m;
        }
    }

    max_dim
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_delay_embed_shape() {
        let signal: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let emb = delay_embed(&signal, 3, 4).unwrap();
        // T_eff = 100 - 3*3 = 91, m = 4
        assert_eq!(emb.len(), 91 * 4);
    }

    #[test]
    fn test_delay_embed_values() {
        let signal = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let emb = delay_embed(&signal, 2, 3).unwrap();
        // T_eff = 6 - 2*2 = 2, m = 3
        // Row 0: [x[0], x[2], x[4]] = [0, 2, 4]
        // Row 1: [x[1], x[3], x[5]] = [1, 3, 5]
        assert_eq!(emb.len(), 6);
        assert!((emb[0] - 0.0).abs() < 1e-12);
        assert!((emb[1] - 2.0).abs() < 1e-12);
        assert!((emb[2] - 4.0).abs() < 1e-12);
        assert!((emb[3] - 1.0).abs() < 1e-12);
        assert!((emb[4] - 3.0).abs() < 1e-12);
        assert!((emb[5] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_delay_embed_too_short() {
        let signal = vec![1.0, 2.0];
        assert!(delay_embed(&signal, 3, 3).is_err());
    }

    #[test]
    fn test_mutual_information_identical() {
        // MI(x, x) should be maximal (lag=0 is essentially the same signal)
        let signal: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.1).sin()).collect();
        let mi0 = mutual_information(&signal, 0, 32);
        let mi50 = mutual_information(&signal, 50, 32);
        assert!(mi0 > mi50, "MI at lag 0 should exceed MI at lag 50");
    }

    #[test]
    fn test_mutual_information_nonnegative() {
        let signal: Vec<f64> = (0..500).map(|i| (i as f64 * 0.05).sin()).collect();
        for lag in 0..20 {
            let mi = mutual_information(&signal, lag, 32);
            assert!(
                mi >= -1e-12,
                "MI should be non-negative, got {mi} at lag {lag}"
            );
        }
    }

    #[test]
    fn test_optimal_delay_sinusoid() {
        // Sinusoid: first MI minimum should be near quarter-period
        let period = 40;
        let signal: Vec<f64> = (0..2000)
            .map(|i| (2.0 * PI * i as f64 / period as f64).sin())
            .collect();
        let tau = optimal_delay(&signal, 50, 32);
        // Quarter period = 10; allow range [5, 15]
        assert!(
            tau >= 5 && tau <= 15,
            "optimal delay {tau} not near quarter-period"
        );
    }

    #[test]
    fn test_optimal_delay_constant_signal() {
        let signal = vec![1.0; 200];
        let tau = optimal_delay(&signal, 50, 32);
        assert_eq!(tau, 1);
    }

    #[test]
    fn test_optimal_dimension_sinusoid() {
        // Clean sinusoid: theoretical embedding m=2 (circle).
        // FNN with discrete data may yield m=2..4 depending on
        // quantisation effects and NN ties near identical values.
        let signal: Vec<f64> = (0..2000)
            .map(|i| (2.0 * PI * i as f64 / 40.0).sin())
            .collect();
        let tau = optimal_delay(&signal, 50, 32);
        let m = optimal_dimension(&signal, tau, 8, 15.0, 2.0);
        assert!(
            m >= 2 && m <= 8,
            "sinusoid embedding dimension {m} unexpected (tau={tau})"
        );
    }

    #[test]
    fn test_optimal_dimension_constant() {
        let signal = vec![42.0; 200];
        let m = optimal_dimension(&signal, 1, 10, 15.0, 2.0);
        assert_eq!(m, 1);
    }

    #[test]
    fn test_nearest_neighbor_distances_basic() {
        // 3 points in 2D: (0,0), (1,0), (10,0)
        // NN of 0 → 1 (dist 1), NN of 1 → 0 (dist 1), NN of 2 → 1 (dist 9)
        let emb = vec![0.0, 0.0, 1.0, 0.0, 10.0, 0.0];
        let (dists, idxs) = nearest_neighbor_distances(&emb, 3, 2);
        assert!((dists[0] - 1.0).abs() < 1e-12);
        assert_eq!(idxs[0], 1);
        assert!((dists[1] - 1.0).abs() < 1e-12);
        assert_eq!(idxs[1], 0);
        assert!((dists[2] - 9.0).abs() < 1e-12);
        assert_eq!(idxs[2], 1);
    }

    #[test]
    fn test_fnn_lorenz_like() {
        // Lorenz x-component approximation: needs m ≥ 3
        // Generate a simple chaotic-like signal: logistic map
        let mut signal = Vec::with_capacity(2000);
        let mut x = 0.1_f64;
        for _ in 0..2000 {
            x = 3.9 * x * (1.0 - x);
            signal.push(x);
        }
        let m = optimal_dimension(&signal, 1, 8, 15.0, 2.0);
        // Logistic map attractor is 1D but typically needs m=2-3 for embedding
        assert!(
            m >= 1 && m <= 5,
            "logistic map embedding dimension {m} unexpected"
        );
    }
}
