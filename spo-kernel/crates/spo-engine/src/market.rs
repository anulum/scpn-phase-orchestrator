// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Financial market Kuramoto synchronisation

//! Kuramoto-based financial market synchronisation analysis.
//!
//! R(t) → 1 precedes market crashes (Black Monday 1987, 2008 crisis).
//! arXiv:1109.1167; CEUR-WS Vol-915.

/// Kuramoto order parameter R(t) across assets at each timestep.
///
/// R(t) = |<exp(iθ)>| where average is over N assets.
///
/// # Arguments
/// * `phases_flat` — row-major (T × N) phase matrix
/// * `t` — number of timesteps
/// * `n` — number of assets
#[must_use]
pub fn market_order_parameter(phases_flat: &[f64], t: usize, n: usize) -> Vec<f64> {
    if n == 0 || t == 0 {
        return vec![];
    }
    let inv_n = 1.0 / n as f64;
    let mut result = Vec::with_capacity(t);
    for row in 0..t {
        let mut sum_cos = 0.0;
        let mut sum_sin = 0.0;
        for col in 0..n {
            let theta = phases_flat[row * n + col];
            sum_cos += theta.cos();
            sum_sin += theta.sin();
        }
        let mc = sum_cos * inv_n;
        let ms = sum_sin * inv_n;
        result.push((mc * mc + ms * ms).sqrt());
    }
    result
}

/// Windowed Phase-Locking Value matrix between assets.
///
/// Returns flattened (n_windows × N × N) PLV matrices.
#[must_use]
pub fn market_plv(phases_flat: &[f64], t: usize, n: usize, window: usize) -> Vec<f64> {
    if t < window || n == 0 || window == 0 {
        return vec![];
    }
    let n_windows = t - window + 1;
    let mut plv = vec![0.0; n_windows * n * n];
    let inv_w = 1.0 / window as f64;

    for w in 0..n_windows {
        for i in 0..n {
            for j in 0..n {
                let mut sum_cos = 0.0;
                let mut sum_sin = 0.0;
                for k in 0..window {
                    let diff = phases_flat[(w + k) * n + i] - phases_flat[(w + k) * n + j];
                    sum_cos += diff.cos();
                    sum_sin += diff.sin();
                }
                let mc = sum_cos * inv_w;
                let ms = sum_sin * inv_w;
                plv[w * n * n + i * n + j] = (mc * mc + ms * ms).sqrt();
            }
        }
    }
    plv
}

/// Classify market synchronisation regimes from R(t).
///
/// 0=desync, 1=transition, 2=synchronised.
#[must_use]
pub fn detect_regimes(r: &[f64], sync_threshold: f64, desync_threshold: f64) -> Vec<i32> {
    r.iter()
        .map(|&v| {
            if v >= sync_threshold {
                2
            } else if v <= desync_threshold {
                0
            } else {
                1
            }
        })
        .collect()
}

/// Detect synchronisation warning signals (R crossing threshold from below).
#[must_use]
pub fn sync_warning(r: &[f64], threshold: f64, lookback: usize) -> Vec<bool> {
    if r.is_empty() {
        return vec![];
    }

    // Smooth if lookback > 1
    let smoothed = if lookback > 1 {
        let mut s = vec![0.0; r.len()];
        for (i, val) in s.iter_mut().enumerate() {
            let start = i.saturating_sub(lookback / 2);
            let end = (i + lookback / 2 + 1).min(r.len());
            let sum: f64 = r[start..end].iter().sum();
            *val = sum / (end - start) as f64;
        }
        s
    } else {
        r.to_vec()
    };

    let mut warnings = vec![false; r.len()];
    for t in 1..r.len() {
        if smoothed[t] >= threshold && smoothed[t - 1] < threshold {
            warnings[t] = true;
        }
    }
    warnings
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    #[test]
    fn test_order_parameter_sync() {
        // All assets same phase → R = 1
        let phases = vec![1.0; 5 * 3]; // 5 timesteps, 3 assets
        let r = market_order_parameter(&phases, 5, 3);
        assert_eq!(r.len(), 5);
        for v in &r {
            assert!((v - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_order_parameter_desync() {
        // Uniformly spread phases → R ≈ 0
        let n = 100;
        let mut phases = Vec::with_capacity(n);
        for i in 0..n {
            phases.push(TAU * i as f64 / n as f64);
        }
        let r = market_order_parameter(&phases, 1, n);
        assert!(r[0] < 0.05);
    }

    #[test]
    fn test_plv_diagonal_one() {
        // PLV(i,i) = 1 always
        let t = 20;
        let n = 3;
        let window = 5;
        let mut phases = Vec::with_capacity(t * n);
        let mut x: u64 = 42;
        for _ in 0..t * n {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
            phases.push((x >> 33) as f64 / (1u64 << 31) as f64 * TAU);
        }
        let plv = market_plv(&phases, t, n, window);
        let n_windows = t - window + 1;
        for w in 0..n_windows {
            for i in 0..n {
                let val = plv[w * n * n + i * n + i];
                assert!((val - 1.0).abs() < 1e-10, "PLV[{w},{i},{i}] = {val}");
            }
        }
    }

    #[test]
    fn test_plv_bounded() {
        let t = 30;
        let n = 4;
        let window = 10;
        let mut phases = Vec::with_capacity(t * n);
        for i in 0..t * n {
            phases.push((i as f64 * 0.3).sin() * 5.0);
        }
        let plv = market_plv(&phases, t, n, window);
        for &v in &plv {
            assert!(v >= 0.0 && v <= 1.0 + 1e-10, "PLV = {v} out of [0,1]");
        }
    }

    #[test]
    fn test_detect_regimes() {
        let r = vec![0.1, 0.5, 0.8, 0.2, 0.9];
        let regimes = detect_regimes(&r, 0.7, 0.3);
        assert_eq!(regimes, vec![0, 1, 2, 0, 2]);
    }

    #[test]
    fn test_sync_warning_crossing() {
        let r = vec![0.3, 0.5, 0.6, 0.8, 0.9, 0.4];
        let w = sync_warning(&r, 0.7, 1);
        assert!(!w[0] && !w[1] && !w[2]);
        assert!(w[3]); // crosses 0.7 from below
        assert!(!w[4]); // already above
    }

    #[test]
    fn test_sync_warning_empty() {
        assert!(sync_warning(&[], 0.7, 1).is_empty());
    }

    #[test]
    fn test_market_order_empty() {
        assert!(market_order_parameter(&[], 0, 0).is_empty());
    }

    #[test]
    fn test_plv_window_too_large() {
        let phases = vec![0.0; 10];
        assert!(market_plv(&phases, 2, 5, 10).is_empty());
    }
}
