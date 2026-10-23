// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Fractal dimension (Grassberger & Procaccia 1983, Kaplan & Yorke 1979)

//! Fractal dimension estimation for phase-space trajectories.
//!
//! - Correlation dimension D₂ via Grassberger-Procaccia algorithm.
//! - Kaplan-Yorke dimension D_KY from Lyapunov spectrum.
//!
//! References:
//! - Grassberger & Procaccia 1983, Phys. Rev. Lett. 50:346-349.
//! - Kaplan & Yorke 1979, Lecture Notes in Mathematics 730:228-237.
//! - Theiler 1987, Phys. Rev. A 36:4456-4462 (Theiler correction).

use rayon::prelude::*;

/// Correlation integral C(ε) = fraction of point pairs within distance ε.
///
/// Grassberger & Procaccia 1983, Eq. 1.
/// Evaluates all T(T-1)/2 unique pairs when they fit `max_pairs`.
/// Otherwise draws the requested valid-pair budget with the legacy threaded
/// LCG, retrying diagonal pairs. Counts use distance strictly less than ε.
///
/// # Arguments
/// * `trajectory` - row-major (T × d) flattened trajectory
/// * `t` - number of time points
/// * `d` - embedding dimension
/// * `epsilons` - (K,) distance thresholds, must be sorted ascending
/// * `max_pairs` - maximum number of pairs to evaluate
/// * `seed` - PRNG seed for subsampling
///
/// # Returns
/// (K,) correlation integral values C(ε) ∈ [0, 1].
///
/// # Errors
/// Refuses invalid shape/count arithmetic, nonfinite coordinates and invalid
/// or unsorted finite radii before indexing or sampling.
pub fn correlation_integral(
    trajectory: &[f64],
    t: usize,
    d: usize,
    epsilons: &[f64],
    max_pairs: usize,
    seed: u64,
) -> Result<Vec<f64>, String> {
    validate_trajectory(trajectory, t, d)?;
    if max_pairs == 0 {
        return Err("max_pairs must be positive".into());
    }
    if epsilons.iter().any(|x| !x.is_finite() || *x < 0.0)
        || epsilons.windows(2).any(|x| x[0] > x[1])
    {
        return Err("epsilons must be sorted finite non-negative radii".into());
    }
    if t < 2 {
        return Ok(vec![0.0; epsilons.len()]);
    }
    let (half, other) = if t.is_multiple_of(2) {
        (t / 2, t - 1)
    } else {
        (t, (t - 1) / 2)
    };
    let total_pairs = half.checked_mul(other).ok_or("pair count overflow")?;

    let dists = if total_pairs <= max_pairs {
        // Parallel computation of all unique pairs
        let results: Vec<Vec<f64>> = (0..t)
            .into_par_iter()
            .map(|i| {
                let mut local_dists = Vec::with_capacity(t - i - 1);
                let ti = &trajectory[i * d..(i + 1) * d];
                for j in (i + 1)..t {
                    let tj = &trajectory[j * d..(j + 1) * d];
                    local_dists.push(distance(ti, tj));
                }
                local_dists
            })
            .collect();
        results.into_iter().flatten().collect::<Vec<f64>>()
    } else {
        // Parallel subsampled pairs
        let n_chunks = rayon::current_num_threads().max(1).min(max_pairs);
        let pairs_per_chunk = max_pairs / n_chunks;

        let results: Vec<Vec<f64>> = (0..n_chunks)
            .into_par_iter()
            .map(|c| {
                let chunk_pairs = pairs_per_chunk + usize::from(c < max_pairs % n_chunks);
                let mut local_dists = Vec::with_capacity(chunk_pairs);
                let mut rng_state = seed ^ (c as u64).wrapping_mul(0x9e3779b97f4a7c15);
                let mut count = 0;
                while count < chunk_pairs {
                    rng_state = rng_state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let i = ((rng_state >> 33) as usize) % t;
                    rng_state = rng_state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let j = ((rng_state >> 33) as usize) % t;
                    if i == j {
                        continue;
                    }
                    let ti = &trajectory[i * d..(i + 1) * d];
                    let tj = &trajectory[j * d..(j + 1) * d];
                    local_dists.push(distance(ti, tj));
                    count += 1;
                }
                local_dists
            })
            .collect();
        results.into_iter().flatten().collect()
    };

    let n = dists.len() as f64;
    let mut c_eps = Vec::with_capacity(epsilons.len());
    for &eps in epsilons {
        let count = dists.par_iter().filter(|&&d| d < eps).count();
        c_eps.push(count as f64 / n);
    }
    Ok(c_eps)
}

/// Estimate attractor diameter from sampled points.
///
/// Uses all points up to 200 samples, then a deterministic strided subset.
#[must_use]
pub fn attractor_diameter(trajectory: &[f64], t: usize, d: usize) -> f64 {
    if validate_trajectory(trajectory, t, d).is_err() {
        return f64::NAN;
    }
    if t <= 1 {
        return 0.0;
    }
    let sample_size = t.min(200);
    let step = if t > 200 { t / 200 } else { 1 };

    (0..sample_size)
        .into_par_iter()
        .map(|si| {
            let i = si * step;
            let ti = &trajectory[i * d..(i + 1) * d];
            let mut local_max = 0.0;
            for sj in (si + 1)..sample_size {
                let j = sj * step;
                let tj = &trajectory[j * d..(j + 1) * d];
                let dist = distance(ti, tj);
                if dist > local_max {
                    local_max = dist;
                }
            }
            local_max
        })
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0.0)
}

/// Correlation dimension result.
#[derive(Debug, Clone)]
pub struct CorrelationDimensionResult {
    pub d2: f64,
    pub epsilons: Vec<f64>,
    pub c_eps: Vec<f64>,
    pub slopes: Vec<f64>,
    pub scaling_lo: f64,
    pub scaling_hi: f64,
}

/// Estimate correlation dimension D₂ from embedded trajectory.
///
/// 1. Compute C(ε) for logarithmically spaced ε.
/// 2. Local slopes in log-log space.
/// 3. Identify scaling region (lowest slope variance window).
/// 4. D₂ = mean slope in scaling region.
///
/// # Arguments
/// * `trajectory` - row-major (T × d) flattened trajectory
/// * `t` - time points
/// * `d` - embedding dimension
/// * `n_epsilons` - ε count, at least two and fitting native float64 array capacity
/// * `max_pairs` - max pairs for correlation integral
/// * `seed` - PRNG seed
///
/// # Returns
/// A result containing D₂, radii, measured pair fractions, local slopes and range.
/// Zero-diameter inputs retain actual pair counts, including zero for T < 2.
///
/// # Errors
/// Refuses invalid dimensions/counts, nonfinite coordinates and unrepresentable
/// radius grids.
#[allow(clippy::too_many_arguments)]
pub fn correlation_dimension(
    trajectory: &[f64],
    t: usize,
    d: usize,
    n_epsilons: usize,
    max_pairs: usize,
    seed: u64,
) -> Result<CorrelationDimensionResult, String> {
    validate_trajectory(trajectory, t, d)?;
    if n_epsilons < 2
        || n_epsilons > isize::MAX as usize / std::mem::size_of::<f64>()
        || max_pairs == 0
    {
        return Err(
            "n_epsilons must fit float64 array capacity and be >=2; max_pairs must be positive"
                .into(),
        );
    }
    let diam = attractor_diameter(trajectory, t, d);
    if !diam.is_finite() || (diam > 0.0 && diam * 0.01 == 0.0) {
        return Err("trajectory diameter cannot produce finite positive radii".into());
    }
    if diam == 0.0 {
        return Ok(CorrelationDimensionResult {
            d2: 0.0,
            epsilons: vec![1.0],
            c_eps: correlation_integral(trajectory, t, d, &[1.0], max_pairs, seed)?,
            slopes: vec![0.0],
            scaling_lo: 1.0,
            scaling_hi: 1.0,
        });
    }

    // Log-spaced ε from 1% to 100% of diameter
    let log_lo = (diam * 0.01_f64).ln();
    let log_hi = diam.ln();
    let epsilons: Vec<f64> = (0..n_epsilons)
        .map(|i| (log_lo + (log_hi - log_lo) * i as f64 / (n_epsilons - 1) as f64).exp())
        .collect();

    if epsilons.iter().any(|x| !x.is_finite() || *x <= 0.0)
        || epsilons.windows(2).any(|x| x[0] >= x[1])
    {
        return Err("epsilon grid must be finite positive and strictly increasing".into());
    }
    let c_eps = correlation_integral(trajectory, t, d, &epsilons, max_pairs, seed)?;

    // Local slopes: d log C / d log ε
    let mut valid_log_eps = Vec::new();
    let mut valid_log_c = Vec::new();
    for (i, &c) in c_eps.iter().enumerate() {
        if c > 0.0 {
            valid_log_eps.push(epsilons[i].ln());
            valid_log_c.push(c.ln());
        }
    }
    if valid_log_eps.len() < 3 {
        let eps_first = epsilons[0];
        let eps_last = *epsilons.last().unwrap_or(&1.0);
        return Ok(CorrelationDimensionResult {
            d2: 0.0,
            epsilons,
            c_eps,
            slopes: vec![0.0],
            scaling_lo: eps_first,
            scaling_hi: eps_last,
        });
    }

    let slopes: Vec<f64> = valid_log_c
        .windows(2)
        .zip(valid_log_eps.windows(2))
        .map(|(lc, le)| (lc[1] - lc[0]) / (le[1] - le[0]))
        .collect();

    // Find scaling region: window with lowest slope variance
    let window = slopes.len().clamp(2, 5);
    let mut best_var = f64::INFINITY;
    let mut best_start = 0usize;
    for i in 0..=slopes.len().saturating_sub(window) {
        let slice = &slopes[i..i + window];
        let mean: f64 = slice.iter().sum::<f64>() / window as f64;
        let var: f64 = slice.iter().map(|&s| (s - mean) * (s - mean)).sum::<f64>() / window as f64;
        if var < best_var {
            best_var = var;
            best_start = i;
        }
    }

    let d2: f64 = slopes[best_start..best_start + window].iter().sum::<f64>() / window as f64;
    let scaling_lo = valid_log_eps[best_start].exp();
    let scaling_hi = valid_log_eps[(best_start + window).min(valid_log_eps.len() - 1)].exp();

    Ok(CorrelationDimensionResult {
        d2,
        epsilons,
        c_eps,
        slopes,
        scaling_lo,
        scaling_hi,
    })
}

/// Kaplan-Yorke dimension from Lyapunov spectrum.
///
/// D_KY = j + (Σ_{i=1}^{j} λ_i) / |λ_{j+1}|
///
/// where j is the largest index such that the cumulative sum of the first
/// j exponents is non-negative.
///
/// Kaplan & Yorke 1979, Lecture Notes in Mathematics 730:228-237.
///
/// # Arguments
/// * `exponents` - finite Lyapunov exponents; a private copy is sorted descending.
///
/// # Returns
/// D_KY. Returns 0.0 for stable fixed points (all exponents negative).
#[must_use]
pub fn kaplan_yorke_dimension(exponents: &[f64]) -> f64 {
    if exponents.iter().any(|x| !x.is_finite()) {
        return f64::NAN;
    }
    let scale = exponents.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return exponents.len() as f64;
    }
    let mut le = exponents.to_vec();
    le.sort_by(|a, b| b.total_cmp(a));
    let mut total = 0.0_f64;
    let mut correction = 0.0_f64;
    let mut accepted = 0.0_f64;
    for (index, exponent) in le.iter().enumerate() {
        let value = exponent / scale;
        let updated = total + value;
        correction += if total.abs() >= value.abs() {
            (total - updated) + value
        } else {
            (value - updated) + total
        };
        total = updated;
        let prefix = total + correction;
        if prefix < 0.0 {
            return index as f64 + accepted / value.abs();
        }
        accepted = prefix;
    }
    le.len() as f64
}

/// Validate shape arithmetic and finite coordinates before indexing a trajectory.
fn validate_trajectory(trajectory: &[f64], t: usize, d: usize) -> Result<(), String> {
    let size = t.checked_mul(d).ok_or("trajectory size overflow")?;
    if d == 0 || trajectory.len() != size || trajectory.iter().any(|x| !x.is_finite()) {
        return Err(
            "trajectory must have positive dimension, matching length and finite coordinates"
                .into(),
        );
    }
    Ok(())
}

/// Compute an overflow-resistant Euclidean norm; infinity denotes an unrepresentable distance.
fn distance(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .fold(0.0_f64, |norm, (a, b)| norm.hypot(a - b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_diameter_curve_matches_actual_pair_fraction() -> Result<(), String> {
        for t in [0, 1, 3] {
            let trajectory = vec![0.0; t];
            let result = correlation_dimension(&trajectory, t, 1, 10, 100, 42)?;
            let integral = correlation_integral(&trajectory, t, 1, &result.epsilons, 100, 42)?;
            assert_eq!(result.d2, 0.0);
            assert_eq!(result.c_eps, integral, "incorrect pair fraction for T={t}");
        }
        Ok(())
    }

    #[test]
    fn extreme_finite_distances_preserve_strict_counts() -> Result<(), String> {
        for (point, radius, expected) in [
            (f64::MAX / 2.0, f64::MAX, 1.0),
            (1e-200, 5e-201, 0.0),
            (1.0, 1.0, 0.0),
        ] {
            assert_eq!(
                correlation_integral(&[0.0, point], 2, 1, &[radius], 1, 42)?,
                vec![expected]
            );
        }
        assert_eq!(
            correlation_integral(&[-f64::MAX, f64::MAX], 2, 1, &[f64::MAX], 1, 42)?,
            vec![0.0]
        );
        Ok(())
    }

    #[test]
    fn finite_spectrum_prefixes_and_tiny_denominators() {
        assert_eq!(
            kaplan_yorke_dimension(&[f64::MAX, f64::MAX, -f64::MAX, -f64::MAX, -f64::MAX]),
            4.0
        );
        for scale in [1.0, 1e-305, 1e305] {
            assert_eq!(kaplan_yorke_dimension(&[-2.0 * scale, 0.0, scale]), 2.5);
        }
        assert!(kaplan_yorke_dimension(&[f64::NAN]).is_nan());
    }

    #[test]
    fn invalid_metadata_refused_before_indexing() {
        for (t, d, trajectory) in [
            (usize::MAX, 2, &[][..]),
            (0, 0, &[][..]),
            (1, 1, &[f64::INFINITY][..]),
        ] {
            assert!(correlation_integral(trajectory, t, d, &[1.0], 1, 42).is_err());
            assert!(correlation_dimension(trajectory, t, d, 10, 1, 42).is_err());
            assert!(attractor_diameter(trajectory, t, d).is_nan());
        }
        assert!(correlation_integral(&[0.0, 1.0], 2, 1, &[1.0], 0, 42).is_err());
        assert!(correlation_integral(&[], 0, 1, &[f64::NAN], 1, 42).is_err());
        assert!(correlation_integral(&[], 0, 1, &[2.0, 1.0], 1, 42).is_err());
        assert!(correlation_dimension(&[-f64::MAX, f64::MAX], 2, 1, 10, 1, 42).is_err());
        assert!(correlation_dimension(&[], 0, 1, usize::MAX, 1, 42).is_err());
        assert!(correlation_dimension(
            &[],
            0,
            1,
            isize::MAX as usize / std::mem::size_of::<f64>() + 1,
            1,
            42,
        )
        .is_err());
    }

    #[test]
    fn subsample_smaller_than_thread_count_is_finite() -> Result<(), String> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .map_err(|e| e.to_string())?;
        for max_pairs in [1, 2, 5] {
            let values =
                pool.install(|| correlation_integral(&[0.0; 6], 6, 1, &[0.0, 1.0], max_pairs, 42))?;
            assert_eq!(values, vec![0.0, 1.0]);
        }
        Ok(())
    }

    #[test]
    fn sparse_radius_windows_and_representability() -> Result<(), String> {
        let points = [0.0, 1.0];
        assert!(correlation_dimension(&points, 2, 1, 1, 1, 42).is_err());
        assert!(correlation_dimension(&points, 2, 1, 10, 0, 42).is_err());
        assert!(correlation_dimension(&[0.0, f64::from_bits(1)], 2, 1, 10, 1, 42).is_err());
        let duplicate_grid = correlation_dimension(&[0.0, 1e-320], 2, 1, 4096, 1, 42);
        assert!(duplicate_grid
            .expect_err("subnormal radius grid must contain duplicates")
            .contains("strictly increasing"));
        let result = correlation_dimension(&points, 2, 1, 2, 1, 42)?;
        assert_eq!(result.d2, 0.0);
        assert_eq!(result.c_eps, vec![0.0, 0.0]);
        assert_eq!(result.slopes, vec![0.0]);
        Ok(())
    }

    #[test]
    fn test_correlation_integral_identical() {
        // All points identical → C(ε) = 1 for any ε > 0
        let traj = vec![1.0; 30]; // T=10, d=3
        let eps = vec![0.1, 1.0, 10.0];
        let c = correlation_integral(&traj, 10, 3, &eps, 50000, 42)
            .expect("identical finite trajectory is valid");
        // All distances = 0, so dist < eps is true for all
        assert!((c[0] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_correlation_integral_monotone() {
        // C(ε) must be non-decreasing
        let traj: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let eps = vec![0.01, 0.1, 0.5, 1.0, 2.0];
        let c = correlation_integral(&traj, 100, 1, &eps, 50000, 42)
            .expect("finite monotonicity trajectory is valid");
        for i in 1..c.len() {
            assert!(
                c[i] >= c[i - 1] - 1e-10,
                "C not monotone: C[{}]={} < C[{}]={}",
                i,
                c[i],
                i - 1,
                c[i - 1]
            );
        }
    }

    #[test]
    fn test_correlation_integral_range() {
        let traj: Vec<f64> = (0..50).map(|i| i as f64 * 0.2).collect();
        let eps = vec![0.5, 5.0];
        let c = correlation_integral(&traj, 50, 1, &eps, 50000, 42)
            .expect("finite range trajectory is valid");
        for &ci in &c {
            assert!((0.0..=1.0001).contains(&ci), "C(ε)={} not in [0,1]", ci);
        }
    }

    #[test]
    fn test_correlation_dimension_positive() {
        // Lorenz-like trajectory in 3D should have D₂ > 0
        // Simple surrogate: helix
        let n = 500;
        let mut traj = Vec::with_capacity(n * 3);
        for i in 0..n {
            let t = i as f64 * 0.1;
            traj.push(t.cos());
            traj.push(t.sin());
            traj.push(t * 0.01);
        }
        let result = correlation_dimension(&traj, n, 3, 20, 50000, 42)
            .expect("finite dimension trajectory is valid");
        let d2 = result.d2;
        assert!(d2 > 0.5, "helix D₂ should be ~1: {}", d2);
    }

    #[test]
    fn test_attractor_diameter_positive() {
        let traj: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let diam = attractor_diameter(&traj, 20, 1);
        assert!((diam - 19.0).abs() < 0.01, "diameter = {}", diam);
    }

    #[test]
    fn test_kaplan_yorke_fixed_point() {
        // All negative → D_KY = 0
        let le = vec![-1.0, -2.0, -3.0];
        assert_eq!(kaplan_yorke_dimension(&le), 0.0);
    }

    #[test]
    fn test_kaplan_yorke_limit_cycle() {
        // λ = [0, -1] → D_KY = 1 + 0/1 = 1
        let le = vec![0.0, -1.0];
        let dky = kaplan_yorke_dimension(&le);
        assert!(
            (dky - 1.0).abs() < 0.01,
            "limit cycle D_KY should be ~1: {}",
            dky
        );
    }

    #[test]
    fn test_kaplan_yorke_chaotic() {
        // λ = [0.5, 0.0, -1.5] → j=1, D_KY = 2 + 0.5/1.5 = 2.333
        let le = vec![0.5, 0.0, -1.5];
        let dky = kaplan_yorke_dimension(&le);
        assert!(
            (dky - 2.333).abs() < 0.01,
            "chaotic D_KY should be ~2.33: {}",
            dky
        );
    }

    #[test]
    fn test_kaplan_yorke_all_positive() {
        // All non-negative → D_KY = N
        let le = vec![1.0, 0.5, 0.1];
        let dky = kaplan_yorke_dimension(&le);
        assert_eq!(dky, 3.0);
    }

    #[test]
    fn test_kaplan_yorke_empty() {
        assert_eq!(kaplan_yorke_dimension(&[]), 0.0);
    }

    #[test]
    fn test_mismatched_length() {
        let result = correlation_integral(&[1.0, 2.0, 3.0], 2, 2, &[1.0], 100, 0);
        assert!(result.is_err());
    }
}
