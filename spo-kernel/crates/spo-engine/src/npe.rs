// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Normalized Persistent Entropy (NPE)

//! NPE from H0 persistence diagram via single-linkage (Kruskal MST).
//!
//! More sensitive than the Kuramoto order parameter R for detecting
//! synchronisation onset. NPE ~ 1 = incoherent, NPE ~ 0 = synchronised.

/// Pairwise circular distance matrix (flat, row-major).
///
/// Returns Vec of length n*n with distances in [0, pi].
#[must_use]
pub fn phase_distance_matrix(phases: &[f64]) -> Vec<f64> {
    let n = phases.len();
    let mut dist = vec![0.0; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let diff = phases[j] - phases[i];
            let d = diff.sin().atan2(diff.cos()).abs();
            dist[i * n + j] = d;
            dist[j * n + i] = d;
        }
    }
    dist
}

/// Compute Normalized Persistent Entropy.
///
/// Uses Kruskal MST on circular distance to extract H0 barcode,
/// then computes normalised Shannon entropy of lifetimes.
///
/// Returns NPE in [0, 1]. 0 = synchronised, 1 = incoherent.
#[must_use]
pub fn compute_npe(phases: &[f64], max_radius: f64) -> f64 {
    let n = phases.len();
    if n < 2 {
        return 0.0;
    }

    // Collect upper-triangle edges: (distance, i, j)
    let n_edges = n * (n - 1) / 2;
    let mut edges: Vec<(f64, usize, usize)> = Vec::with_capacity(n_edges);
    for i in 0..n {
        for j in (i + 1)..n {
            let diff = phases[j] - phases[i];
            let d = diff.sin().atan2(diff.cos()).abs();
            edges.push((d, i, j));
        }
    }

    // Sort by distance
    edges.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Union-Find
    let mut parent: Vec<usize> = (0..n).collect();
    let mut rank: Vec<u8> = vec![0; n];

    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]]; // path halving
            x = parent[x];
        }
        x
    }

    let mut lifetimes: Vec<f64> = Vec::with_capacity(n - 1);

    for &(d, i, j) in &edges {
        if d > max_radius {
            break;
        }
        let ri = find(&mut parent, i);
        let rj = find(&mut parent, j);
        if ri != rj {
            lifetimes.push(d);
            // Union by rank
            if rank[ri] < rank[rj] {
                parent[ri] = rj;
            } else if rank[ri] > rank[rj] {
                parent[rj] = ri;
            } else {
                parent[rj] = ri;
                rank[ri] += 1;
            }
        }
    }

    if lifetimes.is_empty() {
        return 0.0;
    }

    let total: f64 = lifetimes.iter().sum();
    if total < 1e-15 {
        return 0.0;
    }

    // Shannon entropy of persistence probabilities
    let mut entropy = 0.0;
    for &lt in &lifetimes {
        if lt > 0.0 {
            let p = lt / total;
            entropy -= p * p.ln();
        }
    }

    // Normalise by log(k) where k = number of non-zero lifetimes
    let k = lifetimes.iter().filter(|&&lt| lt > 0.0).count();
    let max_entropy = if k > 1 { (k as f64).ln() } else { 1.0 };

    if max_entropy < 1e-15 {
        return 0.0;
    }

    entropy / max_entropy
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn sync_phases_give_zero_npe() {
        let phases = vec![0.0; 8];
        let npe = compute_npe(&phases, PI);
        assert!(npe.abs() < 1e-10, "sync should give NPE=0, got {npe}");
    }

    #[test]
    fn uniform_phases_give_high_npe() {
        let n = 16;
        let phases: Vec<f64> = (0..n).map(|i| 2.0 * PI * i as f64 / n as f64).collect();
        let npe = compute_npe(&phases, PI);
        assert!(npe > 0.5, "uniform phases should give high NPE, got {npe}");
    }

    #[test]
    fn single_oscillator_returns_zero() {
        assert_eq!(compute_npe(&[1.0], PI), 0.0);
    }

    #[test]
    fn empty_returns_zero() {
        assert_eq!(compute_npe(&[], PI), 0.0);
    }

    #[test]
    fn npe_in_unit_interval() {
        let phases = vec![0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let npe = compute_npe(&phases, PI);
        assert!(
            (0.0..=1.0).contains(&npe),
            "NPE should be in [0,1], got {npe}"
        );
    }

    #[test]
    fn distance_matrix_symmetric() {
        let phases = vec![0.0, 1.0, 2.0, 3.0];
        let dist = phase_distance_matrix(&phases);
        let n = 4;
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (dist[i * n + j] - dist[j * n + i]).abs() < 1e-15,
                    "distance matrix not symmetric at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn distance_matrix_diagonal_zero() {
        let phases = vec![0.5, 1.5, 2.5];
        let dist = phase_distance_matrix(&phases);
        for i in 0..3 {
            assert_eq!(dist[i * 3 + i], 0.0);
        }
    }

    #[test]
    fn distance_in_range() {
        let phases = vec![0.0, PI, PI / 2.0, 3.0 * PI / 2.0];
        let dist = phase_distance_matrix(&phases);
        for &d in &dist {
            assert!(d >= 0.0 && d <= PI + 1e-10, "distance {d} out of [0, pi]");
        }
    }
}
