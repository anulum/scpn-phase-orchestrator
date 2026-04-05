// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Synthetic HCP-inspired connectome

//! Generate a structured coupling matrix that mimics known macroscale
//! brain connectivity patterns (Hagmann et al. 2008, PLoS Biol. 6:e159).

const INTRA_HEMI_STRENGTH: f64 = 0.5;
const INTER_HEMI_STRENGTH: f64 = 0.15;
const DMN_HUB_BOOST: f64 = 0.3;

/// Generate a synthetic HCP-inspired coupling matrix.
///
/// Features: intra-hemispheric exponential decay, inter-hemispheric
/// callosal connections, and default mode network hub structure.
///
/// Returns row-major (n × n) symmetric matrix, zero diagonal.
#[must_use]
pub fn load_hcp_connectome(n_regions: usize, seed: u64) -> Vec<f64> {
    if n_regions < 2 {
        return vec![0.0; n_regions * n_regions];
    }
    let n = n_regions;
    let mut knm = vec![0.0; n * n];
    let half = n / 2;

    build_intra_hemi(&mut knm, n, half, seed);
    build_inter_hemi(&mut knm, n, half);
    add_dmn_hubs(&mut knm, n, half);
    symmetrise(&mut knm, n);

    knm
}

/// Intra-hemispheric exponential distance decay with noise.
fn build_intra_hemi(knm: &mut [f64], n: usize, half: usize, seed: u64) {
    let mut rng = seed;
    for hemi_start in [0, half] {
        let hemi_end = if hemi_start == 0 { half } else { n };
        let size = hemi_end - hemi_start;
        for i in 0..size {
            for j in 0..size {
                if i == j {
                    continue;
                }
                let dist = i.abs_diff(j);
                let base = INTRA_HEMI_STRENGTH * (-0.3 * dist as f64).exp();
                // LCG noise
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let noise = ((rng >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.04;
                let val = (base + noise).max(0.0);
                knm[(hemi_start + i) * n + (hemi_start + j)] = val;
            }
        }
    }
}

/// Inter-hemispheric (callosal) connections.
fn build_inter_hemi(knm: &mut [f64], n: usize, half: usize) {
    let spread = 3.min(half);
    for i in 0..half.min(n - half) {
        let j = i + half;
        let weight = INTER_HEMI_STRENGTH;
        for offset in -(spread as i64)..=(spread as i64) {
            let ji = i as i64 + offset;
            let jj = j as i64 + offset - if offset != 0 { offset } else { 0 };
            if ji >= 0 && (ji as usize) < half && (jj as usize) >= half && (jj as usize) < n {
                let w = weight * (-0.5 * (offset as f64).abs()).exp();
                knm[ji as usize * n + jj as usize] = w;
                knm[jj as usize * n + ji as usize] = w;
            }
        }
    }
}

/// Add DMN hub connections.
fn add_dmn_hubs(knm: &mut [f64], n: usize, half: usize) {
    let fractions = [0.15, 0.45, 0.65, 0.85];
    let mut dmn_nodes = Vec::new();
    for &f in &fractions {
        let left = (f * half as f64) as usize;
        if left < n {
            dmn_nodes.push(left);
        }
        let right = left + half;
        if right < n {
            dmn_nodes.push(right);
        }
    }
    for &hub in &dmn_nodes {
        for &other in &dmn_nodes {
            if hub != other {
                knm[hub * n + other] += DMN_HUB_BOOST;
            }
        }
    }
}

/// Symmetrise and clean: W = (W + Wᵀ)/2, diagonal zeroed, clamp ≥ 0.
fn symmetrise(knm: &mut [f64], n: usize) {
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = (knm[i * n + j] + knm[j * n + i]) / 2.0;
            let val = avg.max(0.0);
            knm[i * n + j] = val;
            knm[j * n + i] = val;
        }
        knm[i * n + i] = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_size() {
        let knm = load_hcp_connectome(10, 42);
        assert_eq!(knm.len(), 100);
    }

    #[test]
    fn test_diagonal_zero() {
        let n = 8;
        let knm = load_hcp_connectome(n, 42);
        for i in 0..n {
            assert_eq!(knm[i * n + i], 0.0);
        }
    }

    #[test]
    fn test_symmetric() {
        let n = 10;
        let knm = load_hcp_connectome(n, 42);
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (knm[i * n + j] - knm[j * n + i]).abs() < 1e-15,
                    "K[{i},{j}] ≠ K[{j},{i}]"
                );
            }
        }
    }

    #[test]
    fn test_non_negative() {
        let n = 20;
        let knm = load_hcp_connectome(n, 42);
        for &v in &knm {
            assert!(v >= 0.0, "negative value {v}");
        }
    }

    #[test]
    fn test_intra_stronger_than_inter() {
        let n = 20;
        let half = n / 2;
        let knm = load_hcp_connectome(n, 42);
        let mut intra = 0.0_f64;
        for i in 0..half {
            for j in 0..half {
                intra += knm[i * n + j];
            }
        }
        let mut inter = 0.0_f64;
        for i in 0..half {
            for j in half..n {
                inter += knm[i * n + j];
            }
        }
        assert!(intra > inter, "intra={intra} should > inter={inter}");
    }

    #[test]
    fn test_small_n() {
        let knm = load_hcp_connectome(2, 42);
        assert_eq!(knm.len(), 4);
        assert_eq!(knm[0], 0.0);
        assert_eq!(knm[3], 0.0);
    }

    #[test]
    fn test_deterministic() {
        let a = load_hcp_connectome(10, 42);
        let b = load_hcp_connectome(10, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn test_different_seed() {
        let a = load_hcp_connectome(10, 42);
        let b = load_hcp_connectome(10, 99);
        assert_ne!(a, b);
    }
}
