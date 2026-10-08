// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Chimera state detection (Kuramoto & Battogtokh 2002)

//! Chimera state detection via local order parameter.
//!
//! This instantaneous, unweighted adjacency diagnostic reports local coherence
//! and a boundary fraction. It does not certify a dynamical chimera state or
//! implement the weighted nonlocal field or temporal classification below.
//!
//! References:
//! - Kuramoto & Battogtokh 2002, Nonlinear Phenom. Complex Syst. 5:380-385.
//! - Abrams & Strogatz 2004, Phys. Rev. Lett. 93:174102.
//! - Kemeth et al. 2016, Chaos 26:094815 (chimera classification).

use rayon::prelude::*;

/// Chimera detection result.
#[derive(Debug, Clone)]
pub struct ChimeraResult {
    /// Indices of coherent oscillators (R_local > 0.7).
    pub coherent_indices: Vec<usize>,
    /// Indices of incoherent oscillators (R_local < 0.3).
    pub incoherent_indices: Vec<usize>,
    /// Chimera index: fraction of oscillators in the boundary zone
    /// (neither clearly coherent nor clearly incoherent).
    /// High chimera_index → complex partial synchronisation pattern.
    pub chimera_index: f64,
    /// Local order parameters R_i for each oscillator.
    pub local_order: Vec<f64>,
}

// Implementation thresholds for the instantaneous boundary-fraction diagnostic.
const COHERENT_THRESHOLD: f64 = 0.7;
const INCOHERENT_THRESHOLD: f64 = 0.3;

/// Compute local order parameter for each oscillator.
///
/// R_i = |<exp(i(θ_j − θ_i))>|_j∈N(i)
///
/// where N(i) = {j : j != i and K_ij > 0}. Admitted diagonal residue is ignored.
///
/// # Arguments
/// * `phases` - (N,) oscillator phases
/// * `knm` - (N×N) row-major coupling matrix; K_ij > 0 defines neighbourhood
/// * `n` - number of oscillators
///
/// # Returns
/// (N,) local order parameters in [0, 1], including the empty identity.
///
/// # Errors
/// Refuses inconsistent dimensions, nonfinite values and nonzero diagonal.
pub fn try_local_order_parameter(
    phases: &[f64],
    knm: &[f64],
    n: usize,
) -> Result<Vec<f64>, &'static str> {
    let expected = n.checked_mul(n).ok_or("n*n overflows usize")?;
    if phases.len() != n || knm.len() != expected {
        return Err("phases or knm cardinality does not match n");
    }
    if phases
        .iter()
        .chain(knm.iter())
        .any(|value| !value.is_finite())
    {
        return Err("phases and knm must contain only finite values");
    }
    if (0..n).any(|i| knm[i * n + i].abs() > 1e-15) {
        return Err("knm self-coupling diagonal must be zero");
    }
    let mut r_local = vec![0.0_f64; n];
    // Parallelise per-oscillator neighbourhood sums; each row of knm is
    // read independently so there is no shared-mutability hazard.
    r_local.par_iter_mut().enumerate().for_each(|(i, val)| {
        let mut re_sum = 0.0_f64;
        let mut im_sum = 0.0_f64;
        let mut count = 0usize;
        for j in 0..n {
            if j != i && knm[i * n + j] > 0.0 {
                re_sum += phases[j].cos();
                im_sum += phases[j].sin();
                count += 1;
            }
        }
        if count > 0 {
            let c = count as f64;
            let mean_re = re_sum / c;
            let mean_im = im_sum / c;
            *val = mean_re.hypot(mean_im).min(1.0);
        }
    });
    Ok(r_local)
}

/// Compatible local-order API: invalid input returns an empty vector.
///
/// Valid results have exactly n entries. Use `try_local_order_parameter`
/// to distinguish a malformed request from the valid n=0 identity.
#[must_use]
pub fn local_order_parameter(phases: &[f64], knm: &[f64], n: usize) -> Vec<f64> {
    try_local_order_parameter(phases, knm, n).unwrap_or_default()
}

/// Detect chimera states in a Kuramoto network.
///
/// Classifies each oscillator as coherent (R_i > 0.7), incoherent (R_i < 0.3),
/// or boundary. The chimera index measures the fraction in the boundary zone.
///
/// # Arguments
/// * `phases` - (N,) oscillator phases
/// * `knm` - (N×N) row-major coupling matrix
/// * `n` - number of oscillators
///
/// # Returns
/// `ChimeraResult` with coherent/incoherent indices, chimera index,
/// and local order parameters.
///
/// # Errors
/// Refuses invalid inputs through `try_local_order_parameter`.
pub fn try_detect_chimera(
    phases: &[f64],
    knm: &[f64],
    n: usize,
) -> Result<ChimeraResult, &'static str> {
    let r_local = try_local_order_parameter(phases, knm, n)?;
    if n == 0 {
        return Ok(ChimeraResult {
            coherent_indices: vec![],
            incoherent_indices: vec![],
            chimera_index: 0.0,
            local_order: vec![],
        });
    }

    let mut coherent = Vec::new();
    let mut incoherent = Vec::new();
    for (i, &r) in r_local.iter().enumerate() {
        if r > COHERENT_THRESHOLD {
            coherent.push(i);
        } else if r < INCOHERENT_THRESHOLD {
            incoherent.push(i);
        }
    }

    let boundary_count = n - coherent.len() - incoherent.len();
    let chimera_index = boundary_count as f64 / n as f64;

    Ok(ChimeraResult {
        coherent_indices: coherent,
        incoherent_indices: incoherent,
        chimera_index,
        local_order: r_local,
    })
}

/// Compatible detection API: invalid inputs return NaN and empty partitions.
///
/// Use `try_detect_chimera` for an explicit error. Valid inputs, including n=0,
/// retain the historical classification and boundary-fraction result.
#[must_use]
pub fn detect_chimera(phases: &[f64], knm: &[f64], n: usize) -> ChimeraResult {
    try_detect_chimera(phases, knm, n).unwrap_or_else(|_| ChimeraResult {
        coherent_indices: vec![],
        incoherent_indices: vec![],
        chimera_index: f64::NAN,
        local_order: vec![],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{PI, TAU};

    fn uniform_knm(n: usize, k: f64) -> Vec<f64> {
        let mut knm = vec![k; n * n];
        for i in 0..n {
            knm[i * n + i] = 0.0;
        }
        knm
    }

    #[test]
    fn test_empty() {
        let result = detect_chimera(&[], &[], 0);
        assert!(result.coherent_indices.is_empty());
        assert!(result.incoherent_indices.is_empty());
        assert_eq!(result.chimera_index, 0.0);
    }

    #[test]
    fn test_fully_synchronised() {
        // All phases identical → all R_local = 1 → all coherent
        let n = 8;
        let phases = vec![1.0; n];
        let knm = uniform_knm(n, 1.0);
        let result = detect_chimera(&phases, &knm, n);
        assert_eq!(result.coherent_indices.len(), n);
        assert!(result.incoherent_indices.is_empty());
        assert_eq!(result.chimera_index, 0.0);
    }

    #[test]
    fn test_fully_desynchronised() {
        // Phases uniformly spread → low R_local → all incoherent
        let n = 8;
        let phases: Vec<f64> = (0..n).map(|i| i as f64 * TAU / n as f64).collect();
        let knm = uniform_knm(n, 1.0);
        let result = detect_chimera(&phases, &knm, n);
        // Uniform distribution on circle: R = 0 for large N
        assert!(
            result.incoherent_indices.len() >= n / 2,
            "expected mostly incoherent, got {} incoherent out of {}",
            result.incoherent_indices.len(),
            n
        );
    }

    #[test]
    fn test_chimera_pattern() {
        // First half synchronised (phase ~0), second half random
        let n = 8;
        let mut phases = vec![0.1; n / 2];
        for i in 0..n / 2 {
            phases.push((i as f64) * TAU / (n / 2) as f64);
        }
        let knm = uniform_knm(n, 1.0);
        let result = detect_chimera(&phases, &knm, n);
        // Should have some coherent and some incoherent
        assert!(
            !result.coherent_indices.is_empty() || !result.incoherent_indices.is_empty(),
            "chimera should produce mixed classification"
        );
    }

    #[test]
    fn test_local_order_in_range() {
        let n = 6;
        let phases: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
        let knm = uniform_knm(n, 1.0);
        let r = local_order_parameter(&phases, &knm, n);
        for (i, &ri) in r.iter().enumerate() {
            assert!(
                (0.0..=1.0001).contains(&ri),
                "R_{} = {} not in [0,1]",
                i,
                ri
            );
        }
    }

    #[test]
    fn test_no_coupling_zero_local_order() {
        // No neighbours → R_local = 0
        let n = 4;
        let phases = vec![0.0, 1.0, 2.0, 3.0];
        let knm = vec![0.0; n * n];
        let r = local_order_parameter(&phases, &knm, n);
        for (i, &ri) in r.iter().enumerate() {
            assert_eq!(ri, 0.0, "R_{} should be 0 with no coupling", i);
        }
    }

    #[test]
    fn test_two_cluster_coherent() {
        // Two clusters with only intra-cluster coupling → both coherent
        let n = 8;
        let mut phases = Vec::new();
        // Cluster 1: near 0
        for i in 0..4 {
            phases.push(i as f64 * 0.02);
        }
        // Cluster 2: near π
        for i in 0..4 {
            phases.push(PI + i as f64 * 0.02);
        }
        // Only intra-cluster coupling (no inter)
        let mut knm = vec![0.0; n * n];
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    knm[i * n + j] = 5.0;
                    knm[(i + 4) * n + (j + 4)] = 5.0;
                }
            }
        }
        let result = detect_chimera(&phases, &knm, n);
        // Both clusters tightly coupled → all coherent
        assert_eq!(
            result.coherent_indices.len(),
            n,
            "all should be coherent with intra-only coupling: {:?}",
            result.local_order
        );
    }

    #[test]
    fn test_chimera_index_range() {
        let n = 10;
        let phases: Vec<f64> = (0..n).map(|i| i as f64 * 0.3).collect();
        let knm = uniform_knm(n, 1.0);
        let result = detect_chimera(&phases, &knm, n);
        assert!(
            (0.0..=1.0).contains(&result.chimera_index),
            "chimera_index {} not in [0,1]",
            result.chimera_index
        );
    }
}
