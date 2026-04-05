// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — SSGF geometry carrier W(t)

//! Softplus decode z → W and finite-difference gradient for the SSGF
//! outer cycle.

/// Decode latent vector z → coupling matrix W via softplus(A·z).
///
/// W_ij = log(1 + exp(A·z))_ij, diagonal zeroed.
///
/// # Arguments
/// * `z` – latent vector, length `z_dim`
/// * `a` – projection matrix, row-major (n*n × z_dim)
/// * `n` – number of oscillators
///
/// # Returns
/// Row-major (n×n) coupling matrix, non-negative, zero diagonal.
#[must_use]
pub fn decode(z: &[f64], a: &[f64], n: usize) -> Vec<f64> {
    let z_dim = z.len();
    let nn = n * n;
    let mut w = vec![0.0; nn];
    for i in 0..nn {
        let mut raw = 0.0;
        for j in 0..z_dim {
            raw += a[i * z_dim + j] * z[j];
        }
        // softplus: log(1 + exp(x))
        w[i] = softplus(raw);
    }
    // Zero diagonal
    for i in 0..n {
        w[i * n + i] = 0.0;
    }
    w
}

/// Softplus activation: log(1 + exp(x)), numerically stable.
fn softplus(x: f64) -> f64 {
    if x > 20.0 { x } else if x < -20.0 { 0.0 } else { (1.0 + x.exp()).ln() }
}

/// Finite-difference gradient of a cost function w.r.t. z.
///
/// grad_i = (cost(z + ε·e_i) - cost(z - ε·e_i)) / (2ε)
///
/// The cost is evaluated by decoding z → W and calling the cost function.
#[must_use]
pub fn finite_diff_gradient(
    z: &[f64],
    a: &[f64],
    n: usize,
    cost_fn: &dyn Fn(&[f64]) -> f64,
    epsilon: f64,
) -> Vec<f64> {
    let z_dim = z.len();
    let mut grad = vec![0.0; z_dim];
    for i in 0..z_dim {
        let mut z_plus = z.to_vec();
        z_plus[i] += epsilon;
        let mut z_minus = z.to_vec();
        z_minus[i] -= epsilon;
        let w_plus = decode(&z_plus, a, n);
        let w_minus = decode(&z_minus, a, n);
        grad[i] = (cost_fn(&w_plus) - cost_fn(&w_minus)) / (2.0 * epsilon);
    }
    grad
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_zero_z() {
        let n = 3;
        let z_dim = 4;
        let z = vec![0.0; z_dim];
        let a = vec![1.0; n * n * z_dim];
        let w = decode(&z, &a, n);
        // softplus(0) = ln(2) ≈ 0.693
        let sp0 = 2.0_f64.ln();
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    assert_eq!(w[i * n + j], 0.0);
                } else {
                    assert!((w[i * n + j] - sp0).abs() < 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_decode_diagonal_zero() {
        let n = 4;
        let z = vec![1.0; 3];
        let a = vec![0.5; n * n * 3];
        let w = decode(&z, &a, n);
        for i in 0..n {
            assert_eq!(w[i * n + i], 0.0);
        }
    }

    #[test]
    fn test_decode_non_negative() {
        let n = 3;
        let z = vec![-2.0, 1.0, 0.5];
        let a = vec![0.3; n * n * 3];
        let w = decode(&z, &a, n);
        for &v in &w {
            assert!(v >= 0.0, "W should be non-negative, got {v}");
        }
    }

    #[test]
    fn test_softplus_large_positive() {
        assert!((softplus(100.0) - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_softplus_large_negative() {
        assert!(softplus(-100.0) < 1e-10);
    }

    #[test]
    fn test_softplus_zero() {
        assert!((softplus(0.0) - 2.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_gradient_zero_at_minimum() {
        // Cost = sum(W), minimum at large negative z
        let n = 2;
        let z_dim = 2;
        let z = vec![-10.0; z_dim];
        let a = vec![1.0; n * n * z_dim];
        let cost = |w: &[f64]| -> f64 { w.iter().sum() };
        let grad = finite_diff_gradient(&z, &a, n, &cost, 1e-4);
        // At z = -10, softplus ≈ 0, gradient ≈ 0
        for g in &grad {
            assert!(g.abs() < 0.01, "grad={g} should be ~0 at minimum");
        }
    }

    #[test]
    fn test_gradient_direction() {
        // Cost = sum(W), gradient should be positive (increasing z increases W)
        let n = 2;
        let z_dim = 2;
        let z = vec![0.0; z_dim];
        let a = vec![1.0; n * n * z_dim];
        let cost = |w: &[f64]| -> f64 { w.iter().sum() };
        let grad = finite_diff_gradient(&z, &a, n, &cost, 1e-4);
        for g in &grad {
            assert!(*g > 0.0, "gradient should be positive, got {g}");
        }
    }
}
