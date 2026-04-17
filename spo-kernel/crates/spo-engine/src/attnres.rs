// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — AttnRes coupling modulation

//! Rust port of the Python AttnRes spike.
//!
//! Mirrors
//! `src/scpn_phase_orchestrator/coupling/attention_residuals.py`
//! bit-exactly up to floating-point rounding. See the Python docstring
//! and `docs/internal/research_attention_residuals_2026-04-06.md §3.2`
//! for the physics and motivation.
//!
//! Contract summary:
//!
//! * Pure function, does not mutate inputs.
//! * Output is symmetric with zero diagonal.
//! * `lambda = 0` returns an exact copy of `knm`.
//! * Attention is restricted to a `±block_size` local window, and
//!   zero `knm` entries stay zero (no new edges created).
//!
//! The Python benchmark showed a 236–267% overhead at N ≤ 128 because
//! the modulation was NumPy-side while the baseline step was Rust-side.
//! This Rust implementation closes that gap.
//!
//! Rayon is intentionally NOT used here: each row is O(N) work with
//! tight per-row allocations, and for realistic SPO network sizes
//! (N ≤ 64 layers) the thread-pool wake-up cost dominates. If AttnRes
//! is ever called on N ≫ 1024 matrices, add a `par_iter` branch
//! behind an N threshold rather than making the small-N path slower.

/// Compute the AttnRes-modulated `K_nm` for the given `theta`.
///
/// # Arguments
/// * `knm` — row-major `N × N` coupling matrix, symmetric with zero
///   diagonal. Length must be `n * n`.
/// * `theta` — phase vector, length `n`.
/// * `n` — matrix dimension.
/// * `block_size` — half-width of the local attention window.
///   Must be ≥ 1. Attention is restricted to pairs with
///   `|i − j| ≤ block_size`.
/// * `temperature` — softmax temperature. Must be > 0.
/// * `lambda_` — modulation strength. Must be ≥ 0. A value of 0 returns
///   an exact copy of `knm`.
///
/// # Errors
/// Returns `Err` for shape mismatches and out-of-range hyperparameters.
pub fn attnres_modulate(
    knm: &[f64],
    theta: &[f64],
    n: usize,
    block_size: usize,
    temperature: f64,
    lambda_: f64,
) -> Result<Vec<f64>, String> {
    if knm.len() != n * n {
        return Err(format!(
            "knm length {} does not match n*n = {}",
            knm.len(),
            n * n
        ));
    }
    if theta.len() != n {
        return Err(format!(
            "theta length {} does not match n = {}",
            theta.len(),
            n
        ));
    }
    if block_size == 0 {
        return Err("block_size must be ≥ 1".to_string());
    }
    if temperature <= 0.0 || !temperature.is_finite() {
        return Err("temperature must be finite and > 0".to_string());
    }
    if lambda_ < 0.0 {
        return Err("lambda_ must be ≥ 0".to_string());
    }

    if lambda_ == 0.0 {
        return Ok(knm.to_vec());
    }

    let inv_t = 1.0 / temperature;

    // Single scratch buffer for logits (reused across rows) and one
    // flat output buffer sized once. The symmetrisation uses a second
    // flat buffer for the row-wise intermediate, folded into the final
    // pass. Total allocation: 2 × N·f64 + N·f64 scratch.
    let mut rowwise = vec![0.0_f64; n * n];
    let mut logits = vec![f64::NEG_INFINITY; n];

    for i in 0..n {
        // Reset scratch for this row.
        logits.fill(f64::NEG_INFINITY);

        let lo = i.saturating_sub(block_size);
        let hi = (i + block_size + 1).min(n);
        let mut any_unmasked = false;
        let row_off = i * n;
        for j in lo..hi {
            if j == i || knm[row_off + j] == 0.0 {
                continue;
            }
            logits[j] = (theta[j] - theta[i]).cos() * inv_t;
            any_unmasked = true;
        }

        // Numerically-stable softmax in place.
        if any_unmasked {
            let mut row_max = f64::NEG_INFINITY;
            for &x in &logits {
                if x > row_max {
                    row_max = x;
                }
            }
            let mut denom = 0.0_f64;
            for j in 0..n {
                if logits[j].is_finite() {
                    let e = (logits[j] - row_max).exp();
                    logits[j] = e;
                    denom += e;
                } else {
                    logits[j] = 0.0;
                }
            }
            if denom > 0.0 {
                let inv_denom = 1.0 / denom;
                for slot in &mut logits {
                    *slot *= inv_denom;
                }
            }
        } else {
            // No unmasked entries: zero attention weights for the row.
            logits.fill(0.0);
        }

        for j in 0..n {
            rowwise[row_off + j] = knm[row_off + j] * (1.0 + lambda_ * logits[j]);
        }
    }

    // Symmetrise: (R + R^T) / 2 — keeps K_mod[i,j] == K_mod[j,i]
    // despite per-row softmax asymmetry. Single flat pass, no per-row
    // allocation.
    let mut out = vec![0.0_f64; n * n];
    for i in 0..n {
        let row_i = i * n;
        for j in 0..n {
            if i == j {
                // Diagonal already zero from the allocation.
                continue;
            }
            let row_j = j * n;
            out[row_i + j] = 0.5 * (rowwise[row_i + j] + rowwise[row_j + i]);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_symmetric(n: usize, seed: u64) -> Vec<f64> {
        // Deterministic, simple generator — avoids an RNG dependency
        // for the pure-Rust unit tests.
        let mut k = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in (i + 1)..n {
                let v = 0.3 + 0.01 * ((seed as usize + i * n + j) % 20) as f64;
                k[i * n + j] = v;
                k[j * n + i] = v;
            }
        }
        k
    }

    #[test]
    fn symmetry_preserved() {
        let n = 8;
        let knm = build_symmetric(n, 1);
        let theta: Vec<f64> = (0..n).map(|i| (i as f64) * 0.3).collect();
        let out = attnres_modulate(&knm, &theta, n, 3, 0.1, 0.5).unwrap();
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (out[i * n + j] - out[j * n + i]).abs() < 1e-12,
                    "asymmetric at ({i}, {j})"
                );
            }
        }
    }

    #[test]
    fn zero_diagonal_preserved() {
        let n = 10;
        let knm = build_symmetric(n, 2);
        let theta = vec![0.0; n];
        let out = attnres_modulate(&knm, &theta, n, 4, 0.1, 0.5).unwrap();
        for i in 0..n {
            assert_eq!(out[i * n + i], 0.0);
        }
    }

    #[test]
    fn lambda_zero_is_identity() {
        let n = 6;
        let knm = build_symmetric(n, 3);
        let theta: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
        let out = attnres_modulate(&knm, &theta, n, 2, 0.1, 0.0).unwrap();
        assert_eq!(out, knm);
    }

    #[test]
    fn existing_zeros_stay_zero() {
        let n = 8;
        let mut knm = build_symmetric(n, 4);
        // Knock out a pair symmetrically
        knm[0 * n + 3] = 0.0;
        knm[3 * n + 0] = 0.0;
        let theta = vec![0.0; n];
        let out = attnres_modulate(&knm, &theta, n, 4, 0.1, 0.5).unwrap();
        assert_eq!(out[0 * n + 3], 0.0);
        assert_eq!(out[3 * n + 0], 0.0);
    }

    #[test]
    fn out_of_block_unchanged() {
        let n = 16;
        let knm = build_symmetric(n, 5);
        let theta: Vec<f64> = (0..n).map(|i| i as f64 * 0.4).collect();
        let block = 2;
        let out = attnres_modulate(&knm, &theta, n, block, 0.1, 0.5).unwrap();
        for i in 0..n {
            for j in 0..n {
                if i.abs_diff(j) > block {
                    assert!(
                        (out[i * n + j] - knm[i * n + j]).abs() < 1e-12,
                        "out-of-block ({i}, {j}) was modulated: {} → {}",
                        knm[i * n + j],
                        out[i * n + j]
                    );
                }
            }
        }
    }

    #[test]
    fn shape_mismatch_rejected() {
        let knm = vec![0.0; 9];
        let theta = vec![0.0; 3];
        // n * n = 16 ≠ 9
        assert!(attnres_modulate(&knm, &theta, 4, 2, 0.1, 0.5).is_err());
    }

    #[test]
    fn theta_mismatch_rejected() {
        let knm = vec![0.0; 16];
        let theta = vec![0.0; 5];
        assert!(attnres_modulate(&knm, &theta, 4, 2, 0.1, 0.5).is_err());
    }

    #[test]
    fn block_zero_rejected() {
        let knm = vec![0.0; 16];
        let theta = vec![0.0; 4];
        assert!(attnres_modulate(&knm, &theta, 4, 0, 0.1, 0.5).is_err());
    }

    #[test]
    fn temperature_zero_rejected() {
        let knm = vec![0.0; 16];
        let theta = vec![0.0; 4];
        assert!(attnres_modulate(&knm, &theta, 4, 2, 0.0, 0.5).is_err());
    }

    #[test]
    fn negative_lambda_rejected() {
        let knm = vec![0.0; 16];
        let theta = vec![0.0; 4];
        assert!(attnres_modulate(&knm, &theta, 4, 2, 0.1, -0.1).is_err());
    }

    #[test]
    fn deterministic_same_inputs() {
        let n = 8;
        let knm = build_symmetric(n, 7);
        let theta: Vec<f64> = (0..n).map(|i| 0.1 + i as f64 * 0.2).collect();
        let a = attnres_modulate(&knm, &theta, n, 3, 0.1, 0.3).unwrap();
        let b = attnres_modulate(&knm, &theta, n, 3, 0.1, 0.3).unwrap();
        assert_eq!(a, b);
    }
}
