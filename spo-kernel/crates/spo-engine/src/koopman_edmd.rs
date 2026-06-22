// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Koopman EDMD-with-control least-squares solve

//! Extended Dynamic Mode Decomposition with control (Korda & Mezić 2018).
//!
//! Given row-major lifted snapshots `x_lift` (K×N), controls `inputs` (K×m),
//! lifted successors `y_lift` (K×N) and raw states `states` (K×n), this solves
//! the two Tikhonov-regularised normal systems
//!
//! ```text
//! (ΦᵀΦ + ρI) [Aᵀ; Bᵀ] = Φᵀ Y_lift,     Φ = [X_lift | U]
//! (X_liftᵀ X_lift + ρI) Cᵀ = X_liftᵀ X
//! ```
//!
//! and returns the row-major matrices `A` (N×N), `B` (N×m) and `C` (n×N). The
//! linear systems are solved by Gaussian elimination with partial pivoting over
//! all right-hand-side columns at once, matching the NumPy reference to machine
//! precision.

/// Solve the EDMD-with-control least-squares problem.
///
/// All matrices are row-major flat slices. Returns `(a, b, c)` as row-major flat
/// vectors of lengths `n_lift²`, `n_lift·m` and `n_state·n_lift`.
#[allow(clippy::too_many_arguments)]
pub fn koopman_edmd_solve(
    x_lift: &[f64],
    inputs: &[f64],
    y_lift: &[f64],
    states: &[f64],
    k: usize,
    n_lift: usize,
    m: usize,
    n_state: usize,
    regularisation: f64,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), String> {
    validate_inputs(
        x_lift, inputs, y_lift, states, k, n_lift, m, n_state, regularisation,
    )?;

    let p = n_lift + m;

    // Gram = ΦᵀΦ + ρI  (p×p), Cross = Φᵀ Y_lift  (p×n_lift).
    let mut gram = vec![0.0_f64; p * p];
    let mut cross = vec![0.0_f64; p * n_lift];
    for i in 0..k {
        // Build the i-th row of Φ on the fly: [x_lift row | inputs row].
        for a in 0..p {
            let phi_a = if a < n_lift {
                x_lift[i * n_lift + a]
            } else {
                inputs[i * m + (a - n_lift)]
            };
            for b in 0..p {
                let phi_b = if b < n_lift {
                    x_lift[i * n_lift + b]
                } else {
                    inputs[i * m + (b - n_lift)]
                };
                gram[a * p + b] += phi_a * phi_b;
            }
            for j in 0..n_lift {
                cross[a * n_lift + j] += phi_a * y_lift[i * n_lift + j];
            }
        }
    }
    for a in 0..p {
        gram[a * p + a] += regularisation;
    }
    // M = Gram⁻¹ Cross  (p×n_lift) = [Aᵀ; Bᵀ].
    let m_sol = solve_multi(p, &gram, &cross, n_lift)?;

    let mut a_mat = vec![0.0_f64; n_lift * n_lift];
    for i in 0..n_lift {
        for c in 0..n_lift {
            a_mat[i * n_lift + c] = m_sol[c * n_lift + i];
        }
    }
    let mut b_mat = vec![0.0_f64; n_lift * m];
    for i in 0..n_lift {
        for c in 0..m {
            b_mat[i * m + c] = m_sol[(n_lift + c) * n_lift + i];
        }
    }

    // LiftGram = X_liftᵀ X_lift + ρI (n_lift²), CC = X_liftᵀ X (n_lift×n_state).
    let mut lift_gram = vec![0.0_f64; n_lift * n_lift];
    let mut cc = vec![0.0_f64; n_lift * n_state];
    for i in 0..k {
        for a in 0..n_lift {
            let xa = x_lift[i * n_lift + a];
            for b in 0..n_lift {
                lift_gram[a * n_lift + b] += xa * x_lift[i * n_lift + b];
            }
            for j in 0..n_state {
                cc[a * n_state + j] += xa * states[i * n_state + j];
            }
        }
    }
    for a in 0..n_lift {
        lift_gram[a * n_lift + a] += regularisation;
    }
    // Ct = LiftGram⁻¹ CC (n_lift×n_state) = Cᵀ.
    let ct = solve_multi(n_lift, &lift_gram, &cc, n_state)?;
    let mut c_mat = vec![0.0_f64; n_state * n_lift];
    for i in 0..n_state {
        for c in 0..n_lift {
            c_mat[i * n_lift + c] = ct[c * n_state + i];
        }
    }

    Ok((a_mat, b_mat, c_mat))
}

#[allow(clippy::too_many_arguments)]
fn validate_inputs(
    x_lift: &[f64],
    inputs: &[f64],
    y_lift: &[f64],
    states: &[f64],
    k: usize,
    n_lift: usize,
    m: usize,
    n_state: usize,
    regularisation: f64,
) -> Result<(), String> {
    if k == 0 || n_lift == 0 || n_state == 0 {
        return Err("k, n_lift and n_state must be positive".to_string());
    }
    if x_lift.len() != k * n_lift {
        return Err("x_lift has an inconsistent length".to_string());
    }
    if y_lift.len() != k * n_lift {
        return Err("y_lift has an inconsistent length".to_string());
    }
    if inputs.len() != k * m {
        return Err("inputs has an inconsistent length".to_string());
    }
    if states.len() != k * n_state {
        return Err("states has an inconsistent length".to_string());
    }
    if !regularisation.is_finite() || regularisation < 0.0 {
        return Err("regularisation must be finite and non-negative".to_string());
    }
    Ok(())
}

/// Solve `mat · X = rhs` for `X` (dim×n_rhs) by Gaussian elimination with
/// partial pivoting; `mat` is row-major (dim×dim), `rhs` row-major (dim×n_rhs).
fn solve_multi(
    dim: usize,
    mat: &[f64],
    rhs: &[f64],
    n_rhs: usize,
) -> Result<Vec<f64>, String> {
    // Work on the augmented matrix [mat | rhs] of width dim + n_rhs.
    let width = dim + n_rhs;
    let mut aug = vec![0.0_f64; dim * width];
    for r in 0..dim {
        for c in 0..dim {
            aug[r * width + c] = mat[r * dim + c];
        }
        for c in 0..n_rhs {
            aug[r * width + dim + c] = rhs[r * n_rhs + c];
        }
    }

    for col in 0..dim {
        // Partial pivot: largest magnitude in the column at or below the diagonal.
        let mut pivot_row = col;
        let mut pivot_mag = aug[col * width + col].abs();
        for r in (col + 1)..dim {
            let mag = aug[r * width + col].abs();
            if mag > pivot_mag {
                pivot_mag = mag;
                pivot_row = r;
            }
        }
        if pivot_mag == 0.0 {
            return Err("singular system in Koopman EDMD solve".to_string());
        }
        if pivot_row != col {
            for c in 0..width {
                aug.swap(col * width + c, pivot_row * width + c);
            }
        }
        let pivot = aug[col * width + col];
        for r in 0..dim {
            if r == col {
                continue;
            }
            let factor = aug[r * width + col] / pivot;
            if factor == 0.0 {
                continue;
            }
            for c in col..width {
                aug[r * width + c] -= factor * aug[col * width + c];
            }
        }
    }

    let mut sol = vec![0.0_f64; dim * n_rhs];
    for r in 0..dim {
        let pivot = aug[r * width + r];
        for c in 0..n_rhs {
            sol[r * n_rhs + c] = aug[r * width + dim + c] / pivot;
        }
    }
    Ok(sol)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_a_linear_controlled_system() {
        // x_{k+1} = A x + B u with identity lift; EDMD must recover A, B, C=I.
        let k = 60;
        let (n, m) = (2usize, 1usize);
        let a_true = [0.9_f64, 0.1, 0.0, 0.8];
        let b_true = [0.5_f64, 0.3];
        let mut x = vec![0.0_f64; k * n];
        let mut u = vec![0.0_f64; k * m];
        let mut y = vec![0.0_f64; k * n];
        let mut seed = 1u64;
        let mut rand = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 - 1.0
        };
        for i in 0..k {
            for d in 0..n {
                x[i * n + d] = rand();
            }
            u[i * m] = rand();
            for r in 0..n {
                let mut acc = b_true[r] * u[i * m];
                for c in 0..n {
                    acc += a_true[r * n + c] * x[i * n + c];
                }
                y[i * n + r] = acc;
            }
        }
        let (a, b, c) = koopman_edmd_solve(&x, &u, &y, &x, k, n, m, n, 1e-12).unwrap();
        for idx in 0..n * n {
            assert!((a[idx] - a_true[idx]).abs() < 1e-7, "A[{idx}] mismatch");
        }
        for idx in 0..n * m {
            assert!((b[idx] - b_true[idx]).abs() < 1e-7, "B[{idx}] mismatch");
        }
        // C should be the identity selection.
        for r in 0..n {
            for col in 0..n {
                let expected = if r == col { 1.0 } else { 0.0 };
                assert!((c[r * n + col] - expected).abs() < 1e-7);
            }
        }
    }

    #[test]
    fn rejects_inconsistent_shapes() {
        // x_lift carries one element but k·n_lift = 2 — an inconsistent length.
        let err = koopman_edmd_solve(&[1.0], &[0.0], &[1.0, 2.0], &[1.0, 2.0], 1, 2, 1, 2, 1e-8);
        assert!(err.is_err());
    }
}
