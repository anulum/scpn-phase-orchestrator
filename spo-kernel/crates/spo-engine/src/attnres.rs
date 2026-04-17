// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — AttnRes coupling modulation (full multi-head)

//! Full multi-head Attention-Residuals port
//! (arXiv:2603.15031 Moonshot/Kimi 2026) applied to the SCPN coupling
//! matrix. Mirrors
//! ``src/scpn_phase_orchestrator/coupling/attention_residuals.py``
//! bit-for-bit.
//!
//! The previous single-equation Hebbian proxy has been retired per
//! the ``feedback_no_simplistic_models.md`` rule — this file
//! implements the full Transformer stack: Fourier-feature embedding,
//! per-head Q/K/V projections, scaled dot-product attention with
//! softmax, optional local mask, output projection, symmetric
//! pairwise aggregation onto the coupling matrix.
//!
//! Input layout (row-major flat arrays):
//!
//! * ``knm``        — ``(N, N)`` coupling matrix, symmetric, zero diagonal.
//! * ``theta``      — ``(N,)`` phase vector.
//! * ``w_q``        — ``(H, D, D_h)`` per-head query projection.
//! * ``w_k``        — ``(H, D, D_h)`` per-head key projection.
//! * ``w_v``        — ``(H, D, D_h)`` per-head value projection.
//! * ``w_o``        — ``(H·D_h, D)`` output projection.
//!
//! Here ``D`` is the hidden-state dimension (even; paper lifts the
//! scalar phase onto ``[cos θ, sin θ, cos 2θ, sin 2θ, …]``) and
//! ``D_h = D / H``.

/// Compute the full multi-head AttnRes modulated coupling matrix.
///
/// # Errors
/// Returns `Err` for shape / parameter mismatches.
#[allow(clippy::too_many_arguments)]
pub fn attnres_modulate(
    knm: &[f64],
    theta: &[f64],
    w_q: &[f64],
    w_k: &[f64],
    w_v: &[f64],
    w_o: &[f64],
    n: usize,
    n_heads: usize,
    block_size: i64,
    temperature: f64,
    lambda_: f64,
) -> Result<Vec<f64>, String> {
    // Validation --------------------------------------------------------
    if knm.len() != n * n {
        return Err(format!("knm length {} != n*n {}", knm.len(), n * n));
    }
    if theta.len() != n {
        return Err(format!("theta length {} != n {}", theta.len(), n));
    }
    if n_heads == 0 {
        return Err("n_heads must be ≥ 1".into());
    }
    if temperature <= 0.0 || !temperature.is_finite() {
        return Err("temperature must be finite and > 0".into());
    }
    if lambda_ < 0.0 {
        return Err("lambda_ must be ≥ 0".into());
    }
    // Infer d_model and d_head from w_q length.
    if w_q.len() % n_heads != 0 {
        return Err(format!(
            "w_q length {} not divisible by n_heads {}",
            w_q.len(),
            n_heads
        ));
    }
    let per_head = w_q.len() / n_heads; // = d_model * d_head
    // Use w_o: (H·d_head, d_model) == (n_heads * d_head) rows × d_model cols
    // → w_o.len() = n_heads * d_head * d_model, and per_head = d_model * d_head,
    // so w_o.len() == n_heads * per_head. That doesn't pin d_model; we rely on
    // the caller shapes. The NumPy side always passes d_model = PHASE_EMBED_DIM
    // (default 8). We infer d_model via gcd-ish: w_q is (H, D, D_h) so per_head =
    // D * D_h, and w_o is (H·D_h, D) so w_o.len() = H * D_h * D → D · D_h = D * D_h
    // must hold; we need a second hint. Require w_v, w_k, w_k match w_q in length
    // and w_o.len() == H * per_head. Then resolve D_h = w_o.len() / (H·D). The
    // caller always ships d_model divisible by H, so D_h = D / H.
    if w_k.len() != w_q.len() || w_v.len() != w_q.len() {
        return Err("w_k / w_v must match w_q in length".into());
    }
    if w_o.len() != n_heads * per_head {
        return Err(format!(
            "w_o length {} != n_heads·per_head {}",
            w_o.len(),
            n_heads * per_head
        ));
    }
    // Solve D_h from `per_head = D·D_h` and `D = H·D_h` (d_model even-split).
    // Then D_h² = per_head / H  →  D_h = sqrt(per_head / H).
    let per_head_per_h = per_head as f64 / n_heads as f64;
    let d_head = per_head_per_h.sqrt().round() as usize;
    if d_head * d_head * n_heads != per_head {
        return Err(format!(
            "cannot infer d_head from w_q shape: per_head={per_head}, n_heads={n_heads}"
        ));
    }
    let d_model = n_heads * d_head;

    if lambda_ == 0.0 {
        return Ok(knm.to_vec());
    }

    // 1. Fourier-feature embedding (n, d_model).
    let mut x = vec![0.0_f64; n * d_model];
    for i in 0..n {
        for h_idx in 0..(d_model / 2) {
            let freq = (h_idx + 1) as f64;
            x[i * d_model + 2 * h_idx] = (freq * theta[i]).cos();
            x[i * d_model + 2 * h_idx + 1] = (freq * theta[i]).sin();
        }
    }

    // 2. Per-head Q, K, V via W·X.  q[h, i, e] = sum_d x[i, d] * w_q[h, d, e].
    let mut q = vec![0.0_f64; n_heads * n * d_head];
    let mut k = vec![0.0_f64; n_heads * n * d_head];
    let mut v = vec![0.0_f64; n_heads * n * d_head];
    for h in 0..n_heads {
        for i in 0..n {
            for e in 0..d_head {
                let mut qs = 0.0_f64;
                let mut ks = 0.0_f64;
                let mut vs = 0.0_f64;
                for d in 0..d_model {
                    let xd = x[i * d_model + d];
                    qs += xd * w_q[h * d_model * d_head + d * d_head + e];
                    ks += xd * w_k[h * d_model * d_head + d * d_head + e];
                    vs += xd * w_v[h * d_model * d_head + d * d_head + e];
                }
                q[h * n * d_head + i * d_head + e] = qs;
                k[h * n * d_head + i * d_head + e] = ks;
                v[h * n * d_head + i * d_head + e] = vs;
            }
        }
    }

    // 3. Attention logits per head:
    //   logits[h, i, j] = sum_e q[h,i,e] * k[h,j,e] / (sqrt(d_h) * temp).
    let inv_scale = 1.0 / ((d_head as f64).sqrt() * temperature);
    let mut attn = vec![0.0_f64; n_heads * n * n];
    for h in 0..n_heads {
        for i in 0..n {
            // Build logit row for (h, i).
            let mut row_logits = vec![f64::NEG_INFINITY; n];
            let mut any_unmasked = false;
            for j in 0..n {
                if i == j {
                    continue;
                }
                if knm[i * n + j] == 0.0 {
                    continue;
                }
                if block_size >= 0 {
                    let diff = (i as i64 - j as i64).abs();
                    if diff > block_size {
                        continue;
                    }
                }
                let mut dot = 0.0_f64;
                for e in 0..d_head {
                    dot += q[h * n * d_head + i * d_head + e]
                        * k[h * n * d_head + j * d_head + e];
                }
                row_logits[j] = dot * inv_scale;
                any_unmasked = true;
            }
            if !any_unmasked {
                // leave attn row zero
                continue;
            }
            // Softmax.
            let mut row_max = f64::NEG_INFINITY;
            for &x_val in &row_logits {
                if x_val > row_max {
                    row_max = x_val;
                }
            }
            let mut denom = 0.0_f64;
            for j in 0..n {
                if row_logits[j].is_finite() {
                    let e = (row_logits[j] - row_max).exp();
                    row_logits[j] = e;
                    denom += e;
                } else {
                    row_logits[j] = 0.0;
                }
            }
            if denom > 0.0 {
                let inv_denom = 1.0 / denom;
                for j in 0..n {
                    attn[h * n * n + i * n + j] = row_logits[j] * inv_denom;
                }
            }
        }
    }

    // 4. Apply attention to values: heads[h, i, e] = sum_j attn[h,i,j] * v[h,j,e].
    // Concatenate along the d-axis: concat[i, h*d_head + e] = heads[h, i, e].
    let concat_width = n_heads * d_head;
    let mut concat = vec![0.0_f64; n * concat_width];
    for h in 0..n_heads {
        for i in 0..n {
            for e in 0..d_head {
                let mut s = 0.0_f64;
                for j in 0..n {
                    s += attn[h * n * n + i * n + j]
                        * v[h * n * d_head + j * d_head + e];
                }
                concat[i * concat_width + h * d_head + e] = s;
            }
        }
    }

    // 5. Output projection: o[i, d_out] = sum_c concat[i, c] * w_o[c, d_out].
    let mut o = vec![0.0_f64; n * d_model];
    for i in 0..n {
        for d_out in 0..d_model {
            let mut s = 0.0_f64;
            for c in 0..concat_width {
                s += concat[i * concat_width + c] * w_o[c * d_model + d_out];
            }
            o[i * d_model + d_out] = s;
        }
    }

    // 6. Pairwise similarity a_agg[i, j] = 0.5 * (1 + cos_sim(o[i], o[j])).
    let mut o_norm = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = 0.0_f64;
        for d in 0..d_model {
            let val = o[i * d_model + d];
            s += val * val;
        }
        o_norm[i] = s.sqrt() + 1e-12;
    }
    let mut a_agg = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            // Respect the band + zero-edge mask again (a_agg is a
            // full cosine matrix but only masked entries matter).
            if knm[i * n + j] == 0.0 {
                continue;
            }
            if block_size >= 0 {
                let diff = (i as i64 - j as i64).abs();
                if diff > block_size {
                    continue;
                }
            }
            let mut dot = 0.0_f64;
            for d in 0..d_model {
                dot += o[i * d_model + d] * o[j * d_model + d];
            }
            let cos_sim = dot / (o_norm[i] * o_norm[j]);
            a_agg[i * n + j] = 0.5 * (1.0 + cos_sim);
        }
    }

    // 7. Row-wise modulation then symmetrisation.
    let mut rowwise = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            rowwise[i * n + j] = knm[i * n + j] * (1.0 + lambda_ * a_agg[i * n + j]);
        }
    }
    let mut out = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            out[i * n + j] = 0.5 * (rowwise[i * n + j] + rowwise[j * n + i]);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_projections(
        n_heads: usize,
        d_model: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let d_head = d_model / n_heads;
        // Identity-like: head h gets the slice of X[:, h*d_head .. (h+1)*d_head].
        let mut w = vec![0.0_f64; n_heads * d_model * d_head];
        for h in 0..n_heads {
            for e in 0..d_head {
                let d = h * d_head + e;
                w[h * d_model * d_head + d * d_head + e] = 1.0;
            }
        }
        let w_o = vec![1.0_f64; n_heads * d_head * d_model]
            .iter()
            .enumerate()
            .map(|(idx, _)| {
                // Make w_o the inverse of concat: the (c, d_out) entry
                // is 1 iff concat column c maps back to d_out, else 0.
                // Identity concat → identity output.
                if idx % (d_model + 1) == 0 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        (w.clone(), w.clone(), w, w_o)
    }

    fn ring_knm(n: usize, strength: f64) -> Vec<f64> {
        let mut knm = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    knm[i * n + j] = strength;
                }
            }
        }
        knm
    }

    #[test]
    fn symmetry_preserved() {
        let n = 8;
        let n_heads = 4;
        let d_model = 8;
        let knm = ring_knm(n, 0.3);
        let theta: Vec<f64> = (0..n).map(|i| (i as f64) * 0.3).collect();
        let (q, k, v, o) = identity_projections(n_heads, d_model);
        let out =
            attnres_modulate(&knm, &theta, &q, &k, &v, &o, n, n_heads, -1, 1.0, 0.5)
                .unwrap();
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
    fn zero_diagonal() {
        let n = 6;
        let n_heads = 2;
        let d_model = 4;
        let knm = ring_knm(n, 0.2);
        let theta = vec![0.0; n];
        let (q, k, v, o) = identity_projections(n_heads, d_model);
        let out =
            attnres_modulate(&knm, &theta, &q, &k, &v, &o, n, n_heads, -1, 1.0, 0.5)
                .unwrap();
        for i in 0..n {
            assert_eq!(out[i * n + i], 0.0);
        }
    }

    #[test]
    fn lambda_zero_identity() {
        let n = 6;
        let n_heads = 2;
        let d_model = 4;
        let knm = ring_knm(n, 0.2);
        let theta: Vec<f64> = (0..n).map(|i| i as f64 * 0.4).collect();
        let (q, k, v, o) = identity_projections(n_heads, d_model);
        let out =
            attnres_modulate(&knm, &theta, &q, &k, &v, &o, n, n_heads, -1, 1.0, 0.0)
                .unwrap();
        assert_eq!(out, knm);
    }

    #[test]
    fn zero_edges_stay_zero() {
        let n = 5;
        let n_heads = 2;
        let d_model = 4;
        let mut knm = ring_knm(n, 0.3);
        knm[0 * n + 2] = 0.0;
        knm[2 * n + 0] = 0.0;
        let theta: Vec<f64> = (0..n).map(|i| i as f64 * 0.2).collect();
        let (q, k, v, o) = identity_projections(n_heads, d_model);
        let out =
            attnres_modulate(&knm, &theta, &q, &k, &v, &o, n, n_heads, -1, 1.0, 0.5)
                .unwrap();
        assert_eq!(out[0 * n + 2], 0.0);
        assert_eq!(out[2 * n + 0], 0.0);
    }

    #[test]
    fn shape_mismatches_rejected() {
        let (q, k, v, o) = identity_projections(2, 4);
        assert!(
            attnres_modulate(&[0.0; 9], &[0.0; 4], &q, &k, &v, &o, 4, 2, -1, 1.0, 0.5)
                .is_err()
        );
        assert!(
            attnres_modulate(&[0.0; 16], &[0.0; 3], &q, &k, &v, &o, 4, 2, -1, 1.0, 0.5)
                .is_err()
        );
    }

    #[test]
    fn invalid_hyperparams_rejected() {
        let (q, k, v, o) = identity_projections(2, 4);
        assert!(
            attnres_modulate(&[0.0; 16], &[0.0; 4], &q, &k, &v, &o, 4, 0, -1, 1.0, 0.5)
                .is_err()
        );
        assert!(
            attnres_modulate(&[0.0; 16], &[0.0; 4], &q, &k, &v, &o, 4, 2, -1, 0.0, 0.5)
                .is_err()
        );
        assert!(
            attnres_modulate(
                &[0.0; 16],
                &[0.0; 4],
                &q,
                &k,
                &v,
                &o,
                4,
                2,
                -1,
                1.0,
                -0.1
            )
            .is_err()
        );
    }
}
