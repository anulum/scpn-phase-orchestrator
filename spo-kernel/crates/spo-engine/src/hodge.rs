// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Combinatorial Hodge decomposition of coupling flow

//! Combinatorial (Helmholtz–Hodge) decomposition of the Kuramoto coupling
//! current into gradient, curl, and harmonic edge flows.
//!
//! The oscillator network is a simplicial complex with vertices, the supplied
//! edges `(i, j)` (`i < j`), and triangles `(i, j, k)` (`i < j < k`). The
//! decomposed object is the alternating edge flow
//! `f_ij = ½(K_ij + K_ji)·sin(θ_j − θ_i)`.
//!
//! With node–edge incidence `B1` and edge–triangle incidence `B2`:
//! * `f_grad = B1ᵀ · L0⁺ · (B1 f)`   (curl-free)
//! * `f_curl = B2 · L2⁺ · (B2ᵀ f)`   (divergence-free)
//! * `f_harm = f − f_grad − f_curl`  (harmonic, ker of the Hodge 1-Laplacian)
//!
//! `L0 = B1 B1ᵀ` and `L2 = B2ᵀ B2` are symmetric positive semidefinite, so the
//! pseudoinverse is applied via the symmetric (cyclic Jacobi) eigensolver with
//! a shared relative singular-value cutoff. Each component is returned as a
//! row-major antisymmetric `N×N` matrix where `M[i, j]` is the flow on edge
//! `i → j`.
//!
//! Jiang, Lim, Yao & Ye 2011, Math. Program. 127(1):203-244.

use crate::spectral::symmetric_eigen;
use std::collections::HashMap;

/// Relative eigenvalue cutoff for the symmetric PSD pseudoinverse; mirrors the
/// Python reference `_PINV_RCOND` so the projections agree.
const PINV_RCOND: f64 = 1e-9;
/// Jacobi sweep budget for the Laplacian eigensolves.
const EIGEN_SWEEPS: usize = 60;
/// Jacobi off-diagonal convergence tolerance for the Laplacian eigensolves.
const EIGEN_TOL: f64 = 1e-14;

/// Apply the Moore–Penrose pseudoinverse of a symmetric PSD `mat` (row-major,
/// `dim×dim`) to `vec` via eigendecomposition. Eigenvalues at or below
/// `PINV_RCOND · λ_max` are treated as the null space.
fn psd_pinv_apply(mat: &[f64], dim: usize, vec: &[f64]) -> Vec<f64> {
    if dim == 0 {
        return vec![];
    }
    let decomposed = symmetric_eigen(mat, dim, EIGEN_SWEEPS, EIGEN_TOL);
    let evals = &decomposed.eigenvalues;
    let evecs = &decomposed.eigenvectors;
    let lambda_max = evals.last().copied().unwrap_or(0.0);
    let cutoff = if lambda_max > 0.0 {
        PINV_RCOND * lambda_max
    } else {
        0.0
    };
    // Component r of eigenvector k is evecs[r * dim + k].
    let mut projected = vec![0.0; dim];
    for k in 0..dim {
        if evals[k] <= cutoff {
            continue;
        }
        let mut acc = 0.0;
        for (r, &component) in vec.iter().enumerate().take(dim) {
            acc += evecs[r * dim + k] * component;
        }
        projected[k] = acc / evals[k];
    }
    let mut out = vec![0.0; dim];
    for (r, slot) in out.iter_mut().enumerate() {
        let mut acc = 0.0;
        for (k, &p) in projected.iter().enumerate() {
            acc += evecs[r * dim + k] * p;
        }
        *slot = acc;
    }
    out
}

/// Hodge decomposition result: (gradient, curl, harmonic) as row-major
/// antisymmetric `N×N` flow matrices.
///
/// `edges` is a flattened `(n_edges, 2)` int list of `i < j` pairs; `tris` is a
/// flattened `(n_tris, 3)` int list of `i < j < k` triangles whose edges exist.
#[must_use]
pub fn hodge_decomposition(
    knm_flat: &[f64],
    phases: &[f64],
    n: usize,
    edges: &[i64],
    n_edges: usize,
    tris: &[i64],
    n_tris: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    if n == 0 {
        return (vec![], vec![], vec![]);
    }

    let mut edge_i = vec![0usize; n_edges];
    let mut edge_j = vec![0usize; n_edges];
    let mut flow = vec![0.0; n_edges];
    for e in 0..n_edges {
        let i = edges[2 * e] as usize;
        let j = edges[2 * e + 1] as usize;
        edge_i[e] = i;
        edge_j[e] = j;
        let k_sym = 0.5 * (knm_flat[i * n + j] + knm_flat[j * n + i]);
        flow[e] = k_sym * (phases[j] - phases[i]).sin();
    }

    // L0 = B1 B1ᵀ and div = B1 f, built from edge incidence.
    let mut l0 = vec![0.0; n * n];
    let mut div = vec![0.0; n];
    for e in 0..n_edges {
        let i = edge_i[e];
        let j = edge_j[e];
        l0[i * n + i] += 1.0;
        l0[j * n + j] += 1.0;
        l0[i * n + j] -= 1.0;
        l0[j * n + i] -= 1.0;
        div[i] -= flow[e];
        div[j] += flow[e];
    }
    let potential = psd_pinv_apply(&l0, n, &div);
    let mut f_grad = vec![0.0; n_edges];
    for e in 0..n_edges {
        f_grad[e] = potential[edge_j[e]] - potential[edge_i[e]];
    }

    let mut f_curl = vec![0.0; n_edges];
    if n_tris > 0 {
        let mut edge_index: HashMap<(usize, usize), usize> = HashMap::with_capacity(n_edges);
        for e in 0..n_edges {
            edge_index.insert((edge_i[e], edge_j[e]), e);
        }
        // Each triangle's three signed edge entries of B2.
        let mut tri_edges = vec![[(0usize, 0.0f64); 3]; n_tris];
        for t in 0..n_tris {
            let i = tris[3 * t] as usize;
            let j = tris[3 * t + 1] as usize;
            let k = tris[3 * t + 2] as usize;
            tri_edges[t] = [
                (edge_index[&(i, j)], 1.0),
                (edge_index[&(j, k)], 1.0),
                (edge_index[&(i, k)], -1.0),
            ];
        }
        // L2 = B2ᵀ B2 and c2 = B2ᵀ f.
        let mut l2 = vec![0.0; n_tris * n_tris];
        let mut c2 = vec![0.0; n_tris];
        for t in 0..n_tris {
            for &(e_t, s_t) in &tri_edges[t] {
                c2[t] += s_t * flow[e_t];
            }
            for u in 0..n_tris {
                let mut acc = 0.0;
                for &(e_t, s_t) in &tri_edges[t] {
                    for &(e_u, s_u) in &tri_edges[u] {
                        if e_t == e_u {
                            acc += s_t * s_u;
                        }
                    }
                }
                l2[t * n_tris + u] = acc;
            }
        }
        let tri_pot = psd_pinv_apply(&l2, n_tris, &c2);
        for t in 0..n_tris {
            for &(e, sign) in &tri_edges[t] {
                f_curl[e] += sign * tri_pot[t];
            }
        }
    }

    let mut grad_m = vec![0.0; n * n];
    let mut curl_m = vec![0.0; n * n];
    let mut harm_m = vec![0.0; n * n];
    for e in 0..n_edges {
        let i = edge_i[e];
        let j = edge_j[e];
        let g = f_grad[e];
        let c = f_curl[e];
        let h = flow[e] - g - c;
        grad_m[i * n + j] = g;
        grad_m[j * n + i] = -g;
        curl_m[i * n + j] = c;
        curl_m[j * n + i] = -c;
        harm_m[i * n + j] = h;
        harm_m[j * n + i] = -h;
    }

    (grad_m, curl_m, harm_m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Complete graph on 3 nodes: edges (0,1),(0,2),(1,2), one triangle.
    fn triangle_complex() -> (Vec<i64>, Vec<i64>) {
        (vec![0, 1, 0, 2, 1, 2], vec![0, 1, 2])
    }

    #[test]
    fn empty_returns_empty() {
        let (g, c, h) = hodge_decomposition(&[], &[], 0, &[], 0, &[], 0);
        assert!(g.is_empty() && c.is_empty() && h.is_empty());
    }

    #[test]
    fn components_reconstruct_flow_and_are_antisymmetric() {
        let n = 3;
        let knm = vec![0.0, 1.0, 0.5, 1.0, 0.0, 0.8, 0.5, 0.8, 0.0];
        let phases = vec![0.0, PI / 4.0, PI / 2.0];
        let (edges, tris) = triangle_complex();
        let (g, c, h) = hodge_decomposition(&knm, &phases, n, &edges, 3, &tris, 1);
        for i in 0..n {
            for j in 0..n {
                assert!((g[i * n + j] + g[j * n + i]).abs() < 1e-12);
                assert!((c[i * n + j] + c[j * n + i]).abs() < 1e-12);
                assert!((h[i * n + j] + h[j * n + i]).abs() < 1e-12);
                let k_sym = 0.5 * (knm[i * n + j] + knm[j * n + i]);
                let flow = k_sym * (phases[j] - phases[i]).sin();
                let recon = g[i * n + j] + c[i * n + j] + h[i * n + j];
                assert!((recon - flow).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn single_triangle_has_zero_harmonic() {
        // A filled triangle (β1 = 0) has no harmonic component.
        let n = 3;
        let knm = vec![0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0];
        let phases = vec![0.0, 1.0, 2.3];
        let (edges, tris) = triangle_complex();
        let (_, _, h) = hodge_decomposition(&knm, &phases, n, &edges, 3, &tris, 1);
        for value in &h {
            assert!(value.abs() < 1e-9, "harmonic {value} should vanish");
        }
    }

    #[test]
    fn square_cycle_without_triangles_is_purely_harmonic() {
        // 4-cycle 0-1-2-3-0 with no diagonals: no triangles, β1 = 1.
        let n = 4;
        let mut knm = vec![0.0; n * n];
        for &(i, j) in &[(0usize, 1usize), (1, 2), (2, 3), (0, 3)] {
            knm[i * n + j] = 1.0;
            knm[j * n + i] = 1.0;
        }
        let phases = vec![0.0, 0.7, 1.9, 2.8];
        let edges = vec![0, 1, 0, 3, 1, 2, 2, 3];
        let (g, c, h) = hodge_decomposition(&knm, &phases, n, &edges, 4, &[], 0);
        let curl_max = c.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let harm_max = h.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(
            curl_max < 1e-12,
            "curl {curl_max} should vanish without triangles"
        );
        assert!(harm_max > 1e-3, "harmonic {harm_max} should be non-trivial");
        for i in 0..n {
            for j in 0..n {
                let k_sym = 0.5 * (knm[i * n + j] + knm[j * n + i]);
                let flow = k_sym * (phases[j] - phases[i]).sin();
                let recon = g[i * n + j] + c[i * n + j] + h[i * n + j];
                assert!((recon - flow).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn path_tree_is_pure_gradient() {
        // Path 0-1-2: acyclic, so curl and harmonic vanish.
        let n = 3;
        let mut knm = vec![0.0; n * n];
        for &(i, j) in &[(0usize, 1usize), (1, 2)] {
            knm[i * n + j] = 1.0;
            knm[j * n + i] = 1.0;
        }
        let phases = vec![0.1, 0.9, 1.7];
        let edges = vec![0, 1, 1, 2];
        let (g, c, h) = hodge_decomposition(&knm, &phases, n, &edges, 2, &[], 0);
        for i in 0..n {
            for j in 0..n {
                assert!(c[i * n + j].abs() < 1e-12);
                assert!(h[i * n + j].abs() < 1e-9);
                let k_sym = 0.5 * (knm[i * n + j] + knm[j * n + i]);
                let flow = k_sym * (phases[j] - phases[i]).sin();
                assert!((g[i * n + j] - flow).abs() < 1e-9);
            }
        }
    }
}
