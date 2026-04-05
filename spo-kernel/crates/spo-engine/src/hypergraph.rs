// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Generalised k-body hypergraph Kuramoto coupling

//! Arbitrary k-body interactions beyond pairwise.
//!
//! For a k-hyperedge {i₁, ..., iₖ}, the coupling on oscillator iₘ is:
//!     σ · sin(Σ_{j≠m} θ_{iⱼ} - (k-1)·θ_{iₘ})
//!
//! This generalises:
//!     k=2: sin(θ_j - θ_i)         — standard Kuramoto
//!     k=3: sin(θ_j + θ_k - 2θ_i)  — simplicial/triadic
//!
//! Tanaka & Aoyagi 2011, Phys. Rev. Lett. 106:224101.
//! Skardal & Arenas 2019, Comm. Phys. 2:22.
//! Bick et al. 2023, Nat. Rev. Physics 5:307-317.

use std::f64::consts::TAU;

/// A single hyperedge: list of node indices + coupling strength.
pub struct Hyperedge {
    pub nodes: Vec<usize>,
    pub strength: f64,
}

/// Run hypergraph Kuramoto integration (Euler) for `n_steps`.
///
/// # Arguments
/// * `phases` – initial phases, length `n`
/// * `omegas` – natural frequencies, length `n`
/// * `n` – number of oscillators
/// * `edges` – list of hyperedges (node indices + strength)
/// * `pairwise_knm` – optional row-major pairwise coupling matrix (length `n*n`), or empty
/// * `alpha` – optional row-major phase-lag matrix (length `n*n`), or empty
/// * `zeta` – external drive strength
/// * `psi` – external drive phase
/// * `dt` – time step
/// * `n_steps` – number of Euler steps
#[must_use]
pub fn hypergraph_run(
    phases: &[f64],
    omegas: &[f64],
    n: usize,
    edges: &[Hyperedge],
    pairwise_knm: &[f64],
    alpha: &[f64],
    zeta: f64,
    psi: f64,
    dt: f64,
    n_steps: usize,
) -> Vec<f64> {
    let mut p = phases.to_vec();

    for _ in 0..n_steps {
        let dtheta = hypergraph_derivative(&p, omegas, n, edges, pairwise_knm, alpha, zeta, psi);
        for i in 0..n {
            p[i] = (p[i] + dt * dtheta[i]).rem_euclid(TAU);
        }
    }

    p
}

/// Compute full derivative for hypergraph Kuramoto.
fn hypergraph_derivative(
    p: &[f64],
    omegas: &[f64],
    n: usize,
    edges: &[Hyperedge],
    pairwise_knm: &[f64],
    alpha: &[f64],
    zeta: f64,
    psi: f64,
) -> Vec<f64> {
    let mut dtheta = omegas.to_vec();
    let has_pairwise = pairwise_knm.len() == n * n;
    let has_alpha = alpha.len() == n * n;

    if has_pairwise {
        for i in 0..n {
            let mut pair_sum = 0.0;
            for j in 0..n {
                let a_ij = if has_alpha { alpha[i * n + j] } else { 0.0 };
                pair_sum += pairwise_knm[i * n + j] * (p[j] - p[i] - a_ij).sin();
            }
            dtheta[i] += pair_sum;
        }
    }

    for edge in edges {
        let k = edge.nodes.len();
        let phase_sum: f64 = edge.nodes.iter().map(|&idx| p[idx]).sum();
        for &m in &edge.nodes {
            dtheta[m] += edge.strength * (phase_sum - (k as f64) * p[m]).sin();
        }
    }

    if zeta != 0.0 {
        for i in 0..n {
            dtheta[i] += zeta * (psi - p[i]).sin();
        }
    }

    dtheta
}

/// Compute Kuramoto order parameter R = |<exp(iθ)>|.
#[must_use]
pub fn order_parameter(phases: &[f64]) -> f64 {
    if phases.is_empty() {
        return 0.0;
    }
    let n = phases.len() as f64;
    let sx: f64 = phases.iter().map(|p| p.sin()).sum();
    let cx: f64 = phases.iter().map(|p| p.cos()).sum();
    (sx * sx + cx * cx).sqrt() / n
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_no_edges_free_rotation() {
        let n = 3;
        let phases = vec![0.0, PI / 2.0, PI];
        let omegas = vec![1.0, 2.0, 3.0];
        let result = hypergraph_run(&phases, &omegas, n, &[], &[], &[], 0.0, 0.0, 0.01, 100);
        for i in 0..n {
            let expected = (phases[i] + omegas[i] * 0.01 * 100.0).rem_euclid(TAU);
            assert!(
                (result[i] - expected).abs() < 1e-6,
                "i={i}: got {}, expected {expected}",
                result[i]
            );
        }
    }

    #[test]
    fn test_pairwise_only_synchronises() {
        let n = 5;
        let phases = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let omegas = vec![1.0; n];
        let mut knm = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    knm[i * n + j] = 2.0;
                }
            }
        }
        let result = hypergraph_run(&phases, &omegas, n, &[], &knm, &[], 0.0, 0.0, 0.01, 1000);
        let r = order_parameter(&result);
        assert!(r > 0.8, "R={r} should be > 0.8");
    }

    #[test]
    fn test_3body_edge_effect() {
        let n = 4;
        let phases = vec![0.0, 0.5, 1.0, 1.5];
        let omegas = vec![1.0; n];
        let edges = vec![
            Hyperedge {
                nodes: vec![0, 1, 2],
                strength: 2.0,
            },
            Hyperedge {
                nodes: vec![1, 2, 3],
                strength: 2.0,
            },
        ];
        let without = hypergraph_run(&phases, &omegas, n, &[], &[], &[], 0.0, 0.0, 0.01, 50);
        let with = hypergraph_run(&phases, &omegas, n, &edges, &[], &[], 0.0, 0.0, 0.01, 50);
        let mut diff = 0.0;
        for i in 0..n {
            diff += (without[i] - with[i]).abs();
        }
        assert!(diff > 1e-4, "3-body edges had no effect: diff={diff}");
    }

    #[test]
    fn test_4body_edge() {
        let n = 4;
        let phases = vec![0.0, 0.3, 0.6, 0.9];
        let omegas = vec![1.0; n];
        let edges = vec![Hyperedge {
            nodes: vec![0, 1, 2, 3],
            strength: 1.0,
        }];
        let result = hypergraph_run(&phases, &omegas, n, &edges, &[], &[], 0.0, 0.0, 0.01, 100);
        for p in &result {
            assert!(*p >= 0.0 && *p < TAU, "phase {p} out of range");
        }
    }

    #[test]
    fn test_mixed_order_edges() {
        let n = 5;
        let phases = vec![0.0, 0.4, 0.8, 1.2, 1.6];
        let omegas = vec![1.0; n];
        let edges = vec![
            Hyperedge {
                nodes: vec![0, 1],
                strength: 1.0,
            }, // pairwise
            Hyperedge {
                nodes: vec![1, 2, 3],
                strength: 0.5,
            }, // 3-body
            Hyperedge {
                nodes: vec![2, 3, 4, 0],
                strength: 0.3,
            }, // 4-body
        ];
        let result = hypergraph_run(&phases, &omegas, n, &edges, &[], &[], 0.0, 0.0, 0.005, 200);
        for p in &result {
            assert!(*p >= 0.0 && *p < TAU);
        }
    }

    #[test]
    fn test_external_drive() {
        let n = 3;
        let phases = vec![0.0; n];
        let omegas = vec![0.0; n];
        let result = hypergraph_run(&phases, &omegas, n, &[], &[], &[], 1.0, PI / 2.0, 0.01, 100);
        for p in &result {
            assert!(*p > 0.0, "should move toward psi");
        }
    }

    #[test]
    fn test_zero_steps() {
        let phases = vec![1.0, 2.0, 3.0];
        let omegas = vec![1.0; 3];
        let result = hypergraph_run(&phases, &omegas, 3, &[], &[], &[], 0.0, 0.0, 0.01, 0);
        for (a, b) in result.iter().zip(phases.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn test_order_parameter_synchronised() {
        let phases = vec![1.0, 1.0, 1.0, 1.0];
        assert!((order_parameter(&phases) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_order_parameter_uniform() {
        // Evenly spaced phases: R ≈ 0
        let n = 100;
        let phases: Vec<f64> = (0..n).map(|i| TAU * i as f64 / n as f64).collect();
        assert!(order_parameter(&phases) < 0.05);
    }
}
