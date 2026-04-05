// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Swarmalator dynamics (O'Keeffe et al. 2017)

//! Swarmalator model: coupled spatial + phase dynamics.
//!
//! Position: ẋ_i = (1/N) Σ_j [(x_j−x_i)/|·| · (A + J cos(Δθ)) − B (x_j−x_i)/|·|³]
//! Phase: θ̇_i = ω_i + (K/N) Σ_j sin(Δθ) / |x_j−x_i|
//!
//! O'Keeffe, Hong & Strogatz 2017, Nature Communications 8:1504.

use std::f64::consts::TAU;

/// Run n_steps of swarmalator dynamics.
///
/// Returns (final_pos_flat, final_phases, pos_traj_flat, phase_traj_flat).
/// Position arrays are row-major: pos_flat[i*dim+d] = position of agent i, dimension d.
/// Trajectories: pos_traj[step*n*dim + i*dim + d], phase_traj[step*n + i].
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn swarmalator_run(
    pos_init: &[f64],
    phases_init: &[f64],
    omegas: &[f64],
    n: usize,
    dim: usize,
    dt: f64,
    a: f64,
    b: f64,
    j: f64,
    k: f64,
    n_steps: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let eps = 1e-6;
    let inv_n = 1.0 / n as f64;

    let mut pos = pos_init.to_vec();
    let mut phases = phases_init.to_vec();
    let mut pos_traj = Vec::with_capacity(n_steps * n * dim);
    let mut phase_traj = Vec::with_capacity(n_steps * n);

    for _ in 0..n_steps {
        let mut new_pos = vec![0.0; n * dim];
        let mut new_phases = vec![0.0; n];

        for i in 0..n {
            let mut dx = vec![0.0; dim];
            let mut dtheta = omegas[i];

            for jj in 0..n {
                if i == jj {
                    continue;
                }

                // Displacement and distance
                let mut dist_sq = 0.0;
                for d in 0..dim {
                    let delta = pos[jj * dim + d] - pos[i * dim + d];
                    dist_sq += delta * delta;
                }
                let dist = (dist_sq + eps).sqrt();
                let inv_dist = 1.0 / dist;

                let cos_diff = (phases[jj] - phases[i]).cos();
                let sin_diff = (phases[jj] - phases[i]).sin();

                // Position dynamics
                let attract_coeff = (a + j * cos_diff) * inv_dist;
                let repulse_coeff = b / (dist * dist * dist + eps);

                for d in 0..dim {
                    let delta = pos[jj * dim + d] - pos[i * dim + d];
                    dx[d] += delta * attract_coeff - delta * repulse_coeff;
                }

                // Phase dynamics: K/N sin(Δθ) / |x_j - x_i|
                dtheta += k * inv_n * sin_diff * inv_dist;
            }

            // Apply position update
            for d in 0..dim {
                new_pos[i * dim + d] = pos[i * dim + d] + dt * dx[d] * inv_n;
            }
            // Apply phase update
            let raw = phases[i] + dt * dtheta;
            new_phases[i] = ((raw % TAU) + TAU) % TAU;
        }

        pos = new_pos;
        phases = new_phases;
        pos_traj.extend_from_slice(&pos);
        phase_traj.extend_from_slice(&phases);
    }

    (pos, phases, pos_traj, phase_traj)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_preserves_length() {
        let n = 5;
        let dim = 2;
        let pos = vec![0.0; n * dim];
        let phases = vec![0.0; n];
        let omegas = vec![1.0; n];

        let (fp, fph, pt, pht) = swarmalator_run(
            &pos, &phases, &omegas, n, dim, 0.01, 1.0, 1.0, 0.5, 1.0, 10,
        );
        assert_eq!(fp.len(), n * dim);
        assert_eq!(fph.len(), n);
        assert_eq!(pt.len(), 10 * n * dim);
        assert_eq!(pht.len(), 10 * n);
    }

    #[test]
    fn test_phases_bounded() {
        let n = 8;
        let dim = 2;
        let mut pos = Vec::with_capacity(n * dim);
        for i in 0..n {
            pos.push((i as f64 * 0.5).cos());
            pos.push((i as f64 * 0.5).sin());
        }
        let phases: Vec<f64> = (0..n).map(|i| TAU * i as f64 / n as f64).collect();
        let omegas: Vec<f64> = (0..n).map(|i| 0.5 + 0.1 * i as f64).collect();

        let (_, fph, _, _) = swarmalator_run(
            &pos, &phases, &omegas, n, dim, 0.01, 1.0, 1.0, 0.5, 1.0, 200,
        );
        for (i, &v) in fph.iter().enumerate() {
            assert!(v >= 0.0 && v < TAU, "phase[{i}] = {v} out of [0, 2π)");
        }
    }

    #[test]
    fn test_no_coupling_drift() {
        // J=0, K=0 → phases drift freely, positions drift from repulsion only
        let n = 3;
        let dim = 2;
        let pos = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let phases = vec![0.0, PI / 2.0, PI];
        let omegas = vec![1.0, 2.0, 3.0];

        let (_, fph, _, _) = swarmalator_run(
            &pos, &phases, &omegas, n, dim, 0.01, 0.0, 1.0, 0.0, 0.0, 100,
        );
        // Different omegas → different final phases
        assert!((fph[0] - fph[1]).abs() > 0.1 || (fph[1] - fph[2]).abs() > 0.1);
    }

    #[test]
    fn test_deterministic() {
        let n = 4;
        let dim = 2;
        let pos: Vec<f64> = (0..n * dim).map(|i| i as f64 * 0.3).collect();
        let phases: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
        let omegas = vec![1.0; n];

        let r1 = swarmalator_run(&pos, &phases, &omegas, n, dim, 0.01, 1.0, 1.0, 0.5, 1.0, 50);
        let r2 = swarmalator_run(&pos, &phases, &omegas, n, dim, 0.01, 1.0, 1.0, 0.5, 1.0, 50);
        assert_eq!(r1.0, r2.0);
        assert_eq!(r1.1, r2.1);
    }

    #[test]
    fn test_attraction_pulls_together() {
        // Strong attraction, no repulsion → agents should converge
        let n = 3;
        let dim = 2;
        let pos = vec![0.0, 0.0, 5.0, 0.0, 0.0, 5.0]; // spread out
        let phases = vec![0.0; n]; // same phase → max attraction
        let omegas = vec![0.0; n];

        let (fp, _, _, _) = swarmalator_run(
            &pos, &phases, &omegas, n, dim, 0.01, 5.0, 0.0, 1.0, 0.0, 500,
        );
        // Compute max pairwise distance
        let mut max_dist = 0.0_f64;
        for i in 0..n {
            for jj in i + 1..n {
                let mut d2 = 0.0;
                for d in 0..dim {
                    let delta = fp[i * dim + d] - fp[jj * dim + d];
                    d2 += delta * delta;
                }
                max_dist = max_dist.max(d2.sqrt());
            }
        }
        assert!(max_dist < 5.0, "agents should converge, max_dist = {max_dist}");
    }

    #[test]
    fn test_3d_works() {
        let n = 4;
        let dim = 3;
        let pos = vec![0.0; n * dim];
        let phases = vec![0.0; n];
        let omegas = vec![1.0; n];

        let (fp, fph, _, _) = swarmalator_run(
            &pos, &phases, &omegas, n, dim, 0.01, 1.0, 1.0, 0.5, 1.0, 10,
        );
        assert_eq!(fp.len(), n * dim);
        assert_eq!(fph.len(), n);
    }

    #[test]
    fn test_single_agent() {
        let (fp, fph, _, _) = swarmalator_run(
            &[1.0, 2.0], &[0.5], &[1.0], 1, 2, 0.01, 1.0, 1.0, 0.5, 1.0, 10,
        );
        assert_eq!(fp.len(), 2);
        assert_eq!(fph.len(), 1);
    }
}
