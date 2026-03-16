// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Coherence monitor

use spo_types::UPDEState;

/// Tracks coherence (order parameter R) partitioned into good vs bad layer subsets.
pub struct CoherenceMonitor {
    good_layers: Vec<usize>,
    bad_layers: Vec<usize>,
}

impl CoherenceMonitor {
    #[must_use]
    pub fn new(good_layers: Vec<usize>, bad_layers: Vec<usize>) -> Self {
        Self {
            good_layers,
            bad_layers,
        }
    }

    #[must_use]
    pub fn compute_r_good(&self, upde_state: &UPDEState) -> f64 {
        mean_r(upde_state, &self.good_layers)
    }

    #[must_use]
    pub fn compute_r_bad(&self, upde_state: &UPDEState) -> f64 {
        mean_r(upde_state, &self.bad_layers)
    }

    /// Layer pairs whose PLV exceeds `threshold` in `cross_layer_alignment`.
    ///
    /// `cla` is a flattened n×n matrix where `cla[i*n + j]` = PLV(layer_i, layer_j).
    /// Lachaux et al. 1999 — PLV ≥ 0.9 indicates phase-locking.
    #[must_use]
    pub fn detect_phase_lock(&self, upde_state: &UPDEState, threshold: f64) -> Vec<(usize, usize)> {
        let n = upde_state.layers.len();
        let cla = &upde_state.cross_layer_alignment;
        if cla.len() != n * n {
            return vec![];
        }
        let mut locked = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if cla[i * n + j] >= threshold {
                    locked.push((i, j));
                }
            }
        }
        locked
    }
}

fn mean_r(upde_state: &UPDEState, indices: &[usize]) -> f64 {
    let vals: Vec<f64> = indices
        .iter()
        .filter_map(|&i| upde_state.layers.get(i).map(|l| l.r))
        .collect();
    if vals.is_empty() {
        return 0.0;
    }
    vals.iter().sum::<f64>() / vals.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use spo_types::{LayerState, Regime};

    fn make_state(rs: &[f64]) -> UPDEState {
        UPDEState {
            layers: rs.iter().map(|&r| LayerState { r, psi: 0.0 }).collect(),
            cross_layer_alignment: vec![],
            stability_proxy: 0.0,
            regime: Regime::Nominal,
        }
    }

    #[test]
    fn good_bad_split() {
        let cm = CoherenceMonitor::new(vec![0, 1], vec![2, 3]);
        let state = make_state(&[0.9, 0.8, 0.2, 0.1]);
        assert!((cm.compute_r_good(&state) - 0.85).abs() < 1e-12);
        assert!((cm.compute_r_bad(&state) - 0.15).abs() < 1e-12);
    }

    #[test]
    fn out_of_bounds_indices() {
        let cm = CoherenceMonitor::new(vec![0, 10], vec![]);
        let state = make_state(&[0.5]);
        assert!((cm.compute_r_good(&state) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn empty_indices() {
        let cm = CoherenceMonitor::new(vec![], vec![]);
        let state = make_state(&[0.5, 0.6]);
        assert_eq!(cm.compute_r_good(&state), 0.0);
        assert_eq!(cm.compute_r_bad(&state), 0.0);
    }

    fn make_state_with_cla(rs: &[f64], cla: Vec<f64>) -> UPDEState {
        UPDEState {
            layers: rs.iter().map(|&r| LayerState { r, psi: 0.0 }).collect(),
            cross_layer_alignment: cla,
            stability_proxy: 0.0,
            regime: Regime::Nominal,
        }
    }

    #[test]
    fn detect_phase_lock_finds_locked_pairs() {
        let cm = CoherenceMonitor::new(vec![0, 1], vec![2]);
        // 3x3 CLA: (0,1)=0.95, (0,2)=0.5, (1,2)=0.92
        #[rustfmt::skip]
        let cla = vec![
            0.0, 0.95, 0.50,
            0.95, 0.0, 0.92,
            0.50, 0.92, 0.0,
        ];
        let state = make_state_with_cla(&[0.9, 0.8, 0.3], cla);
        let locked = cm.detect_phase_lock(&state, 0.9);
        assert_eq!(locked, vec![(0, 1), (1, 2)]);
    }

    #[test]
    fn detect_phase_lock_empty_on_low_plv() {
        let cm = CoherenceMonitor::new(vec![0], vec![1]);
        let cla = vec![0.0, 0.5, 0.5, 0.0];
        let state = make_state_with_cla(&[0.9, 0.8], cla);
        assert!(cm.detect_phase_lock(&state, 0.9).is_empty());
    }

    #[test]
    fn detect_phase_lock_wrong_cla_size() {
        let cm = CoherenceMonitor::new(vec![0], vec![1]);
        let state = make_state_with_cla(&[0.9, 0.8], vec![0.0, 0.95]);
        assert!(cm.detect_phase_lock(&state, 0.9).is_empty());
    }
}
