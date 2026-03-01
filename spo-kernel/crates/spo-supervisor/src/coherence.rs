// SCPN Phase Orchestrator — Coherence Monitor

use spo_types::UPDEState;

pub struct CoherenceMonitor {
    good_layers: Vec<usize>,
    bad_layers: Vec<usize>,
}

impl CoherenceMonitor {
    pub fn new(good_layers: Vec<usize>, bad_layers: Vec<usize>) -> Self {
        Self {
            good_layers,
            bad_layers,
        }
    }

    pub fn compute_r_good(&self, upde_state: &UPDEState) -> f64 {
        mean_r(upde_state, &self.good_layers)
    }

    pub fn compute_r_bad(&self, upde_state: &UPDEState) -> f64 {
        mean_r(upde_state, &self.bad_layers)
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
}
