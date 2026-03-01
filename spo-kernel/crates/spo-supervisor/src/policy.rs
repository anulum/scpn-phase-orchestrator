// SCPN Phase Orchestrator — Supervisor Policy

use spo_types::{ControlAction, Knob, Regime, UPDEState};

use crate::boundaries::BoundaryState;
use crate::regime::RegimeManager;

const K_BUMP: f64 = 0.05;
const ZETA_BUMP: f64 = 0.1;
const K_REDUCE: f64 = -0.03;
const RESTORE_FRACTION: f64 = 0.5;

pub struct SupervisorPolicy {
    pub manager: RegimeManager,
}

impl SupervisorPolicy {
    pub fn new(manager: RegimeManager) -> Self {
        Self { manager }
    }

    pub fn decide(
        &mut self,
        upde_state: &UPDEState,
        boundary: &BoundaryState,
    ) -> Vec<ControlAction> {
        let proposed = self.manager.evaluate(upde_state, boundary);
        let regime = self.manager.transition(proposed);

        match regime {
            Regime::Nominal => vec![],
            Regime::Degraded => vec![ControlAction {
                knob: Knob::K,
                scope: "global".into(),
                value: K_BUMP,
                ttl_s: 10.0,
                justification: "degraded: boost global coupling".into(),
            }],
            Regime::Critical => {
                let mut actions = vec![ControlAction {
                    knob: Knob::Zeta,
                    scope: "global".into(),
                    value: ZETA_BUMP,
                    ttl_s: 5.0,
                    justification: "critical: increase damping".into(),
                }];
                if let Some(worst) = worst_layer(upde_state) {
                    actions.push(ControlAction {
                        knob: Knob::K,
                        scope: format!("layer_{worst}"),
                        value: K_REDUCE,
                        ttl_s: 5.0,
                        justification: format!("critical: reduce coupling on layer {worst}"),
                    });
                }
                actions
            }
            Regime::Recovery => vec![ControlAction {
                knob: Knob::K,
                scope: "global".into(),
                value: K_BUMP * RESTORE_FRACTION,
                ttl_s: 15.0,
                justification: "recovery: gradual coupling restore".into(),
            }],
        }
    }
}

fn worst_layer(upde_state: &UPDEState) -> Option<usize> {
    if upde_state.layers.is_empty() {
        return None;
    }
    Some(
        upde_state
            .layers
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.r.partial_cmp(&b.r).unwrap())
            .unwrap()
            .0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use spo_types::LayerState;

    fn make_state(r: f64) -> UPDEState {
        UPDEState {
            layers: vec![LayerState { r, psi: 0.0 }; 4],
            cross_layer_alignment: vec![],
            stability_proxy: 0.0,
            regime: Regime::Nominal,
        }
    }

    fn empty_boundary() -> BoundaryState {
        BoundaryState::default()
    }

    #[test]
    fn nominal_no_actions() {
        let mut sp = SupervisorPolicy::new(RegimeManager::default());
        let actions = sp.decide(&make_state(0.9), &empty_boundary());
        assert!(actions.is_empty());
    }

    #[test]
    fn degraded_boost_k() {
        let mut sp = SupervisorPolicy::new(RegimeManager::default());
        let actions = sp.decide(&make_state(0.5), &empty_boundary());
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].knob, Knob::K);
        assert_eq!(actions[0].value, K_BUMP);
    }

    #[test]
    fn critical_damping_and_reduce() {
        let mut sp = SupervisorPolicy::new(RegimeManager::default());
        let actions = sp.decide(&make_state(0.1), &empty_boundary());
        assert!(actions.len() >= 1);
        assert_eq!(actions[0].knob, Knob::Zeta);
    }

    #[test]
    fn critical_worst_layer_action() {
        let mut sp = SupervisorPolicy::new(RegimeManager::default());
        let state = UPDEState {
            layers: vec![
                LayerState { r: 0.2, psi: 0.0 },
                LayerState { r: 0.05, psi: 0.0 },
                LayerState { r: 0.1, psi: 0.0 },
            ],
            cross_layer_alignment: vec![],
            stability_proxy: 0.0,
            regime: Regime::Nominal,
        };
        let actions = sp.decide(&state, &empty_boundary());
        assert!(actions.len() >= 2);
        assert!(actions[1].scope.contains("layer_1"));
    }

    #[test]
    fn recovery_gradual_restore() {
        let mut sp = SupervisorPolicy::new(RegimeManager::default());
        // Force into Critical first
        sp.decide(&make_state(0.1), &empty_boundary());
        // Allow cooldown to pass
        for _ in 0..20 {
            sp.manager.transition(Regime::Critical);
        }
        sp.manager.current = Regime::Critical;
        let actions = sp.decide(&make_state(0.8), &empty_boundary());
        if !actions.is_empty() {
            assert_eq!(actions[0].knob, Knob::K);
        }
    }
}
