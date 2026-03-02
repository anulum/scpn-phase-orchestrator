// SCPN Phase Orchestrator — Regime FSM

use spo_types::{Regime, UPDEState};

use crate::boundaries::BoundaryState;

const R_CRITICAL: f64 = 0.3;
const R_DEGRADED: f64 = 0.6;

pub struct RegimeManager {
    pub current: Regime,
    hysteresis: f64,
    cooldown_steps: u64,
    step_counter: u64,
    last_transition: u64,
}

impl RegimeManager {
    #[must_use]
    pub fn new(hysteresis: f64, cooldown_steps: u64) -> Self {
        Self {
            current: Regime::Nominal,
            hysteresis,
            cooldown_steps,
            step_counter: 0,
            last_transition: 0,
        }
    }

    #[must_use]
    pub fn evaluate(&self, upde_state: &UPDEState, boundary: &BoundaryState) -> Regime {
        if !boundary.hard_violations.is_empty() {
            return Regime::Critical;
        }
        let avg_r = upde_state.mean_r();

        // Downward transitions use bare thresholds
        if avg_r < R_CRITICAL {
            return Regime::Critical;
        }

        // Upward transitions: require threshold + hysteresis to prevent oscillation.
        // From Critical/Recovery: need R >= R_CRITICAL + h to leave Critical zone,
        // but we already passed R >= R_CRITICAL above, so check the band boundary.
        let is_recovering = matches!(self.current, Regime::Critical | Regime::Recovery);

        if avg_r < R_DEGRADED {
            if is_recovering {
                return Regime::Recovery;
            }
            return Regime::Degraded;
        }

        // avg_r >= R_DEGRADED — candidate is Nominal
        // Hysteresis band: stay in Degraded until R exceeds R_DEGRADED + hysteresis
        if self.current == Regime::Degraded && avg_r < R_DEGRADED + self.hysteresis {
            return Regime::Degraded;
        }
        if is_recovering && avg_r < R_DEGRADED + self.hysteresis {
            return Regime::Recovery;
        }

        Regime::Nominal
    }

    pub fn transition(&mut self, proposed: Regime) -> Regime {
        self.step_counter += 1;

        if proposed == self.current {
            return self.current;
        }

        let in_cooldown = self.last_transition > 0
            && self.step_counter.saturating_sub(self.last_transition) < self.cooldown_steps;
        if in_cooldown && proposed != Regime::Critical {
            return self.current;
        }

        self.last_transition = self.step_counter;
        self.current = proposed;
        proposed
    }

    #[must_use]
    pub fn step_counter(&self) -> u64 {
        self.step_counter
    }
}

impl Default for RegimeManager {
    fn default() -> Self {
        Self::new(0.05, 10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spo_types::LayerState;

    fn make_state(r: f64) -> UPDEState {
        UPDEState {
            layers: vec![LayerState { r, psi: 0.0 }],
            cross_layer_alignment: vec![],
            stability_proxy: 0.0,
            regime: Regime::Nominal,
        }
    }

    fn empty_boundary() -> BoundaryState {
        BoundaryState::default()
    }

    #[test]
    fn high_r_nominal() {
        let rm = RegimeManager::default();
        let regime = rm.evaluate(&make_state(0.9), &empty_boundary());
        assert_eq!(regime, Regime::Nominal);
    }

    #[test]
    fn low_r_degraded() {
        let rm = RegimeManager::default();
        let regime = rm.evaluate(&make_state(0.5), &empty_boundary());
        assert_eq!(regime, Regime::Degraded);
    }

    #[test]
    fn very_low_r_critical() {
        let rm = RegimeManager::default();
        let regime = rm.evaluate(&make_state(0.1), &empty_boundary());
        assert_eq!(regime, Regime::Critical);
    }

    #[test]
    fn hard_violation_critical() {
        let rm = RegimeManager::default();
        let mut bs = empty_boundary();
        bs.hard_violations.push("test violation".into());
        let regime = rm.evaluate(&make_state(0.9), &bs);
        assert_eq!(regime, Regime::Critical);
    }

    #[test]
    fn recovery_from_critical_in_degraded_band() {
        let mut rm = RegimeManager::default();
        rm.current = Regime::Critical;
        // R=0.5 is in [R_CRITICAL, R_DEGRADED) — transitions to Recovery
        let regime = rm.evaluate(&make_state(0.5), &empty_boundary());
        assert_eq!(regime, Regime::Recovery);
    }

    #[test]
    fn critical_to_nominal_when_r_high() {
        let mut rm = RegimeManager::default();
        rm.current = Regime::Critical;
        // R=0.8 exceeds R_DEGRADED + hysteresis — goes straight to Nominal
        let regime = rm.evaluate(&make_state(0.8), &empty_boundary());
        assert_eq!(regime, Regime::Nominal);
    }

    #[test]
    fn transition_cooldown() {
        let mut rm = RegimeManager::new(0.05, 5);
        rm.transition(Regime::Degraded);
        // Within cooldown, non-critical transitions blocked
        let result = rm.transition(Regime::Nominal);
        assert_eq!(result, Regime::Degraded);
    }

    #[test]
    fn critical_bypasses_cooldown() {
        let mut rm = RegimeManager::new(0.05, 100);
        rm.transition(Regime::Degraded);
        let result = rm.transition(Regime::Critical);
        assert_eq!(result, Regime::Critical);
    }

    #[test]
    fn same_regime_no_transition() {
        let mut rm = RegimeManager::default();
        rm.current = Regime::Nominal;
        let result = rm.transition(Regime::Nominal);
        assert_eq!(result, Regime::Nominal);
    }

    #[test]
    fn empty_layers_critical() {
        let rm = RegimeManager::default();
        let state = UPDEState {
            layers: vec![],
            cross_layer_alignment: vec![],
            stability_proxy: 0.0,
            regime: Regime::Nominal,
        };
        let regime = rm.evaluate(&state, &empty_boundary());
        assert_eq!(regime, Regime::Critical);
    }

    #[test]
    fn hysteresis_prevents_premature_upgrade() {
        // hysteresis=0.05, so Degraded→Nominal requires R >= 0.65 (not just 0.60)
        let mut rm = RegimeManager::new(0.05, 10);
        rm.current = Regime::Degraded;

        // R=0.62: within hysteresis band, stays Degraded
        let regime = rm.evaluate(&make_state(0.62), &empty_boundary());
        assert_eq!(regime, Regime::Degraded);

        // R=0.66: above band, promotes to Nominal
        let regime = rm.evaluate(&make_state(0.66), &empty_boundary());
        assert_eq!(regime, Regime::Nominal);
    }

    #[test]
    fn hysteresis_recovery_path() {
        let mut rm = RegimeManager::new(0.05, 10);
        rm.current = Regime::Critical;

        // R=0.5: above R_CRITICAL, in Recovery
        let regime = rm.evaluate(&make_state(0.5), &empty_boundary());
        assert_eq!(regime, Regime::Recovery);

        // R=0.62: in hysteresis band above R_DEGRADED, still Recovery
        rm.current = Regime::Recovery;
        let regime = rm.evaluate(&make_state(0.62), &empty_boundary());
        assert_eq!(regime, Regime::Recovery);

        // R=0.66: above band, promotes to Nominal
        let regime = rm.evaluate(&make_state(0.66), &empty_boundary());
        assert_eq!(regime, Regime::Nominal);
    }
}
