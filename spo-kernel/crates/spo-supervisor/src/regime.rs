// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Regime classifier

use std::collections::VecDeque;

use spo_types::{Regime, UPDEState};

use crate::boundaries::BoundaryState;
use crate::events::{EventBus, EventKind, RegimeEvent};

const R_CRITICAL: f64 = 0.3;
const R_DEGRADED: f64 = 0.6;
const MAX_LOG_LEN: usize = 100;

fn regime_rank(r: Regime) -> u8 {
    match r {
        Regime::Nominal => 0,
        Regime::Degraded => 1,
        Regime::Recovery => 2,
        Regime::Critical => 3,
    }
}

/// Regime finite-state machine with hysteresis, cooldown, downward-streak
/// hold, and optional EventBus notification.
pub struct RegimeManager {
    pub current: Regime,
    hysteresis: f64,
    cooldown_steps: u64,
    hysteresis_hold_steps: u64,
    step_counter: u64,
    last_transition: u64,
    downward_streak: u64,
    pub transition_log: VecDeque<(u64, Regime, Regime)>,
    event_bus: Option<EventBus>,
}

impl RegimeManager {
    #[must_use]
    pub fn new(hysteresis: f64, cooldown_steps: u64) -> Self {
        Self {
            current: Regime::Nominal,
            hysteresis,
            cooldown_steps,
            hysteresis_hold_steps: 0,
            step_counter: 0,
            last_transition: 0,
            downward_streak: 0,
            transition_log: VecDeque::new(),
            event_bus: None,
        }
    }

    #[must_use]
    pub fn with_hold_steps(mut self, hold: u64) -> Self {
        self.hysteresis_hold_steps = hold;
        self
    }

    pub fn set_event_bus(&mut self, bus: EventBus) {
        self.event_bus = Some(bus);
    }

    #[must_use]
    pub fn event_bus(&self) -> Option<&EventBus> {
        self.event_bus.as_ref()
    }

    pub fn event_bus_mut(&mut self) -> Option<&mut EventBus> {
        self.event_bus.as_mut()
    }

    #[must_use]
    pub fn evaluate(&self, upde_state: &UPDEState, boundary: &BoundaryState) -> Regime {
        if !boundary.hard_violations.is_empty() {
            return Regime::Critical;
        }
        let avg_r = upde_state.mean_r();

        if avg_r < R_CRITICAL {
            return Regime::Critical;
        }

        let is_recovering = matches!(self.current, Regime::Critical | Regime::Recovery);

        if avg_r < R_DEGRADED {
            if is_recovering {
                return Regime::Recovery;
            }
            return Regime::Degraded;
        }

        if self.current == Regime::Degraded && avg_r < R_DEGRADED + self.hysteresis {
            return Regime::Degraded;
        }
        if is_recovering && avg_r < R_DEGRADED + self.hysteresis {
            return Regime::Recovery;
        }

        if self.current == Regime::Critical {
            return Regime::Recovery;
        }

        Regime::Nominal
    }

    pub fn transition(&mut self, proposed: Regime) -> Regime {
        self.step_counter += 1;

        if proposed == self.current {
            self.downward_streak = 0;
            return self.current;
        }

        let is_downward = regime_rank(proposed) > regime_rank(self.current);
        if is_downward && proposed != Regime::Critical && self.hysteresis_hold_steps > 0 {
            self.downward_streak += 1;
            if self.downward_streak < self.hysteresis_hold_steps {
                return self.current;
            }
        } else {
            self.downward_streak = 0;
        }

        let in_cooldown = self.last_transition > 0
            && self.step_counter.saturating_sub(self.last_transition) < self.cooldown_steps;
        if in_cooldown && proposed != Regime::Critical {
            return self.current;
        }

        self.commit_transition(proposed)
    }

    /// Bypass cooldown and hysteresis — used by event-driven triggers.
    pub fn force_transition(&mut self, regime: Regime) -> Regime {
        self.step_counter += 1;
        if regime == self.current {
            return self.current;
        }
        self.commit_transition(regime)
    }

    fn commit_transition(&mut self, new: Regime) -> Regime {
        let prev = self.current;
        self.last_transition = self.step_counter;
        self.current = new;
        self.downward_streak = 0;
        if self.transition_log.len() == MAX_LOG_LEN {
            self.transition_log.pop_front();
        }
        self.transition_log
            .push_back((self.step_counter, prev, new));

        if let Some(bus) = &mut self.event_bus {
            bus.post(RegimeEvent::new(
                EventKind::RegimeTransition,
                self.step_counter,
                format!("{prev:?}->{new:?}"),
            ));
        }
        new
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
        let regime = rm.evaluate(&make_state(0.5), &empty_boundary());
        assert_eq!(regime, Regime::Recovery);
    }

    #[test]
    fn critical_goes_through_recovery() {
        // Python parity: Critical never jumps directly to Nominal.
        let mut rm = RegimeManager::default();
        rm.current = Regime::Critical;
        // R=0.8 above all thresholds — still returns Recovery (must step through)
        let regime = rm.evaluate(&make_state(0.8), &empty_boundary());
        assert_eq!(regime, Regime::Recovery);
        // Once in Recovery with R above hysteresis band → Nominal
        rm.current = Regime::Recovery;
        let regime = rm.evaluate(&make_state(0.8), &empty_boundary());
        assert_eq!(regime, Regime::Nominal);
    }

    #[test]
    fn transition_cooldown() {
        let mut rm = RegimeManager::new(0.05, 5);
        rm.transition(Regime::Degraded);
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
        let mut rm = RegimeManager::new(0.05, 10);
        rm.current = Regime::Degraded;
        let regime = rm.evaluate(&make_state(0.62), &empty_boundary());
        assert_eq!(regime, Regime::Degraded);
        let regime = rm.evaluate(&make_state(0.66), &empty_boundary());
        assert_eq!(regime, Regime::Nominal);
    }

    #[test]
    fn hysteresis_recovery_path() {
        let mut rm = RegimeManager::new(0.05, 10);
        rm.current = Regime::Critical;
        let regime = rm.evaluate(&make_state(0.5), &empty_boundary());
        assert_eq!(regime, Regime::Recovery);
        rm.current = Regime::Recovery;
        let regime = rm.evaluate(&make_state(0.62), &empty_boundary());
        assert_eq!(regime, Regime::Recovery);
        let regime = rm.evaluate(&make_state(0.66), &empty_boundary());
        assert_eq!(regime, Regime::Nominal);
    }

    #[test]
    fn downward_streak_blocks_premature_degradation() {
        let mut rm = RegimeManager::new(0.05, 0).with_hold_steps(3);
        // First two downward proposals blocked by streak
        assert_eq!(rm.transition(Regime::Degraded), Regime::Nominal);
        assert_eq!(rm.transition(Regime::Degraded), Regime::Nominal);
        // Third consecutive → accepted
        assert_eq!(rm.transition(Regime::Degraded), Regime::Degraded);
    }

    #[test]
    fn downward_streak_resets_on_same_regime() {
        let mut rm = RegimeManager::new(0.05, 0).with_hold_steps(3);
        rm.transition(Regime::Degraded); // streak=1, blocked
        rm.transition(Regime::Nominal); // same as current → streak reset
        rm.transition(Regime::Degraded); // streak=1 again, blocked
        assert_eq!(rm.current, Regime::Nominal);
    }

    #[test]
    fn critical_bypasses_streak() {
        let mut rm = RegimeManager::new(0.05, 0).with_hold_steps(100);
        assert_eq!(rm.transition(Regime::Critical), Regime::Critical);
    }

    #[test]
    fn transition_log_bounded() {
        let mut rm = RegimeManager::new(0.05, 0);
        for i in 0..150 {
            let regime = if i % 2 == 0 {
                Regime::Degraded
            } else {
                Regime::Nominal
            };
            rm.transition(regime);
        }
        assert!(rm.transition_log.len() <= MAX_LOG_LEN);
    }

    #[test]
    fn event_bus_receives_transitions() {
        let mut rm = RegimeManager::new(0.05, 0);
        rm.set_event_bus(EventBus::new(50));
        rm.transition(Regime::Degraded);
        rm.transition(Regime::Critical);

        let bus = rm.event_bus().expect("bus set");
        assert_eq!(bus.count(), 2);
        let first = &bus.history()[0];
        assert_eq!(first.kind, EventKind::RegimeTransition);
        assert!(first.detail.contains("Nominal"));
        assert!(first.detail.contains("Degraded"));
    }

    #[test]
    fn force_transition_with_event_bus() {
        let mut rm = RegimeManager::new(0.05, 0);
        rm.set_event_bus(EventBus::new(10));
        rm.force_transition(Regime::Critical);
        let bus = rm.event_bus().unwrap();
        assert_eq!(bus.count(), 1);
    }
}
