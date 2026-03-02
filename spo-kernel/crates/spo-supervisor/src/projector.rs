// SCPN Phase Orchestrator — Action Projector
//!
//! Clips control actions to value bounds and rate limits.

use std::collections::HashMap;

use spo_types::{ControlAction, Knob};

pub struct ActionProjector {
    rate_limits: HashMap<Knob, f64>,
    value_bounds: HashMap<Knob, (f64, f64)>,
}

impl ActionProjector {
    #[must_use]
    pub fn new(rate_limits: HashMap<Knob, f64>, value_bounds: HashMap<Knob, (f64, f64)>) -> Self {
        Self {
            rate_limits,
            value_bounds,
        }
    }

    #[must_use]
    pub fn project(&self, action: &ControlAction, previous_value: f64) -> ControlAction {
        let (lo, hi) = self
            .value_bounds
            .get(&action.knob)
            .copied()
            .unwrap_or((f64::NEG_INFINITY, f64::INFINITY));

        let mut clamped = action.value.clamp(lo, hi);

        if let Some(&rate_limit) = self.rate_limits.get(&action.knob) {
            let delta = clamped - previous_value;
            if delta.abs() > rate_limit {
                clamped = previous_value + rate_limit * delta.signum();
                clamped = clamped.clamp(lo, hi);
            }
        }

        ControlAction {
            knob: action.knob,
            scope: action.scope.clone(),
            value: clamped,
            ttl_s: action.ttl_s,
            justification: action.justification.clone(),
        }
    }

    #[must_use]
    pub fn project_batch(
        &self,
        actions: &[ControlAction],
        previous_values: &HashMap<Knob, f64>,
    ) -> Vec<ControlAction> {
        actions
            .iter()
            .map(|a| {
                let prev = previous_values.get(&a.knob).copied().unwrap_or(0.0);
                self.project(a, prev)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_action(knob: Knob, value: f64) -> ControlAction {
        ControlAction {
            knob,
            scope: "global".into(),
            value,
            ttl_s: 10.0,
            justification: "test".into(),
        }
    }

    #[test]
    fn value_clamped_to_bounds() {
        let mut bounds = HashMap::new();
        bounds.insert(Knob::K, (0.0, 1.0));
        let proj = ActionProjector::new(HashMap::new(), bounds);
        let result = proj.project(&make_action(Knob::K, 2.0), 0.5);
        assert_eq!(result.value, 1.0);
    }

    #[test]
    fn rate_limited() {
        let mut rate_limits = HashMap::new();
        rate_limits.insert(Knob::K, 0.1);
        let proj = ActionProjector::new(rate_limits, HashMap::new());
        let result = proj.project(&make_action(Knob::K, 5.0), 0.0);
        assert!((result.value - 0.1).abs() < 1e-12);
    }

    #[test]
    fn negative_rate_limit() {
        let mut rate_limits = HashMap::new();
        rate_limits.insert(Knob::Zeta, 0.05);
        let proj = ActionProjector::new(rate_limits, HashMap::new());
        let result = proj.project(&make_action(Knob::Zeta, -1.0), 0.0);
        assert!((result.value - (-0.05)).abs() < 1e-12);
    }

    #[test]
    fn no_bounds_passthrough() {
        let proj = ActionProjector::new(HashMap::new(), HashMap::new());
        let result = proj.project(&make_action(Knob::Alpha, 42.0), 0.0);
        assert_eq!(result.value, 42.0);
    }

    #[test]
    fn batch_projection() {
        let mut bounds = HashMap::new();
        bounds.insert(Knob::K, (0.0, 1.0));
        let proj = ActionProjector::new(HashMap::new(), bounds);
        let actions = vec![make_action(Knob::K, 2.0), make_action(Knob::K, -1.0)];
        let prev = HashMap::new();
        let results = proj.project_batch(&actions, &prev);
        assert_eq!(results[0].value, 1.0);
        assert_eq!(results[1].value, 0.0);
    }
}
