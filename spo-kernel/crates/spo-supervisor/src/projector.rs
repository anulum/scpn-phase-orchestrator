// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — State projector

//!
//! Clips control actions to value bounds and rate limits.

use std::collections::HashMap;

use spo_types::{ControlAction, Knob};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdaptiveRateLimitConfig {
    pub min_limit_units: u32,
    pub nominal_limit_units: u32,
    pub max_limit_units: u32,
    pub risk_gain_units: u32,
    pub risk_full_scale_units: u32,
}

impl AdaptiveRateLimitConfig {
    #[must_use]
    pub const fn is_valid(self) -> bool {
        self.min_limit_units <= self.nominal_limit_units
            && self.nominal_limit_units <= self.max_limit_units
            && self.risk_full_scale_units > 0
    }
}

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

        let clamped = project_value(
            action.value,
            previous_value,
            lo,
            hi,
            self.rate_limits.get(&action.knob).copied(),
        );

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

#[must_use]
#[cfg_attr(kani, kani::requires(proposed.is_finite()))]
#[cfg_attr(kani, kani::requires(previous.is_finite()))]
#[cfg_attr(kani, kani::requires(lo.is_finite()))]
#[cfg_attr(kani, kani::requires(hi.is_finite()))]
#[cfg_attr(kani, kani::requires(lo <= hi))]
#[cfg_attr(
    kani,
    kani::requires(match rate_limit {
        Some(limit) => limit.is_finite() && limit >= 0.0,
        None => true,
    })
)]
#[cfg_attr(kani, kani::ensures(|result| result.is_finite()))]
#[cfg_attr(kani, kani::ensures(|result| *result >= lo && *result <= hi))]
pub fn project_value(
    proposed: f64,
    previous: f64,
    lo: f64,
    hi: f64,
    rate_limit: Option<f64>,
) -> f64 {
    let mut clamped = proposed.clamp(lo, hi);

    if let Some(rate_limit) = rate_limit {
        let delta = clamped - previous;
        if delta.abs() > rate_limit {
            clamped = previous + rate_limit * delta.signum();
            clamped = clamped.clamp(lo, hi);
        }
    }
    clamped
}

#[must_use]
#[cfg_attr(kani, kani::requires(config.is_valid()))]
#[cfg_attr(kani, kani::ensures(|result| *result >= config.min_limit_units))]
#[cfg_attr(kani, kani::ensures(|result| *result <= config.max_limit_units))]
pub fn compute_adaptive_rate_limit_fixed(
    risk_signal_units: u32,
    config: AdaptiveRateLimitConfig,
) -> u32 {
    if !config.is_valid() {
        return config.min_limit_units.min(config.max_limit_units);
    }

    let saturated_risk = risk_signal_units.min(config.risk_full_scale_units);
    let adaptive_increment = (u64::from(config.risk_gain_units) * u64::from(saturated_risk))
        / u64::from(config.risk_full_scale_units);
    let raw_limit = u64::from(config.nominal_limit_units) + adaptive_increment;
    let clamped = raw_limit
        .max(u64::from(config.min_limit_units))
        .min(u64::from(config.max_limit_units));

    clamped as u32
}

#[must_use]
#[cfg_attr(kani, kani::requires(lo <= previous && previous <= hi))]
#[cfg_attr(kani, kani::requires(lo <= hi))]
#[cfg_attr(kani, kani::ensures(|result| *result >= lo && *result <= hi))]
#[cfg_attr(
    kani,
    kani::ensures(|result| {
        let delta = if *result >= previous {
            i64::from(*result) - i64::from(previous)
        } else {
            i64::from(previous) - i64::from(*result)
        };
        delta <= i64::from(rate_limit)
    })
)]
pub fn project_fixed_point_value(
    proposed: i32,
    previous: i32,
    lo: i32,
    hi: i32,
    rate_limit: u32,
) -> i32 {
    let bounded = proposed.clamp(lo, hi);
    let delta = i64::from(bounded) - i64::from(previous);
    let limit = i64::from(rate_limit);
    let previous_wide = i64::from(previous);

    let limited = if delta > limit {
        previous_wide + limit
    } else if delta < -limit {
        previous_wide - limit
    } else {
        i64::from(bounded)
    };

    let lower = i64::from(lo);
    let upper = i64::from(hi);
    limited.clamp(lower, upper) as i32
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
    fn project_value_respects_bounds_without_hashmap() {
        let result = project_value(2.0, 0.5, 0.0, 1.0, None);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn project_value_respects_rate_limit_without_hashmap() {
        let result = project_value(5.0, 0.0, -10.0, 10.0, Some(0.1));
        assert!((result - 0.1).abs() < 1e-12);
    }

    #[test]
    fn adaptive_fixed_point_rate_limit_scales_with_risk_and_clamps() {
        let config = AdaptiveRateLimitConfig {
            min_limit_units: 2,
            nominal_limit_units: 4,
            max_limit_units: 12,
            risk_gain_units: 16,
            risk_full_scale_units: 100,
        };

        assert_eq!(compute_adaptive_rate_limit_fixed(0, config), 4);
        assert_eq!(compute_adaptive_rate_limit_fixed(50, config), 12);
        assert_eq!(compute_adaptive_rate_limit_fixed(200, config), 12);
    }

    #[test]
    fn fixed_point_projection_respects_adaptive_rate_limit() {
        let config = AdaptiveRateLimitConfig {
            min_limit_units: 1,
            nominal_limit_units: 3,
            max_limit_units: 9,
            risk_gain_units: 6,
            risk_full_scale_units: 100,
        };
        let rate_limit = compute_adaptive_rate_limit_fixed(100, config);

        let projected = project_fixed_point_value(100, 10, -50, 50, rate_limit);
        assert_eq!(projected, 19);
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
