// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Rule-based policy engine (Rust port)

//! Declarative policy rule evaluation with cooldown and fire limits.
//!
//! Parity target: `scpn_phase_orchestrator.supervisor.policy_rules.PolicyEngine`.

use std::collections::HashMap;

use crate::petri_net::GuardOp;
use spo_types::UPDEState;

/// Single condition: metric op threshold.
#[derive(Debug, Clone)]
pub struct Condition {
    pub metric: String,
    pub op: GuardOp,
    pub threshold: f64,
}

impl Condition {
    fn evaluate(&self, ctx: &HashMap<String, f64>) -> bool {
        let Some(&val) = ctx.get(&self.metric) else {
            return false;
        };
        if !val.is_finite() {
            return false;
        }
        match self.op {
            GuardOp::Gt => val > self.threshold,
            GuardOp::Ge => val >= self.threshold,
            GuardOp::Lt => val < self.threshold,
            GuardOp::Le => val <= self.threshold,
            GuardOp::Eq => (val - self.threshold).abs() < 1e-12,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Logic {
    And,
    Or,
}

/// AND/OR combinator over multiple conditions.
#[derive(Debug, Clone)]
pub enum RuleCondition {
    Single(Condition),
    Compound {
        conditions: Vec<Condition>,
        logic: Logic,
    },
}

impl RuleCondition {
    fn evaluate(&self, ctx: &HashMap<String, f64>) -> bool {
        match self {
            Self::Single(c) => c.evaluate(ctx),
            Self::Compound { conditions, logic } => match logic {
                Logic::And => conditions.iter().all(|c| c.evaluate(ctx)),
                Logic::Or => conditions.iter().any(|c| c.evaluate(ctx)),
            },
        }
    }
}

/// Output action from a fired rule.
#[derive(Debug, Clone)]
pub struct RuleAction {
    pub knob: String,
    pub scope: String,
    pub value: f64,
    pub ttl_s: f64,
}

/// Named policy rule with regime filter, condition, actions, cooldown, max_fires.
#[derive(Debug, Clone)]
pub struct PolicyRule {
    pub name: String,
    pub regimes: Vec<String>,
    pub condition: RuleCondition,
    pub actions: Vec<RuleAction>,
    pub cooldown_s: f64,
    pub max_fires: u32, // 0 = unlimited
}

/// Fired action with provenance.
#[derive(Debug, Clone)]
pub struct FiredAction {
    pub knob: String,
    pub scope: String,
    pub value: f64,
    pub ttl_s: f64,
    pub rule_name: String,
}

/// Rule-based policy engine with cooldown and fire-count tracking.
#[derive(Debug)]
pub struct RuleEngine {
    rules: Vec<PolicyRule>,
    fire_counts: HashMap<String, u32>,
    last_fire_t: HashMap<String, f64>,
    clock: f64,
}

impl RuleEngine {
    /// Build a rule engine, panicking if rules are invalid.
    ///
    /// # Panics
    /// Panics when `rules` contains non-finite thresholds/action values,
    /// negative TTL/cooldown values, empty action fields, or empty condition
    /// metrics. Use [`Self::try_new`] when policy input is user supplied.
    #[must_use]
    pub fn new(rules: Vec<PolicyRule>) -> Self {
        Self::try_new(rules).expect("valid policy rules")
    }

    /// Build a rule engine after validating the policy surface.
    ///
    /// # Errors
    /// Returns a human-readable error when rule thresholds, action values,
    /// TTLs, or cooldowns are non-finite or outside their valid ranges.
    pub fn try_new(rules: Vec<PolicyRule>) -> Result<Self, String> {
        for rule in &rules {
            validate_rule(rule)?;
        }
        Ok(Self {
            rules,
            fire_counts: HashMap::new(),
            last_fire_t: HashMap::new(),
            clock: 0.0,
        })
    }

    pub fn advance_clock(&mut self, dt: f64) {
        self.clock += dt;
    }

    #[must_use]
    pub fn clock(&self) -> f64 {
        self.clock
    }

    /// Return policy rule names in evaluation order.
    #[must_use]
    pub fn rule_names(&self) -> Vec<String> {
        self.rules.iter().map(|rule| rule.name.clone()).collect()
    }

    /// Return the number of times a rule has fired.
    #[must_use]
    pub fn fire_count(&self, rule_name: &str) -> u32 {
        self.fire_counts.get(rule_name).copied().unwrap_or(0)
    }

    /// Return all recorded fire counts keyed by rule name.
    #[must_use]
    pub fn fire_counts(&self) -> &HashMap<String, u32> {
        &self.fire_counts
    }

    /// Return the last engine-clock time at which a rule fired.
    #[must_use]
    pub fn last_fire_time(&self, rule_name: &str) -> Option<f64> {
        self.last_fire_t.get(rule_name).copied()
    }

    /// Return all recorded last-fire times keyed by rule name.
    #[must_use]
    pub fn last_fire_times(&self) -> &HashMap<String, f64> {
        &self.last_fire_t
    }

    /// Evaluate all rules against regime + metric context. Returns fired actions.
    pub fn evaluate(&mut self, regime: &str, ctx: &HashMap<String, f64>) -> Vec<FiredAction> {
        let regime_upper = regime.to_uppercase();
        let mut actions = Vec::new();

        for rule in &self.rules {
            if !rule.regimes.contains(&regime_upper) {
                continue;
            }
            if !rule.condition.evaluate(ctx) {
                continue;
            }
            if rule.cooldown_s > 0.0 {
                let last = self
                    .last_fire_t
                    .get(&rule.name)
                    .copied()
                    .unwrap_or(-rule.cooldown_s - 1.0);
                if self.clock - last < rule.cooldown_s {
                    continue;
                }
            }
            let fires = self.fire_counts.get(&rule.name).copied().unwrap_or(0);
            if rule.max_fires > 0 && fires >= rule.max_fires {
                continue;
            }
            *self.fire_counts.entry(rule.name.clone()).or_insert(0) += 1;
            self.last_fire_t.insert(rule.name.clone(), self.clock);
            for a in &rule.actions {
                actions.push(FiredAction {
                    knob: a.knob.clone(),
                    scope: a.scope.clone(),
                    value: a.value,
                    ttl_s: a.ttl_s,
                    rule_name: rule.name.clone(),
                });
            }
        }
        actions
    }

    /// Evaluate against a structured UPDE state plus good/bad layer partitions.
    pub fn evaluate_state(
        &mut self,
        regime: &str,
        state: &UPDEState,
        good_layers: &[usize],
        bad_layers: &[usize],
        extra: &HashMap<String, f64>,
    ) -> Vec<FiredAction> {
        let ctx = state_context(state, good_layers, bad_layers, extra);
        self.evaluate(regime, &ctx)
    }

    pub fn reset(&mut self) {
        self.fire_counts.clear();
        self.last_fire_t.clear();
        self.clock = 0.0;
    }
}

fn validate_rule(rule: &PolicyRule) -> Result<(), String> {
    if rule.name.is_empty() {
        return Err("policy rule name must not be empty".into());
    }
    validate_condition(&rule.condition)?;
    if !rule.cooldown_s.is_finite() || rule.cooldown_s < 0.0 {
        return Err(format!(
            "policy rule {:?}: cooldown_s must be finite and >= 0",
            rule.name
        ));
    }
    if rule.actions.is_empty() {
        return Err(format!(
            "policy rule {:?}: actions must not be empty",
            rule.name
        ));
    }
    for action in &rule.actions {
        if action.knob.is_empty() {
            return Err(format!(
                "policy rule {:?}: action knob must not be empty",
                rule.name
            ));
        }
        if action.scope.is_empty() {
            return Err(format!(
                "policy rule {:?}: action scope must not be empty",
                rule.name
            ));
        }
        if !action.value.is_finite() {
            return Err(format!(
                "policy rule {:?}: action value must be finite",
                rule.name
            ));
        }
        if !action.ttl_s.is_finite() || action.ttl_s < 0.0 {
            return Err(format!(
                "policy rule {:?}: action ttl_s must be finite and >= 0",
                rule.name
            ));
        }
    }
    Ok(())
}

fn validate_condition(condition: &RuleCondition) -> Result<(), String> {
    match condition {
        RuleCondition::Single(condition) => validate_single_condition(condition),
        RuleCondition::Compound { conditions, .. } => {
            if conditions.is_empty() {
                return Err("compound policy condition must not be empty".into());
            }
            for condition in conditions {
                validate_single_condition(condition)?;
            }
            Ok(())
        }
    }
}

fn validate_single_condition(condition: &Condition) -> Result<(), String> {
    if condition.metric.is_empty() {
        return Err("policy condition metric must not be empty".into());
    }
    if !condition.threshold.is_finite() {
        return Err(format!(
            "policy condition {:?}: threshold must be finite",
            condition.metric
        ));
    }
    Ok(())
}

/// Build the metric context used by rule evaluation from a UPDE state.
#[must_use]
pub fn state_context(
    state: &UPDEState,
    good_layers: &[usize],
    bad_layers: &[usize],
    extra: &HashMap<String, f64>,
) -> HashMap<String, f64> {
    let mut ctx = extra.clone();
    ctx.insert("stability_proxy".into(), state.stability_proxy);
    ctx.insert("R".into(), state.mean_r());

    for (idx, layer) in state.layers.iter().enumerate() {
        ctx.insert(format!("R.{idx}"), layer.r);
        ctx.insert(format!("R_{idx}"), layer.r);
        ctx.insert(format!("psi.{idx}"), layer.psi);
        ctx.insert(format!("psi_{idx}"), layer.psi);
    }

    for (idx, &layer_idx) in good_layers.iter().enumerate() {
        if let Some(layer) = state.layers.get(layer_idx) {
            ctx.insert(format!("R_good.{idx}"), layer.r);
            ctx.insert(format!("R_good_{idx}"), layer.r);
        }
    }
    for (idx, &layer_idx) in bad_layers.iter().enumerate() {
        if let Some(layer) = state.layers.get(layer_idx) {
            ctx.insert(format!("R_bad.{idx}"), layer.r);
            ctx.insert(format!("R_bad_{idx}"), layer.r);
        }
    }

    for metric in &state.channel_metrics {
        let channel = metric.channel.as_str();
        ctx.insert(format!("R.channel.{channel}"), metric.r);
        ctx.insert(format!("R_channel_{channel}"), metric.r);
        ctx.insert(format!("psi.channel.{channel}"), metric.psi);
        ctx.insert(format!("psi_channel_{channel}"), metric.psi);
        ctx.insert(format!("channel_weight.{channel}"), metric.weight);
        ctx.insert(format!("channel_weight_{channel}"), metric.weight);
    }

    ctx
}

#[cfg(test)]
mod tests {
    use super::*;
    use spo_types::{Channel, ChannelMetric, LayerState, Regime, UPDEState};

    fn simple_rule(name: &str, regime: &str, metric: &str, op: GuardOp, thresh: f64) -> PolicyRule {
        PolicyRule {
            name: name.into(),
            regimes: vec![regime.to_uppercase()],
            condition: RuleCondition::Single(Condition {
                metric: metric.into(),
                op,
                threshold: thresh,
            }),
            actions: vec![RuleAction {
                knob: "K".into(),
                scope: "global".into(),
                value: 0.1,
                ttl_s: 10.0,
            }],
            cooldown_s: 0.0,
            max_fires: 0,
        }
    }

    fn state(rs: &[f64]) -> UPDEState {
        UPDEState {
            layers: rs
                .iter()
                .copied()
                .map(|r| LayerState { r, psi: 0.0 })
                .collect(),
            channel_metrics: vec![],
            cross_layer_alignment: vec![],
            stability_proxy: 0.42,
            regime: Regime::Nominal,
        }
    }

    fn state_with_channels(rs: &[f64], channel_metrics: Vec<ChannelMetric>) -> UPDEState {
        UPDEState {
            layers: rs
                .iter()
                .copied()
                .map(|r| LayerState { r, psi: 0.0 })
                .collect(),
            channel_metrics,
            cross_layer_alignment: vec![],
            stability_proxy: 0.42,
            regime: Regime::Nominal,
        }
    }

    #[test]
    fn fires_when_condition_met() {
        let mut eng = RuleEngine::new(vec![simple_rule(
            "boost",
            "DEGRADED",
            "stability_proxy",
            GuardOp::Lt,
            0.5,
        )]);
        let ctx: HashMap<String, f64> = [("stability_proxy".into(), 0.3)].into();
        let actions = eng.evaluate("degraded", &ctx);
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].rule_name, "boost");
    }

    #[test]
    fn try_new_rejects_non_finite_threshold() {
        let err = RuleEngine::try_new(vec![simple_rule(
            "bad",
            "DEGRADED",
            "stability_proxy",
            GuardOp::Lt,
            f64::NAN,
        )])
        .expect_err("NaN threshold must be rejected");

        assert!(err.contains("threshold must be finite"));
    }

    #[test]
    fn try_new_rejects_bad_action_values() {
        let mut rule = simple_rule("bad_action", "DEGRADED", "x", GuardOp::Lt, 1.0);
        rule.actions[0].ttl_s = -1.0;

        let err = RuleEngine::try_new(vec![rule]).expect_err("negative TTL must be rejected");

        assert!(err.contains("ttl_s must be finite and >= 0"));
    }

    #[test]
    fn try_new_rejects_empty_actions() {
        let mut rule = simple_rule("no_action", "DEGRADED", "x", GuardOp::Lt, 1.0);
        rule.actions.clear();

        let err = RuleEngine::try_new(vec![rule]).expect_err("empty actions must fail");

        assert!(err.contains("actions must not be empty"));
    }

    #[test]
    fn non_finite_context_metric_does_not_fire() {
        let mut eng = RuleEngine::new(vec![simple_rule(
            "boost",
            "DEGRADED",
            "stability_proxy",
            GuardOp::Gt,
            0.5,
        )]);
        let ctx: HashMap<String, f64> = [("stability_proxy".into(), f64::INFINITY)].into();

        assert!(eng.evaluate("degraded", &ctx).is_empty());
    }

    #[test]
    fn state_context_exposes_layer_and_partition_metrics() {
        let ctx = state_context(&state(&[0.2, 0.8, 0.4]), &[1], &[0, 2], &HashMap::new());

        assert_eq!(ctx["stability_proxy"], 0.42);
        assert!((ctx["R"] - (0.2 + 0.8 + 0.4) / 3.0).abs() < 1e-12);
        assert_eq!(ctx["R.1"], 0.8);
        assert_eq!(ctx["R_good.0"], 0.8);
        assert_eq!(ctx["R_bad.1"], 0.4);
    }

    #[test]
    fn state_context_exposes_named_channel_metrics() {
        let state = state_with_channels(
            &[0.2, 0.8, 0.4],
            vec![
                ChannelMetric::new(Channel::P, 0.75, 1.0, 1.0).expect("valid P metric"),
                ChannelMetric::new(Channel::new("Risk"), 0.25, 2.0, 0.5)
                    .expect("valid Risk metric"),
            ],
        );

        let ctx = state_context(&state, &[], &[], &HashMap::new());

        assert_eq!(ctx["R.channel.P"], 0.75);
        assert_eq!(ctx["R_channel_Risk"], 0.25);
        assert_eq!(ctx["psi.channel.Risk"], 2.0);
        assert_eq!(ctx["channel_weight_Risk"], 0.5);
    }

    #[test]
    fn evaluate_state_uses_good_bad_layer_metrics() {
        let rule = simple_rule("suppress_bad", "CRITICAL", "R_bad.0", GuardOp::Lt, 0.3);
        let mut eng = RuleEngine::new(vec![rule]);
        let actions =
            eng.evaluate_state("critical", &state(&[0.2, 0.9]), &[1], &[0], &HashMap::new());

        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].rule_name, "suppress_bad");
    }

    #[test]
    fn evaluate_state_uses_named_channel_metrics() {
        let rule = simple_rule("risk_guard", "CRITICAL", "R.channel.Risk", GuardOp::Lt, 0.3);
        let mut eng = RuleEngine::new(vec![rule]);
        let state = state_with_channels(
            &[0.2, 0.9],
            vec![ChannelMetric::new(Channel::new("Risk"), 0.25, 2.0, 0.5)
                .expect("valid Risk metric")],
        );

        let actions = eng.evaluate_state("critical", &state, &[], &[], &HashMap::new());

        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].rule_name, "risk_guard");
    }

    #[test]
    fn skips_wrong_regime() {
        let mut eng = RuleEngine::new(vec![simple_rule(
            "boost",
            "DEGRADED",
            "stability_proxy",
            GuardOp::Lt,
            0.5,
        )]);
        let ctx: HashMap<String, f64> = [("stability_proxy".into(), 0.3)].into();
        let actions = eng.evaluate("nominal", &ctx);
        assert!(actions.is_empty());
    }

    #[test]
    fn skips_unmet_condition() {
        let mut eng = RuleEngine::new(vec![simple_rule(
            "boost",
            "DEGRADED",
            "stability_proxy",
            GuardOp::Lt,
            0.5,
        )]);
        let ctx: HashMap<String, f64> = [("stability_proxy".into(), 0.8)].into();
        let actions = eng.evaluate("degraded", &ctx);
        assert!(actions.is_empty());
    }

    #[test]
    fn cooldown_blocks_refire() {
        let rule = PolicyRule {
            cooldown_s: 5.0,
            ..simple_rule("r1", "DEGRADED", "x", GuardOp::Lt, 1.0)
        };
        let mut eng = RuleEngine::new(vec![rule]);
        let ctx: HashMap<String, f64> = [("x".into(), 0.5)].into();

        assert_eq!(eng.evaluate("degraded", &ctx).len(), 1);
        assert!(eng.evaluate("degraded", &ctx).is_empty()); // cooldown
        eng.advance_clock(6.0);
        assert_eq!(eng.evaluate("degraded", &ctx).len(), 1); // cooldown expired
    }

    #[test]
    fn max_fires_limit() {
        let rule = PolicyRule {
            max_fires: 2,
            ..simple_rule("r1", "DEGRADED", "x", GuardOp::Lt, 1.0)
        };
        let mut eng = RuleEngine::new(vec![rule]);
        let ctx: HashMap<String, f64> = [("x".into(), 0.5)].into();

        assert_eq!(eng.evaluate("degraded", &ctx).len(), 1);
        assert_eq!(eng.evaluate("degraded", &ctx).len(), 1);
        assert!(eng.evaluate("degraded", &ctx).is_empty()); // max_fires reached
    }

    #[test]
    fn exposes_rule_fire_diagnostics() {
        let rule = PolicyRule {
            max_fires: 2,
            cooldown_s: 0.5,
            ..simple_rule("r1", "DEGRADED", "x", GuardOp::Lt, 1.0)
        };
        let mut eng = RuleEngine::new(vec![rule]);
        let ctx: HashMap<String, f64> = [("x".into(), 0.5)].into();

        assert_eq!(eng.rule_names(), vec!["r1".to_string()]);
        assert_eq!(eng.fire_count("r1"), 0);
        assert_eq!(eng.last_fire_time("r1"), None);

        assert_eq!(eng.evaluate("degraded", &ctx).len(), 1);

        assert_eq!(eng.fire_count("r1"), 1);
        assert_eq!(eng.fire_counts()["r1"], 1);
        assert_eq!(eng.last_fire_time("r1"), Some(0.0));
        assert_eq!(eng.last_fire_times()["r1"], 0.0);

        eng.advance_clock(1.0);
        assert_eq!(eng.evaluate("degraded", &ctx).len(), 1);
        assert_eq!(eng.fire_count("r1"), 2);
        assert_eq!(eng.last_fire_time("r1"), Some(1.0));
    }

    #[test]
    fn compound_and() {
        let rule = PolicyRule {
            name: "compound_and".into(),
            regimes: vec!["DEGRADED".into()],
            condition: RuleCondition::Compound {
                conditions: vec![
                    Condition {
                        metric: "a".into(),
                        op: GuardOp::Gt,
                        threshold: 0.5,
                    },
                    Condition {
                        metric: "b".into(),
                        op: GuardOp::Lt,
                        threshold: 0.3,
                    },
                ],
                logic: Logic::And,
            },
            actions: vec![RuleAction {
                knob: "K".into(),
                scope: "global".into(),
                value: 0.1,
                ttl_s: 5.0,
            }],
            cooldown_s: 0.0,
            max_fires: 0,
        };
        let mut eng = RuleEngine::new(vec![rule]);

        // Both met
        let ctx: HashMap<String, f64> = [("a".into(), 0.8), ("b".into(), 0.1)].into();
        assert_eq!(eng.evaluate("degraded", &ctx).len(), 1);

        // Only a met
        let ctx2: HashMap<String, f64> = [("a".into(), 0.8), ("b".into(), 0.5)].into();
        assert!(eng.evaluate("degraded", &ctx2).is_empty());
    }

    #[test]
    fn compound_or() {
        let rule = PolicyRule {
            name: "compound_or".into(),
            regimes: vec!["DEGRADED".into()],
            condition: RuleCondition::Compound {
                conditions: vec![
                    Condition {
                        metric: "a".into(),
                        op: GuardOp::Gt,
                        threshold: 0.9,
                    },
                    Condition {
                        metric: "b".into(),
                        op: GuardOp::Lt,
                        threshold: 0.1,
                    },
                ],
                logic: Logic::Or,
            },
            actions: vec![RuleAction {
                knob: "K".into(),
                scope: "global".into(),
                value: 0.1,
                ttl_s: 5.0,
            }],
            cooldown_s: 0.0,
            max_fires: 0,
        };
        let mut eng = RuleEngine::new(vec![rule]);

        // Only b met
        let ctx: HashMap<String, f64> = [("a".into(), 0.5), ("b".into(), 0.05)].into();
        assert_eq!(eng.evaluate("degraded", &ctx).len(), 1);

        // Neither met
        let ctx2: HashMap<String, f64> = [("a".into(), 0.5), ("b".into(), 0.5)].into();
        assert!(eng.evaluate("degraded", &ctx2).is_empty());
    }

    #[test]
    fn reset_clears_state() {
        let rule = PolicyRule {
            max_fires: 1,
            ..simple_rule("r1", "DEGRADED", "x", GuardOp::Lt, 1.0)
        };
        let mut eng = RuleEngine::new(vec![rule]);
        let ctx: HashMap<String, f64> = [("x".into(), 0.5)].into();

        eng.evaluate("degraded", &ctx);
        assert!(eng.evaluate("degraded", &ctx).is_empty());
        eng.reset();
        assert_eq!(eng.evaluate("degraded", &ctx).len(), 1);
    }

    #[test]
    fn multiple_actions_per_rule() {
        let rule = PolicyRule {
            name: "multi".into(),
            regimes: vec!["CRITICAL".into()],
            condition: RuleCondition::Single(Condition {
                metric: "x".into(),
                op: GuardOp::Lt,
                threshold: 0.2,
            }),
            actions: vec![
                RuleAction {
                    knob: "K".into(),
                    scope: "global".into(),
                    value: 0.1,
                    ttl_s: 5.0,
                },
                RuleAction {
                    knob: "zeta".into(),
                    scope: "layer_0".into(),
                    value: 0.5,
                    ttl_s: 3.0,
                },
            ],
            cooldown_s: 0.0,
            max_fires: 0,
        };
        let mut eng = RuleEngine::new(vec![rule]);
        let ctx: HashMap<String, f64> = [("x".into(), 0.1)].into();
        let actions = eng.evaluate("critical", &ctx);
        assert_eq!(actions.len(), 2);
        assert_eq!(actions[0].knob, "K");
        assert_eq!(actions[1].knob, "zeta");
    }

    #[test]
    fn missing_metric_skips() {
        let mut eng = RuleEngine::new(vec![simple_rule(
            "r1",
            "DEGRADED",
            "nonexistent",
            GuardOp::Lt,
            1.0,
        )]);
        let actions = eng.evaluate("degraded", &HashMap::new());
        assert!(actions.is_empty());
    }
}
