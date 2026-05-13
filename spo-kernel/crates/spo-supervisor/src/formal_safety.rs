// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Kani formal safety harnesses

//! Kani harnesses for supervisor safety contracts.
//!
//! These harnesses intentionally call the same crate functions used at runtime.
//! They are not duplicate mini-models.

use spo_types::Regime;

use crate::projector::{
    compute_adaptive_rate_limit_fixed, project_fixed_point_value, project_value,
    AdaptiveRateLimitConfig,
};
use crate::regime::{classify_regime_from_summary, R_CRITICAL, R_DEGRADED};

#[kani::proof]
fn action_projector_value_clipping_contract() {
    let proposed: f64 = kani::any();
    let previous: f64 = kani::any();
    let lo: f64 = kani::any();
    let hi: f64 = kani::any();

    kani::assume(proposed.is_finite());
    kani::assume(previous.is_finite());
    kani::assume(lo.is_finite());
    kani::assume(hi.is_finite());
    kani::assume(lo <= hi);

    let projected = project_value(proposed, previous, lo, hi, None);
    assert!(projected >= lo, "projected value must respect lower bound");
    assert!(projected <= hi, "projected value must respect upper bound");
}

#[kani::proof]
fn adaptive_rate_limit_contract() {
    let min_limit: u16 = kani::any();
    let nominal_limit: u16 = kani::any();
    let max_limit: u16 = kani::any();
    let risk_gain: u16 = kani::any();
    let risk_full_scale: u16 = kani::any();
    let risk_signal: u16 = kani::any();

    kani::assume(min_limit <= nominal_limit);
    kani::assume(nominal_limit <= max_limit);
    kani::assume(risk_full_scale > 0);

    let config = AdaptiveRateLimitConfig {
        min_limit_units: u32::from(min_limit),
        nominal_limit_units: u32::from(nominal_limit),
        max_limit_units: u32::from(max_limit),
        risk_gain_units: u32::from(risk_gain),
        risk_full_scale_units: u32::from(risk_full_scale),
    };

    let limit = compute_adaptive_rate_limit_fixed(u32::from(risk_signal), config);
    assert!(
        limit >= config.min_limit_units,
        "adaptive limit must respect lower bound"
    );
    assert!(
        limit <= config.max_limit_units,
        "adaptive limit must respect upper bound"
    );
}

#[kani::proof]
fn action_projector_adaptive_fixed_point_rate_limit_contract() {
    let proposed_raw: i16 = kani::any();
    let previous_raw: i16 = kani::any();
    let lo_raw: i16 = kani::any();
    let hi_raw: i16 = kani::any();
    let risk_signal: u16 = kani::any();

    kani::assume(lo_raw <= hi_raw);
    kani::assume(previous_raw >= lo_raw);
    kani::assume(previous_raw <= hi_raw);

    let config = AdaptiveRateLimitConfig {
        min_limit_units: 1,
        nominal_limit_units: 4,
        max_limit_units: 32,
        risk_gain_units: 28,
        risk_full_scale_units: 100,
    };
    let rate_limit = compute_adaptive_rate_limit_fixed(u32::from(risk_signal), config);
    let projected = project_fixed_point_value(
        i32::from(proposed_raw),
        i32::from(previous_raw),
        i32::from(lo_raw),
        i32::from(hi_raw),
        rate_limit,
    );

    let delta = if projected >= i32::from(previous_raw) {
        i64::from(projected) - i64::from(previous_raw)
    } else {
        i64::from(previous_raw) - i64::from(projected)
    };
    assert!(
        projected >= i32::from(lo_raw),
        "projected value must respect lower bound"
    );
    assert!(
        projected <= i32::from(hi_raw),
        "projected value must respect upper bound"
    );
    assert!(
        delta <= i64::from(rate_limit),
        "projected delta must not exceed adaptive rate limit"
    );
}

#[kani::proof]
fn nominal_safe_summary_never_classifies_critical() {
    let mean_r: f64 = kani::any();
    let hysteresis: f64 = kani::any();

    kani::assume(mean_r.is_finite());
    kani::assume((R_CRITICAL..=1.0).contains(&mean_r));
    kani::assume(hysteresis.is_finite());
    kani::assume(hysteresis >= 0.0);

    let proposed = classify_regime_from_summary(Regime::Nominal, mean_r, 0, hysteresis);
    assert_ne!(
        proposed,
        Regime::Critical,
        "nominal safe envelope must not classify as critical"
    );
}

#[kani::proof]
fn critical_never_evaluates_directly_to_nominal() {
    let mean_r: f64 = kani::any();
    let hard_violation_count: usize = kani::any();
    let hysteresis: f64 = kani::any();

    kani::assume(mean_r.is_finite());
    kani::assume((0.0..=1.0).contains(&mean_r));
    kani::assume(hysteresis.is_finite());
    kani::assume(hysteresis >= 0.0);

    let proposed =
        classify_regime_from_summary(Regime::Critical, mean_r, hard_violation_count, hysteresis);
    assert_ne!(
        proposed,
        Regime::Nominal,
        "critical regime must pass through recovery before nominal"
    );
}

#[kani::proof]
fn degraded_band_from_nominal_is_not_critical() {
    let mean_r: f64 = kani::any();
    let hysteresis: f64 = kani::any();

    kani::assume(mean_r.is_finite());
    kani::assume((R_CRITICAL..R_DEGRADED).contains(&mean_r));
    kani::assume(hysteresis.is_finite());
    kani::assume(hysteresis >= 0.0);

    let proposed = classify_regime_from_summary(Regime::Nominal, mean_r, 0, hysteresis);
    assert_eq!(
        proposed,
        Regime::Degraded,
        "nominal degraded-band envelope must degrade, not jump to critical"
    );
}
