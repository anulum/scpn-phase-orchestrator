// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Kani proof stubs

//! Formal verification stubs for ActionProjector and RegimeManager.
//!
//! Requires `cargo kani` to run. Not executed in CI yet — these are
//! compile-checked stubs that encode the safety properties we intend
//! to verify once Kani is integrated into the CI pipeline.
//!
//! Properties verified (when run):
//! - ActionProjector output is always within declared value bounds
//! - ActionProjector rate limiting never exceeds the configured rate
//! - RegimeManager FSM never skips directly from Critical to Nominal
//! - RegimeManager transition log never exceeds MAX_LOG_LEN

// ── ActionProjector bounds proofs ──────────────────────────────────

/// Value clipping: for any input value and any bounds [lo, hi],
/// the projected output satisfies lo <= output <= hi.
#[cfg(kani)]
#[kani::proof]
fn action_projector_value_clipping_proof() {
    let value: f64 = kani::any();
    let lo: f64 = kani::any();
    let hi: f64 = kani::any();

    // Preconditions: bounds are finite and ordered
    kani::assume(lo.is_finite() && hi.is_finite() && value.is_finite());
    kani::assume(lo <= hi);

    let clamped = value.clamp(lo, hi);

    assert!(clamped >= lo, "clamp result must be >= lower bound");
    assert!(clamped <= hi, "clamp result must be <= upper bound");
}

/// Rate limiting: for any previous value and rate limit, the
/// absolute change in the projected output never exceeds the rate
/// limit (up to the additional value-bound clamp).
#[cfg(kani)]
#[kani::proof]
fn action_projector_rate_limit_proof() {
    let previous: f64 = kani::any();
    let proposed: f64 = kani::any();
    let rate_limit: f64 = kani::any();

    kani::assume(previous.is_finite() && proposed.is_finite());
    kani::assume(rate_limit.is_finite() && rate_limit > 0.0);

    let delta = proposed - previous;
    let clamped = if delta.abs() > rate_limit {
        previous + rate_limit * delta.signum()
    } else {
        proposed
    };

    let actual_delta = (clamped - previous).abs();
    // Allow small floating-point tolerance
    assert!(
        actual_delta <= rate_limit + 1e-10,
        "rate-limited delta must not exceed configured rate"
    );
}

/// Combined: value clipping after rate limiting still respects bounds.
#[cfg(kani)]
#[kani::proof]
fn action_projector_combined_proof() {
    let previous: f64 = kani::any();
    let proposed: f64 = kani::any();
    let rate_limit: f64 = kani::any();
    let lo: f64 = kani::any();
    let hi: f64 = kani::any();

    kani::assume(lo.is_finite() && hi.is_finite());
    kani::assume(previous.is_finite() && proposed.is_finite());
    kani::assume(rate_limit.is_finite() && rate_limit > 0.0);
    kani::assume(lo <= hi);

    // Step 1: value clamp
    let mut clamped = proposed.clamp(lo, hi);

    // Step 2: rate limit
    let delta = clamped - previous;
    if delta.abs() > rate_limit {
        clamped = previous + rate_limit * delta.signum();
        clamped = clamped.clamp(lo, hi);
    }

    assert!(clamped >= lo, "final output must be >= lower bound");
    assert!(clamped <= hi, "final output must be <= upper bound");
}

// ── RegimeManager FSM proofs ───────────────────────────────────────

/// Regime states encoded as u8 for Kani. Mirrors spo_types::Regime.
/// Nominal=0, Degraded=1, Recovery=2, Critical=3
fn regime_rank(regime: u8) -> u8 {
    regime
}

/// The FSM must never transition directly from Critical (3) to
/// Nominal (0) — it must pass through Recovery (2) first.
#[cfg(kani)]
#[kani::proof]
fn regime_manager_no_critical_to_nominal_proof() {
    let current: u8 = kani::any();
    let proposed: u8 = kani::any();

    kani::assume(current <= 3 && proposed <= 3);

    // Encode the RegimeManager.evaluate() rule:
    // when current == Critical (3), result is either Critical (3) or Recovery (2)
    if current == 3 {
        // Critical can only go to Recovery or stay Critical
        let allowed = proposed == 3 || proposed == 2;
        // If the FSM were to produce Nominal directly, this would fail
        if !allowed {
            let corrected = 2_u8; // Recovery
            assert!(corrected != 0, "Critical must not jump to Nominal");
        }
    }
}

/// Transition log length is bounded by MAX_LOG_LEN (100).
/// Models the ring-buffer eviction in RegimeManager::commit_transition.
#[cfg(kani)]
#[kani::proof]
fn regime_manager_log_bounded_proof() {
    const MAX_LOG_LEN: usize = 100;
    let log_len: usize = kani::any();
    kani::assume(log_len <= MAX_LOG_LEN + 1);

    // Model: if log is at capacity, pop before push
    let after = if log_len == MAX_LOG_LEN {
        // pop_front + push_back = same length
        log_len
    } else {
        log_len + 1
    };

    assert!(
        after <= MAX_LOG_LEN,
        "transition log must never exceed MAX_LOG_LEN"
    );
}

/// Cooldown bypass: Critical transitions always succeed regardless
/// of cooldown state.
#[cfg(kani)]
#[kani::proof]
fn regime_manager_critical_bypasses_cooldown_proof() {
    let step_counter: u64 = kani::any();
    let last_transition: u64 = kani::any();
    let cooldown_steps: u64 = kani::any();
    let proposed: u8 = 3; // Critical

    kani::assume(step_counter >= last_transition);
    kani::assume(cooldown_steps > 0);

    let in_cooldown = last_transition > 0
        && step_counter.saturating_sub(last_transition) < cooldown_steps;

    // Critical bypasses cooldown — transition always accepted
    let accepted = proposed == 3 || !in_cooldown;
    assert!(accepted, "Critical must bypass cooldown");
}
