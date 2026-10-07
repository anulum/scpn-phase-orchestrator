// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Public finite-horizon trial contracts

//! Original exported APIs, independent analytic cases and checked failures.

use spo_engine::basin_stability::{basin_stability, try_basin_stability};
use spo_engine::bifurcation::{
    find_critical_coupling, steady_state_r, trace_sync_transition, try_find_critical_coupling,
    try_steady_state_r, try_trace_sync_transition,
};
use std::f64::consts::FRAC_PI_2;

#[test]
fn tiny_nonzero_couplings_have_their_declared_euler_effect() {
    let actual = try_steady_state_r(
        &[0.0, FRAC_PI_2],
        &[0.0; 2],
        &[0.0, 5e-31, 5e-31, 0.0],
        &[0.0; 4],
        2,
        1.0,
        1e30,
        0,
        1,
    )
    .expect("finite tiny-edge trial");
    assert!((actual - ((FRAC_PI_2 - 1.0) / 2.0).cos()).abs() < 2e-15);
}

#[test]
fn directed_signed_lagged_and_self_edges_follow_target_source_orientation() {
    let phases = [0.0, FRAC_PI_2];
    let omega = [0.2, -0.1];
    let k = [0.4, -0.7, 0.0, 0.5];
    let a: [f64; 4] = [0.3, -0.2, 0.0, -0.4];
    let dt = 0.09;
    let theta0 = dt * (omega[0] + k[0] * (-a[0]).sin() + k[1] * (FRAC_PI_2 - a[1]).sin());
    let theta1 = FRAC_PI_2 + dt * (omega[1] + k[3] * (-a[3]).sin());
    let expected = ((theta1 - theta0) / 2.0).cos().abs();
    let actual = steady_state_r(&phases, &omega, &k, &a, 2, 1.0, dt, 0, 1);
    assert!((actual - expected).abs() < 2e-15);
    assert_eq!(phases, [0.0, FRAC_PI_2]);
}

#[test]
fn shapes_and_finite_domains_are_checked_before_indexing() {
    let p = [0.0, 1.0];
    let o = [0.0; 2];
    let k = [0.0; 4];
    assert!(try_steady_state_r(&[], &[], &[], &[], 0, 1.0, 0.1, 0, 1).is_err());
    assert!(try_steady_state_r(&p, &o, &k, &k, usize::MAX, 1.0, 0.1, 0, 1).is_err());
    for (pp, oo, kk, aa) in [
        (&p[..1], &o[..], &k[..], &k[..]),
        (&p[..], &o[..1], &k[..], &k[..]),
        (&p[..], &o[..], &k[..3], &k[..]),
        (&p[..], &o[..], &k[..], &k[..3]),
    ] {
        assert!(try_steady_state_r(pp, oo, kk, aa, 2, 1.0, 0.1, 0, 0).is_err());
    }
    for (pp, oo, kk, aa) in [
        (&[f64::NAN, 0.0][..], &o[..], &k[..], &k[..]),
        (&p[..], &[0.0, f64::INFINITY][..], &k[..], &k[..]),
        (&p[..], &o[..], &[0.0, f64::NAN, 0.0, 0.0][..], &k[..]),
        (
            &p[..],
            &o[..],
            &k[..],
            &[0.0, 0.0, f64::NEG_INFINITY, 0.0][..],
        ),
    ] {
        assert!(try_steady_state_r(pp, oo, kk, aa, 2, 1.0, 0.1, 0, 0).is_err());
    }
    for (scale, dt) in [
        (f64::INFINITY, 0.1),
        (1.0, f64::NAN),
        (1.0, 0.0),
        (1.0, -0.1),
    ] {
        assert!(try_steady_state_r(&p, &o, &k, &k, 2, scale, dt, 0, 1).is_err());
    }
    assert!(steady_state_r(&p[..1], &o, &k, &k, 2, 1.0, 0.1, 0, 1).is_nan());
}

#[test]
fn arithmetic_errors_are_distinct_from_empty_windows_and_zero_edges() {
    let p = [0.0, 1.0];
    let o = [0.0; 2];
    let a = [0.0; 4];
    assert!(try_steady_state_r(&p, &o, &[0.0, 1e308, 0.0, 0.0], &a, 2, 2.0, 1.0, 0, 1).is_err());
    assert!(try_steady_state_r(
        &[1e308, -1e308],
        &o,
        &[0.0, 1.0, 0.0, 0.0],
        &a,
        2,
        1.0,
        1.0,
        0,
        1
    )
    .is_err());
    assert!(try_steady_state_r(&p, &[1e308; 2], &a, &a, 2, 1.0, 2.0, 0, 1).is_err());
    assert!(try_steady_state_r(&p, &[1e308; 2], &a, &a, 2, 1.0, 2.0, 1, 1).is_err());
    // Two finite edge contributions overflow the velocity before dt scaling.
    assert!(try_steady_state_r(
        &[0.0, FRAC_PI_2, FRAC_PI_2],
        &[0.0; 3],
        &[0.0, 1e308, 1e308, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        &[0.0; 9],
        3,
        1.0,
        1e-308,
        0,
        1,
    )
    .is_err());
    assert_eq!(
        try_steady_state_r(&p, &[1e308; 2], &a, &a, 2, 1.0, 2.0, usize::MAX, 0)
            .expect("unused transient"),
        0.0
    );
    let actual =
        try_steady_state_r(&[1e308, -1e308], &o, &a, &a, 2, 1.0, 1.0, 0, 1).expect("zero edges");
    assert!((actual - (1e308_f64).cos().abs()).abs() < 2e-15);
}

#[test]
fn post_step_windows_and_independent_grid_trials_agree() {
    let p = [0.0, 0.9];
    let o = [0.2, -0.3];
    let k = [0.0, 0.7, -0.2, 0.0];
    let a = [0.0; 4];
    let (grid, values, critical) =
        try_trace_sync_transition(&o, &k, &a, 2, &p, 0.0, 3.0, 4, 0.04, 2, 3).expect("valid grid");
    assert_eq!(grid, vec![0.0, 1.0, 2.0, 3.0]);
    for (&scale, &value) in grid.iter().zip(&values) {
        assert_eq!(value, steady_state_r(&p, &o, &k, &a, 2, scale, 0.04, 2, 3));
    }
    assert!(critical.is_nan());
    let (_, empty, critical) = trace_sync_transition(&o, &k, &a, 2, &p, 0.0, 3.0, 4, 0.04, 20, 0);
    assert_eq!(empty, vec![0.0; 4]);
    assert!(critical.is_nan());
    assert!(find_critical_coupling(&o, &k, &a, 2, &p, 0.04, 20, 0, 0.01).is_nan());
}

#[test]
fn invalid_grid_search_and_composite_arithmetic_propagate_errors() {
    let p = [0.0, 1.0];
    let o = [0.0; 2];
    let k = [0.0; 4];
    for (lo, hi, points) in [
        (0.0, 1.0, 0),
        (1.0, 0.0, 2),
        (0.0, f64::NAN, 2),
        (f64::INFINITY, 2.0, 2),
    ] {
        assert!(try_trace_sync_transition(&o, &k, &k, 2, &p, lo, hi, points, 0.1, 0, 1).is_err());
    }
    for tol in [0.0, -0.1, f64::INFINITY, f64::NAN] {
        assert!(try_find_critical_coupling(&o, &k, &k, 2, &p, 0.1, 0, 1, tol).is_err());
    }
    assert!(try_trace_sync_transition(&[1e308; 2], &k, &k, 2, &p, 0.0, 1.0, 2, 2.0, 0, 1).is_err());
    assert!(try_find_critical_coupling(&[1e308; 2], &k, &k, 2, &p, 2.0, 0, 1, 0.1).is_err());
    let (grid, values, critical) = trace_sync_transition(&o, &k, &k, 2, &p, 0.0, 1.0, 0, 0.1, 0, 1);
    assert!(grid.is_empty() && values.is_empty() && critical.is_nan());
}

#[test]
fn lcg_sampling_and_inclusive_thresholds_retain_their_legacy_contract() {
    let o = [0.0; 2];
    let k = [0.0; 4];
    let mut state = 29_u64;
    let result =
        try_basin_stability(&o, &k, &k, 2, 0.1, 0, 1, 8, 0.5, state).expect("valid LCG sampler");
    for &r in &result.r_finals {
        let mut p = [0.0; 2];
        for phase in &mut p {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *phase = ((state >> 33) as f64 / (1_u64 << 31) as f64) * std::f64::consts::TAU;
        }
        assert!((r - ((p[1] - p[0]) / 2.0).cos().abs()).abs() < 2e-15);
    }
    assert_eq!(
        result.n_converged,
        result.r_finals.iter().filter(|&&r| r >= 0.5).count()
    );
    assert_eq!(result.s_b, result.n_converged as f64 / 8.0);
    let empty = basin_stability(&[], &[], &[], 0, 0.1, 0, 1, 0, 0.8, 29);
    assert_eq!(empty.s_b, 0.0);
    assert!(empty.r_finals.is_empty());
    let identity = try_basin_stability(&o, &k, &k, 2, 0.1, 99, 0, 4, 0.0, 29)
        .expect("zero window classification");
    assert_eq!(identity.s_b, 1.0);
    assert_eq!(identity.n_converged, 4);
    for threshold in [-0.1, 1.1, f64::NAN] {
        assert!(try_basin_stability(&o, &k, &k, 2, 0.1, 0, 1, 1, threshold, 29).is_err());
    }
    assert!(try_basin_stability(&[1e308; 2], &k, &k, 2, 2.0, 0, 1, 1, 0.5, 29).is_err());
    let checked_empty =
        try_basin_stability(&o, &k, &k, 2, 0.1, 0, 1, 0, 0.5, 29).expect("valid empty sampler");
    assert_eq!(checked_empty.s_b, 0.0);
    assert_eq!(checked_empty.n_converged, 0);
    assert!(checked_empty.r_finals.is_empty());
    let invalid_legacy = basin_stability(&o, &k, &k, 2, 0.0, 0, 1, 1, 0.5, 29);
    assert!(invalid_legacy.s_b.is_nan());
    assert!(invalid_legacy.r_finals.is_empty());
    assert_eq!(invalid_legacy.n_converged, 0);
}

#[test]
fn analytic_first_upcrossing_and_subthreshold_search_midpoints_agree() {
    let gap = std::f64::consts::PI - 0.01;
    let phases = [0.0, gap];
    let omega = [0.0; 2];
    let coupling = [0.0, 1.0, 1.0, 0.0];
    let lag = [0.0; 4];
    let expected = |scale: f64| ((gap - 2.0 * 0.8 * scale * gap.sin()) / 2.0).cos().abs();
    let (grid, values, crossing) =
        try_trace_sync_transition(&omega, &coupling, &lag, 2, &phases, 0.0, 20.0, 3, 0.8, 0, 1)
            .expect("finite analytic upcrossing");
    assert_eq!(grid, vec![0.0, 10.0, 20.0]);
    for (&scale, &value) in grid.iter().zip(&values) {
        assert!((value - expected(scale)).abs() < 2e-15);
    }
    assert!(values[0] < 0.1 && values[1] < 0.1 && values[2] >= 0.1);
    let interpolated = 10.0 + 10.0 * (0.1 - expected(10.0)) / (expected(20.0) - expected(10.0));
    assert!((crossing - interpolated).abs() < 2e-13);
    assert!(expected(10.0) < 0.1 && expected(15.0) >= 0.1);
    assert_eq!(
        try_find_critical_coupling(&omega, &coupling, &lag, 2, &phases, 0.8, 0, 1, 6.0)
            .expect("finite analytic search"),
        12.5
    );
}

#[test]
fn finite_extreme_signed_grid_interpolation_does_not_overflow() {
    let gap = std::f64::consts::PI - 0.1;
    let phases = [0.0, gap];
    let omega = [0.0; 2];
    let coupling = [0.0, 1e-308, 1e-308, 0.0];
    let lag = [0.0; 4];
    let (grid, values, critical) = try_trace_sync_transition(
        &omega, &coupling, &lag, 2, &phases, -1e308, 1e308, 2, 1.0, 0, 1,
    )
    .expect("finite signed extreme grid");
    assert_eq!(grid, vec![-1e308, 1e308]);
    let expected_lo = ((gap + 2.0 * (1e308 * 1e-308) * gap.sin()) / 2.0)
        .cos()
        .abs();
    let expected_hi = ((gap - 2.0 * (1e308 * 1e-308) * gap.sin()) / 2.0)
        .cos()
        .abs();
    assert!((values[0] - expected_lo).abs() < 2e-15);
    assert!((values[1] - expected_hi).abs() < 2e-15);
    assert!(expected_lo < 0.1 && expected_hi >= 0.1);
    let fraction = (0.1 - expected_lo) / (expected_hi - expected_lo);
    let expected = (1.0 - fraction) * -1e308 + fraction * 1e308;
    assert!(critical.is_finite());
    assert!((critical - expected).abs() / 1e308 < 2e-15);
}

#[test]
fn superthreshold_lower_bracket_preserves_legacy_interval_midpoint() {
    let phases: [f64; 4] = [0.0, 1.0, 2.0, 3.0];
    let omega = [0.0; 4];
    let graph = [0.0; 16];
    let cosine = phases.iter().map(|value| value.cos()).sum::<f64>() / 4.0;
    let sine = phases.iter().map(|value| value.sin()).sum::<f64>() / 4.0;
    let expected = cosine.hypot(sine);
    assert!(expected > 0.1);
    let (_, values, crossing) =
        try_trace_sync_transition(&omega, &graph, &graph, 4, &phases, 0.0, 20.0, 3, 0.01, 0, 1)
            .expect("constant finite response");
    assert!(values.iter().all(|&value| (value - expected).abs() < 2e-15));
    assert!(crossing.is_nan());
    assert_eq!(
        try_find_critical_coupling(&omega, &graph, &graph, 4, &phases, 0.01, 0, 1, 0.05)
            .expect("compatible interval result"),
        0.01953125
    );
    assert_eq!(
        find_critical_coupling(&omega, &graph, &graph, 4, &phases, 0.01, 0, 1, 0.05),
        0.01953125
    );
}

#[test]
fn legacy_native_singleton_equal_and_negative_grids_remain_valid() {
    let p = [0.0, 1.0];
    let o = [0.0; 2];
    let k = [0.0; 4];
    for (lo, hi, points) in [(-1.0, 1.0, 3), (1.0, 1.0, 2), (0.0, 1.0, 1)] {
        let (grid, values, critical) =
            try_trace_sync_transition(&o, &k, &k, 2, &p, lo, hi, points, 0.1, 0, 1)
                .expect("legacy native grid");
        assert_eq!(grid.len(), points);
        assert_eq!(grid[0], lo);
        if points > 1 {
            assert_eq!(grid[points - 1], hi);
        }
        assert!(values.iter().all(|&r| (r - 0.5_f64.cos()).abs() < 2e-15));
        assert!(critical.is_nan());
    }
}
