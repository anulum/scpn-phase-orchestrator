// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Property-based tests

use proptest::prelude::*;
use spo_engine::{compute_order_parameter, compute_plv, CouplingBuilder};
use spo_types::CouplingConfig;
use std::f64::consts::TAU;

fn phases_vec(n: usize) -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(0.0..TAU, n)
}

proptest! {
    #[test]
    fn order_parameter_r_in_unit_interval(phases in phases_vec(16)) {
        let (r, _) = compute_order_parameter(&phases);
        prop_assert!(r >= 0.0, "R={r} < 0");
        prop_assert!(r <= 1.0 + 1e-12, "R={r} > 1");
    }

    #[test]
    fn order_parameter_psi_in_range(phases in phases_vec(8)) {
        let (_, psi) = compute_order_parameter(&phases);
        prop_assert!(psi >= 0.0, "psi={psi} < 0");
        prop_assert!(psi < TAU + 1e-12, "psi={psi} >= 2pi");
    }

    #[test]
    fn plv_in_unit_interval(
        a in phases_vec(20),
        b in phases_vec(20),
    ) {
        let plv = compute_plv(&a, &b).unwrap();
        prop_assert!(plv >= 0.0, "PLV={plv} < 0");
        prop_assert!(plv <= 1.0 + 1e-12, "PLV={plv} > 1");
    }

    #[test]
    fn plv_identical_is_one(phases in phases_vec(10)) {
        let plv = compute_plv(&phases, &phases).unwrap();
        prop_assert!((plv - 1.0).abs() < 1e-9, "PLV={plv} != 1 for identical");
    }

    #[test]
    fn coupling_symmetric_and_nonneg(
        n in 2..16_usize,
        base in 0.01..2.0_f64,
        decay in 0.01..5.0_f64,
    ) {
        let cfg = CouplingConfig { base_strength: base, decay_alpha: decay };
        let cs = CouplingBuilder::build(n, &cfg).unwrap();
        for i in 0..n {
            prop_assert_eq!(cs.knm[i * n + i], 0.0);
            for j in 0..n {
                let diff = (cs.knm[i * n + j] - cs.knm[j * n + i]).abs();
                prop_assert!(diff < 1e-12, "asymmetric at [{},{}]: diff={}", i, j, diff);
                prop_assert!(cs.knm[i * n + j] >= 0.0, "negative at [{},{}]", i, j);
            }
        }
    }

    #[test]
    fn coupling_monotonic_decay(
        n in 3..12_usize,
        base in 0.1..2.0_f64,
        decay in 0.1..5.0_f64,
    ) {
        let cfg = CouplingConfig { base_strength: base, decay_alpha: decay };
        let cs = CouplingBuilder::build(n, &cfg).unwrap();
        // First row: K[0,1] >= K[0,2] >= ... >= K[0,n-1]
        for j in 2..n {
            prop_assert!(
                cs.knm[j - 1] >= cs.knm[j],
                "K[0,{}]={} < K[0,{}]={}",
                j - 1, cs.knm[j - 1], j, cs.knm[j]
            );
        }
    }
}
