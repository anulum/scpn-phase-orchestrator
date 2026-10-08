// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Real native chimera contracts

//! Exercise exported checked and compatible APIs on genuine numerical inputs.

use spo_engine::chimera::{
    detect_chimera, local_order_parameter, try_detect_chimera, try_local_order_parameter,
};

#[test]
fn self_residue_is_not_a_neighbour() {
    assert_eq!(
        try_local_order_parameter(&[2.0], &[1e-16], 1).expect("admitted diagonal residue"),
        vec![0.0]
    );
    let state = try_detect_chimera(&[2.0], &[1e-16], 1).expect("valid isolated classification");
    assert!(state.coherent_indices.is_empty());
    assert_eq!(state.incoherent_indices, vec![0]);
}

#[test]
fn finite_unwrapped_phases_do_not_need_finite_differences() {
    let phases = [1e308, -1e308];
    assert_eq!(local_order_parameter(&phases, &[0.0; 4], 2), vec![0.0, 0.0]);
    assert_eq!(
        local_order_parameter(&phases, &[0.0, 1.0, 1.0, 0.0], 2),
        vec![1.0, 1.0]
    );
}

#[test]
fn positive_directed_edges_are_unweighted() {
    let phases = [0.0, 0.0, std::f64::consts::PI];
    let k = [0.0, 1e-300, 1e300, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let values = try_local_order_parameter(&phases, &k, 3).expect("valid directed graph");
    assert!(values[0] < 1e-15);
    assert_eq!(values[1], 0.0);
    assert_eq!(values[2], 1.0);
}

#[test]
fn malformed_requests_fail_without_indexing_or_allocating_from_n() {
    for (p, k, n) in [
        (vec![], vec![], usize::MAX),
        (vec![0.0], vec![], 1),
        (vec![0.0, 1.0], vec![0.0; 3], 2),
        (vec![f64::NAN], vec![0.0], 1),
        (vec![0.0], vec![f64::INFINITY], 1),
        (vec![0.0], vec![1e-14], 1),
        (vec![0.0], vec![], 0),
    ] {
        assert!(try_local_order_parameter(&p, &k, n).is_err());
        assert!(try_detect_chimera(&p, &k, n).is_err());
        assert!(local_order_parameter(&p, &k, n).is_empty());
        let legacy = detect_chimera(&p, &k, n);
        assert!(legacy.chimera_index.is_nan());
        assert!(legacy.local_order.is_empty());
    }
    assert!(try_local_order_parameter(&[], &[], 0)
        .expect("valid empty local order")
        .is_empty());
    assert_eq!(
        try_detect_chimera(&[], &[], 0)
            .expect("valid empty classification")
            .chimera_index,
        0.0
    );
}
