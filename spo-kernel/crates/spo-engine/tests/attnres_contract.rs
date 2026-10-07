// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Public phase attention numerical contracts

//! Exercise the exported native coupling API with independent analytic cases.

use spo_engine::attnres::attnres_modulate;

fn identity_weights(width: usize, heads: usize) -> (Vec<f64>, Vec<f64>) {
    let head_width = width / heads;
    let mut projection = vec![0.0; width * width];
    for head in 0..heads {
        for feature in 0..head_width {
            projection[head * width * head_width
                + (head * head_width + feature) * head_width
                + feature] = 1.0;
        }
    }
    let mut output = vec![0.0; width * width];
    for feature in 0..width {
        output[feature * width + feature] = 1.0;
    }
    (projection, output)
}

#[test]
fn two_orthogonal_phases_have_half_strength_readout() {
    let (projection, output) = identity_weights(2, 1);
    let coupling = [0.0, -0.3, -0.3, 0.0];
    let phases = [0.0, std::f64::consts::FRAC_PI_2];
    let actual = attnres_modulate(
        &coupling,
        &phases,
        &projection,
        &projection,
        &projection,
        &output,
        2,
        1,
        -1,
        1.0,
        0.5,
    )
    .expect("orthogonal signed pair is valid");
    assert!((actual[1] + 0.375).abs() < 1e-12);
    assert_eq!(actual[1], actual[2]);
    assert_eq!(actual[0], 0.0);
    assert_eq!(actual[3], 0.0);
}

#[test]
fn empty_and_identity_calls_validate_before_returning() {
    let (projection, output) = identity_weights(4, 2);
    assert_eq!(
        attnres_modulate(
            &[],
            &[],
            &projection,
            &projection,
            &projection,
            &output,
            0,
            2,
            -1,
            1.0,
            0.5
        )
        .expect("empty graph with valid projections is valid"),
        Vec::<f64>::new()
    );
    assert!(attnres_modulate(
        &[],
        &[],
        &projection,
        &projection,
        &projection,
        &[],
        0,
        2,
        -1,
        1.0,
        0.0
    )
    .is_err());
    assert!(attnres_modulate(
        &[],
        &[],
        &projection,
        &projection,
        &projection,
        &output,
        usize::MAX,
        2,
        -1,
        1.0,
        0.0
    )
    .is_err());
}

#[test]
fn finite_hyperparameters_and_canonical_band_are_required() {
    let (projection, output) = identity_weights(2, 1);
    let coupling = [0.0, 0.3, 0.3, 0.0];
    let phases = [0.0, 0.4];
    for strength in [f64::NAN, f64::INFINITY, -0.1] {
        assert!(attnres_modulate(
            &coupling,
            &phases,
            &projection,
            &projection,
            &projection,
            &output,
            2,
            1,
            -1,
            1.0,
            strength
        )
        .is_err());
    }
    for band in [-2, 0] {
        assert!(attnres_modulate(
            &coupling,
            &phases,
            &projection,
            &projection,
            &projection,
            &output,
            2,
            1,
            band,
            1.0,
            0.5
        )
        .is_err());
    }
}

#[test]
fn malformed_shapes_and_topologies_are_not_identity_iterates() {
    let (projection, output) = identity_weights(2, 1);
    for coupling in [
        [0.1, 0.3, 0.3, 0.0],
        [0.0, 0.3, 0.1, 0.0],
        [0.0, f64::NAN, f64::NAN, 0.0],
    ] {
        assert!(attnres_modulate(
            &coupling,
            &[0.0, 0.4],
            &projection,
            &projection,
            &projection,
            &output,
            2,
            1,
            -1,
            1.0,
            0.0
        )
        .is_err());
    }
    for width in [0, 1, 3] {
        let values = vec![0.0; width * width];
        assert!(attnres_modulate(
            &[0.0],
            &[0.0],
            &values,
            &values,
            &values,
            &values,
            1,
            1,
            -1,
            1.0,
            0.0
        )
        .is_err());
    }
}

#[test]
fn nonfinite_inputs_and_unrepresentable_intermediates_are_errors() {
    let (projection, output) = identity_weights(2, 1);
    let coupling = [0.0, 0.3, 0.3, 0.0];
    assert!(attnres_modulate(
        &coupling,
        &[f64::INFINITY, 0.0],
        &projection,
        &projection,
        &projection,
        &output,
        2,
        1,
        -1,
        1.0,
        0.5
    )
    .is_err());
    assert!(attnres_modulate(
        &coupling,
        &[0.0, 0.4],
        &projection,
        &projection,
        &projection,
        &output,
        2,
        1,
        -1,
        f64::from_bits(1),
        0.5
    )
    .is_err());
    let huge = vec![f64::MAX; 4];
    assert!(attnres_modulate(
        &coupling,
        &[0.4, 0.5],
        &huge,
        &projection,
        &projection,
        &output,
        2,
        1,
        -1,
        1.0,
        0.5
    )
    .is_err());
}

#[test]
fn symmetrisation_does_not_overflow_a_representable_coupling() {
    let values = [0.0; 4];
    let coupling = [0.0, 1.0e308, 1.0e308, 0.0];
    let result = attnres_modulate(
        &coupling,
        &[0.0, 0.4],
        &values,
        &values,
        &values,
        &values,
        2,
        1,
        -1,
        1.0,
        0.1,
    )
    .expect("representable large coupling is valid");
    assert!((result[1] / 1.0e308 - 1.05).abs() < 1e-14);
}

#[test]
fn invalid_native_metadata_and_each_nonfinite_buffer_are_refused() {
    let (projection, output) = identity_weights(2, 1);
    let buffers = vec![
        vec![0.0, 0.3, 0.3, 0.0],
        vec![0.1, 0.7],
        projection.clone(),
        projection.clone(),
        projection.clone(),
        output,
    ];
    for index in 0..buffers.len() {
        let mut invalid = buffers.clone();
        invalid[index][0] = f64::NAN;
        assert!(attnres_modulate(
            &invalid[0],
            &invalid[1],
            &invalid[2],
            &invalid[3],
            &invalid[4],
            &invalid[5],
            2,
            1,
            -1,
            1.0,
            0.0
        )
        .is_err());
    }
    for temperature in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(attnres_modulate(
            &buffers[0],
            &buffers[1],
            &buffers[2],
            &buffers[3],
            &buffers[4],
            &buffers[5],
            2,
            1,
            -1,
            temperature,
            0.5
        )
        .is_err());
    }
    for heads in [0, 3] {
        assert!(attnres_modulate(
            &buffers[0],
            &buffers[1],
            &buffers[2],
            &buffers[3],
            &buffers[4],
            &buffers[5],
            2,
            heads,
            -1,
            1.0,
            0.5
        )
        .is_err());
    }
    for index in 0..buffers.len() {
        let mut invalid = buffers.clone();
        invalid[index].push(0.0);
        assert!(attnres_modulate(
            &invalid[0],
            &invalid[1],
            &invalid[2],
            &invalid[3],
            &invalid[4],
            &invalid[5],
            2,
            1,
            -1,
            1.0,
            0.0
        )
        .is_err());
    }
    assert!(attnres_modulate(&[], &[], &[], &[], &[], &[], usize::MAX, 1, -1, 1.0, 0.5).is_err());
}

#[test]
fn native_masking_preserves_unattended_edges_and_empty_neighbour_rows() {
    let (projection, output) = identity_weights(2, 1);
    let coupling = [0.0, 0.3, 0.2, 0.3, 0.0, 0.0, 0.2, 0.0, 0.0];
    let result = attnres_modulate(
        &coupling,
        &[0.1, 0.7, 1.2],
        &projection,
        &projection,
        &projection,
        &output,
        3,
        1,
        1,
        1.0,
        0.5,
    )
    .expect("banded graph has valid isolated attention rows");
    let expected = 0.3 * (1.0 + 0.25 * (1.0 + 0.6_f64.cos() / (1.0_f64 + 1e-12).powi(2)));
    assert!((result[1] - expected).abs() < 1e-12);
    assert_eq!(result[2], 0.2);
    assert_eq!(result[6], 0.2);
    assert_eq!(result[5], 0.0);
    assert_eq!(result[7], 0.0);
    let excluded = [0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0];
    let enormous: Vec<f64> = projection.iter().map(|value| value * 1e155).collect();
    assert_eq!(
        attnres_modulate(
            &excluded,
            &[0.1, 0.4, 0.9],
            &enormous,
            &enormous,
            &projection,
            &output,
            3,
            1,
            1,
            1.0,
            0.5
        )
        .expect("unattended logits need not be evaluated"),
        excluded
    );
}

#[test]
fn native_allowed_logits_norms_and_final_coupling_must_be_representable() {
    let (projection, output) = identity_weights(2, 1);
    let coupling = [0.0, 0.3, 0.3, 0.0];
    let enormous: Vec<f64> = projection.iter().map(|value| value * 1e155).collect();
    assert!(attnres_modulate(
        &coupling,
        &[0.1, 0.7],
        &enormous,
        &enormous,
        &projection,
        &output,
        2,
        1,
        -1,
        1.0,
        0.5
    )
    .is_err());
    assert!(attnres_modulate(
        &coupling,
        &[0.1, 0.7],
        &projection,
        &projection,
        &enormous,
        &output,
        2,
        1,
        -1,
        1.0,
        0.5
    )
    .is_err());
    let zero = [0.0; 4];
    assert!(attnres_modulate(
        &[0.0, f64::MAX, f64::MAX, 0.0],
        &[0.1, 0.7],
        &zero,
        &zero,
        &zero,
        &zero,
        2,
        1,
        -1,
        1.0,
        0.5
    )
    .is_err());
}

#[test]
fn large_gain_cannot_turn_cosine_roundoff_into_a_sign_reversal() {
    let (projection, output) = identity_weights(2, 1);
    let large_output: Vec<f64> = output.iter().map(|value| value * 1e120).collect();
    let phase = 0.1972727272727273;
    for edge in [0.3, -0.3] {
        for antipodal in [true, false] {
            let second = if antipodal {
                phase + std::f64::consts::PI
            } else {
                phase
            };
            let result = attnres_modulate(
                &[0.0, edge, edge, 0.0],
                &[phase, second],
                &projection,
                &projection,
                &projection,
                &large_output,
                2,
                1,
                -1,
                1.0,
                1e16,
            )
            .expect("endpoint score must remain within its physical bounds");
            let expected = if antipodal { edge } else { edge * (1.0 + 1e16) };
            assert!((result[1] - expected).abs() <= 1e-12 + expected.abs() * 2e-15);
            assert_eq!(result[1].is_sign_negative(), edge.is_sign_negative());
            assert!(result[1].abs() >= edge.abs());
            assert!(result[1].abs() <= edge.abs() * (1.0 + 1e16));
        }
    }
}

#[test]
fn empty_graph_does_not_evaluate_an_unused_temperature_scale() {
    let (projection, output) = identity_weights(2, 1);
    assert!(attnres_modulate(
        &[],
        &[],
        &projection,
        &projection,
        &projection,
        &output,
        0,
        1,
        -1,
        f64::from_bits(1),
        0.5
    )
    .expect("valid empty graph has no attention calculation")
    .is_empty());
}
