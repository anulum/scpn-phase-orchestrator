// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Digital-twin divergence kernel

use std::f64::consts::PI;

const TWO_PI: f64 = 2.0 * PI;

/// Phase Jensen–Shannon divergence and order-parameter Wasserstein-1 distance
/// between a model tick and its observed twin tick.
///
/// Mirrors the NumPy reference in
/// `scpn_phase_orchestrator.monitor.twin_confidence` to within 1e-9:
///
/// * Phases are wrapped to `[0, 2π)`, binned into `n_bins` equal-width bins, and
///   normalised to probability mass functions; the symmetric Jensen–Shannon
///   divergence (natural log, range `[0, ln 2]`) is returned.
/// * The order-parameter windows are sorted ascending and the mean absolute
///   difference (the closed-form 1-D Wasserstein-1 distance, range `[0, 1]`) is
///   returned.
///
/// # Arguments
/// * `model_phases` - (N,) model phases in radians
/// * `observed_phases` - (N,) observed phases in radians
/// * `model_order` - (W,) model order-parameter window in [0, 1]
/// * `observed_order` - (W,) observed order-parameter window in [0, 1]
/// * `n_bins` - number of phase histogram bins
///
/// # Returns
/// `[phase_js_divergence, order_wasserstein]`.
///
/// # Errors
/// Returns an error if the phase or order lengths disagree, an array is empty,
/// or `n_bins` is zero.
pub fn twin_divergence(
    model_phases: &[f64],
    observed_phases: &[f64],
    model_order: &[f64],
    observed_order: &[f64],
    n_bins: usize,
) -> Result<[f64; 2], String> {
    if model_phases.is_empty() {
        return Err("model_phases must contain at least one phase".to_string());
    }
    if observed_phases.len() != model_phases.len() {
        return Err(format!(
            "observed_phases length {} != model_phases length {}",
            observed_phases.len(),
            model_phases.len()
        ));
    }
    if model_order.is_empty() {
        return Err("model_order must contain at least one sample".to_string());
    }
    if observed_order.len() != model_order.len() {
        return Err(format!(
            "observed_order length {} != model_order length {}",
            observed_order.len(),
            model_order.len()
        ));
    }
    if n_bins == 0 {
        return Err("n_bins must be a positive integer".to_string());
    }

    let p = phase_histogram(model_phases, n_bins);
    let q = phase_histogram(observed_phases, n_bins);
    let js = jensen_shannon(&p, &q);
    let w1 = wasserstein1(model_order, observed_order);
    Ok([js, w1])
}

fn phase_histogram(phases: &[f64], n_bins: usize) -> Vec<f64> {
    let width = TWO_PI / n_bins as f64;
    let mut counts = vec![0.0_f64; n_bins];
    for &phase in phases {
        let wrapped = phase - (phase / TWO_PI).floor() * TWO_PI;
        let mut idx = (wrapped / width).floor() as i64;
        if idx < 0 {
            idx = 0;
        }
        let upper = (n_bins - 1) as i64;
        if idx > upper {
            idx = upper;
        }
        counts[idx as usize] += 1.0;
    }
    let total: f64 = counts.iter().sum();
    if total <= 0.0 {
        return vec![1.0 / n_bins as f64; n_bins];
    }
    counts.iter().map(|&c| c / total).collect()
}

fn jensen_shannon(p: &[f64], q: &[f64]) -> f64 {
    let m: Vec<f64> = p.iter().zip(q).map(|(&pi, &qi)| 0.5 * (pi + qi)).collect();
    0.5 * kl(p, &m) + 0.5 * kl(q, &m)
}

fn kl(p: &[f64], m: &[f64]) -> f64 {
    p.iter()
        .zip(m)
        .filter(|(&pi, _)| pi > 0.0)
        .map(|(&pi, &mi)| pi * (pi / mi).ln())
        .sum()
}

fn wasserstein1(model_order: &[f64], observed_order: &[f64]) -> f64 {
    let mut sorted_model = model_order.to_vec();
    let mut sorted_obs = observed_order.to_vec();
    sorted_model.sort_by(f64::total_cmp);
    sorted_obs.sort_by(f64::total_cmp);
    let sum: f64 = sorted_model
        .iter()
        .zip(&sorted_obs)
        .map(|(&a, &b)| (a - b).abs())
        .sum();
    sum / sorted_model.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_streams_have_zero_divergence() {
        let phases = [0.1, 1.2, 2.3, 3.4, 4.5];
        let order = [0.4, 0.5, 0.6];
        let out = twin_divergence(&phases, &phases, &order, &order, 36).unwrap();
        assert!(out[0].abs() < 1e-12);
        assert!(out[1].abs() < 1e-12);
    }

    #[test]
    fn disjoint_phase_support_approaches_ln2() {
        let model = [0.05; 8];
        let observed = [PI + 0.05; 8];
        let order = [0.5; 4];
        let out = twin_divergence(&model, &observed, &order, &order, 36).unwrap();
        assert!((out[0] - 2.0_f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn order_shift_gives_exact_wasserstein() {
        let phases = [0.1, 0.2, 0.3];
        let model = [0.2, 0.4, 0.6];
        let observed = [0.5, 0.7, 0.9];
        let out = twin_divergence(&phases, &phases, &model, &observed, 12).unwrap();
        assert!((out[1] - 0.3).abs() < 1e-12);
    }

    #[test]
    fn length_mismatch_errors() {
        let a = [0.1, 0.2];
        let b = [0.1];
        let order = [0.5];
        assert!(twin_divergence(&a, &b, &order, &order, 8).is_err());
    }
}
