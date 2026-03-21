// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — WASM bindings

//! Minimal WASM wrapper exposing a single-step UPDE integrator
//! for browser and edge deployments.
//!
//! Build: `wasm-pack build --target web spo-kernel/crates/spo-wasm`

use std::cell::RefCell;
use wasm_bindgen::prelude::*;

thread_local! {
    static PHASES: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

/// Initialise the phase array with `n` oscillators at zero phase.
#[wasm_bindgen]
pub fn init(n: usize) {
    PHASES.with(|p| *p.borrow_mut() = vec![0.0; n]);
}

/// Advance all oscillators by one Euler step.
///
/// `omega_json`: JSON array of natural frequencies (length must match init `n`).
/// `coupling`: global coupling strength K.
/// `dt`: integration timestep.
///
/// Returns the Kuramoto order parameter R after the step.
#[must_use]
#[wasm_bindgen]
pub fn step(omega_json: &str, coupling: f64, dt: f64) -> f64 {
    let omega: Vec<f64> = serde_json::from_str(omega_json).unwrap_or_default();

    PHASES.with(|cell| {
        let mut phases = cell.borrow_mut();
        let n = phases.len();
        if n == 0 || omega.len() != n {
            return 0.0;
        }

        // Kuramoto mean-field coupling
        let (sin_sum, cos_sum): (f64, f64) = phases
            .iter()
            .fold((0.0, 0.0), |(s, c), &th| (s + th.sin(), c + th.cos()));
        let r = (sin_sum * sin_sum + cos_sum * cos_sum).sqrt() / n as f64;
        let psi = sin_sum.atan2(cos_sum);

        // Euler step: dθ_i/dt = ω_i + K·R·sin(ψ − θ_i)
        for i in 0..n {
            phases[i] += (omega[i] + coupling * r * (psi - phases[i]).sin()) * dt;
        }

        // Recompute R after step
        let (s2, c2): (f64, f64) = phases
            .iter()
            .fold((0.0, 0.0), |(s, c), &th| (s + th.sin(), c + th.cos()));
        (s2 * s2 + c2 * c2).sqrt() / n as f64
    })
}

/// Return current phases as JSON array.
#[must_use]
#[wasm_bindgen]
pub fn get_phases() -> String {
    PHASES.with(|cell| serde_json::to_string(&*cell.borrow()).unwrap_or_else(|_| "[]".to_string()))
}
