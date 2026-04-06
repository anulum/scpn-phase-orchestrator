// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — WASM bindings (Optimized)

use wasm_bindgen::prelude::*;
use js_sys::Float64Array;

#[wasm_bindgen]
pub struct WasmEngine {
    n: usize,
    phases: Vec<f64>,
    sin_theta: Vec<f64>,
    cos_theta: Vec<f64>,
}

#[wasm_bindgen]
impl WasmEngine {
    #[wasm_bindgen(constructor)]
    pub fn new(n: usize) -> Self {
        Self {
            n,
            phases: vec![0.0; n],
            sin_theta: vec![0.0; n],
            cos_theta: vec![0.0; n],
        }
    }

    pub fn set_phases(&mut self, new_phases: Float64Array) {
        new_phases.copy_to(&mut self.phases);
    }

    pub fn get_phases(&self) -> Float64Array {
        unsafe { Float64Array::view(&self.phases) }
    }

    pub fn step(&mut self, omegas: Float64Array, coupling: f64, dt: f64) -> f64 {
        let n = self.n;
        let mut sum_s = 0.0;
        let mut sum_c = 0.0;

        for i in 0..n {
            let (s, c) = self.phases[i].sin_cos();
            self.sin_theta[i] = s;
            self.cos_theta[i] = c;
            sum_s += s;
            sum_c += c;
        }

        let inv_n = 1.0 / n as f64;
        let r = (sum_s * sum_s + sum_c * sum_c).sqrt() * inv_n;
        let (s_psi, c_psi) = if r > 1e-15 {
            let psi = sum_s.atan2(sum_c);
            psi.sin_cos()
        } else {
            (0.0, 0.0)
        };

        let ks = coupling * r;
        let omegas_vec = omegas.to_vec();

        for i in 0..n {
            // sin(psi - theta) = s_psi * cos_theta - c_psi * sin_theta
            let coupling_term = ks * (s_psi * self.cos_theta[i] - c_psi * self.sin_theta[i]);
            self.phases[i] += (omegas_vec[i] + coupling_term) * dt;
        }

        r
    }

    pub fn run(&mut self, omegas: Float64Array, coupling: f64, dt: f64, n_steps: usize) -> f64 {
        let mut r = 0.0;
        for _ in 0..n_steps {
            r = self.step(omegas.clone(), coupling, dt);
        }
        r
    }
}
