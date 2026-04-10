// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Vectorised LIF neuron ensemble

use rayon::prelude::*;
use spo_types::SpoError;

#[derive(Debug, Clone)]
pub struct LIFParams {
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau_mem: f64,
    pub dt: f64,
    pub resistance: f64,
    pub refractory_period: i32,
    pub noise_std: f64,
}

impl Default for LIFParams {
    fn default() -> Self {
        Self {
            v_rest: 0.0,
            v_reset: 0.0,
            v_threshold: 1.0,
            tau_mem: 20.0,
            dt: 1.0,
            resistance: 1.0,
            refractory_period: 0,
            noise_std: 0.0,
        }
    }
}

pub struct LIFEnsemble {
    n_layers: usize,
    neurons_per_layer: usize,
    n_total: usize,
    params: LIFParams,
    v: Vec<f64>,
    refractory: Vec<i32>,
    spike_counts: Vec<u64>,
    step_count: u64,
    dt_over_tau: f64,
    r_dt: f64,
    input_terms: Vec<f64>,
}

impl LIFEnsemble {
    pub fn new(
        n_layers: usize,
        neurons_per_layer: usize,
        params: LIFParams,
    ) -> Result<Self, SpoError> {
        if n_layers == 0 || neurons_per_layer == 0 {
            return Err(SpoError::InvalidConfig(
                "n_layers and neurons_per_layer must be > 0".into(),
            ));
        }
        if params.tau_mem <= 0.0 {
            return Err(SpoError::InvalidConfig("tau_mem must be > 0".into()));
        }
        let n_total = n_layers * neurons_per_layer;
        Ok(Self {
            n_layers,
            neurons_per_layer,
            n_total,
            v: vec![params.v_rest; n_total],
            refractory: vec![0; n_total],
            spike_counts: vec![0u64; n_total],
            step_count: 0,
            dt_over_tau: params.dt / params.tau_mem,
            r_dt: params.resistance * params.dt,
            input_terms: vec![0.0; n_total],
            params,
        })
    }

    pub fn step(&mut self, currents: &[f64], n_substeps: usize) -> Result<Vec<f64>, SpoError> {
        if currents.len() != self.n_layers {
            return Err(SpoError::InvalidConfig(format!(
                "expected {} currents, got {}",
                self.n_layers,
                currents.len()
            )));
        }
        let npl = self.neurons_per_layer;
        let r_dt = self.r_dt;

        self.input_terms
            .par_chunks_mut(npl)
            .enumerate()
            .for_each(|(layer, chunk)| {
                chunk.fill(currents[layer] * r_dt);
            });

        let v_rest = self.params.v_rest;
        let v_reset = self.params.v_reset;
        let v_threshold = self.params.v_threshold;
        let refr_p = self.params.refractory_period;
        let dt_tau = self.dt_over_tau;

        for _ in 0..n_substeps {
            self.v
                .par_iter_mut()
                .zip(self.refractory.par_iter_mut())
                .zip(self.spike_counts.par_iter_mut())
                .zip(self.input_terms.par_iter())
                .for_each(|(((v, refr), sc), &inp)| {
                    if *refr > 0 {
                        *v = v_rest;
                        *refr -= 1;
                    } else {
                        *v += -(*v - v_rest) * dt_tau + inp;
                        if *v >= v_threshold {
                            *sc += 1;
                            *v = v_reset;
                            *refr = refr_p;
                        }
                    }
                });
            self.step_count += 1;
        }
        self.compute_rates()
    }

    fn compute_rates(&self) -> Result<Vec<f64>, SpoError> {
        let duration_s = self.step_count as f64 * self.params.dt / 1000.0;
        if duration_s <= 0.0 {
            return Ok(vec![0.0; self.n_layers]);
        }
        let npl = self.neurons_per_layer;
        let mut rates = Vec::with_capacity(self.n_layers);
        for layer in 0..self.n_layers {
            let start = layer * npl;
            let total: u64 = self.spike_counts[start..start + npl].iter().sum();
            rates.push(total as f64 / (npl as f64 * duration_s));
        }
        Ok(rates)
    }

    pub fn neuron_state(&self, i: usize) -> Option<(f64, i32)> {
        if i < self.n_total {
            Some((self.v[i], self.refractory[i]))
        } else {
            None
        }
    }
    pub fn n_total(&self) -> usize {
        self.n_total
    }
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }
    pub fn neurons_per_layer(&self) -> usize {
        self.neurons_per_layer
    }
    pub fn step_count(&self) -> u64 {
        self.step_count
    }
    pub fn voltages(&self) -> &[f64] {
        &self.v
    }
    pub fn refractory_counters(&self) -> &[i32] {
        &self.refractory
    }
    pub fn spike_counts_slice(&self) -> &[u64] {
        &self.spike_counts
    }

    pub fn reset(&mut self) {
        self.v.fill(self.params.v_rest);
        self.refractory.fill(0);
        self.spike_counts.fill(0);
        self.step_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn init_sizes() {
        let e = LIFEnsemble::new(10, 100, LIFParams::default()).unwrap();
        assert_eq!(e.n_total(), 1000);
    }
    #[test]
    fn high_current_spikes() {
        let mut e = LIFEnsemble::new(2, 8, LIFParams::default()).unwrap();
        let rates = e.step(&[2.0, 0.5], 100).expect("ok");
        assert!(rates[0] > 0.0);
    }
}
