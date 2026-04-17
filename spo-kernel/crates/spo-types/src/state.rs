// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — State types

use serde::{Deserialize, Serialize};

use crate::action::Regime;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Channel {
    P,
    I,
    S,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhaseState {
    pub theta: f64,
    pub omega: f64,
    pub amplitude: f64,
    pub quality: f64,
    pub channel: Channel,
    pub node_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerState {
    pub r: f64,
    pub psi: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UPDEState {
    pub layers: Vec<LayerState>,
    pub cross_layer_alignment: Vec<f64>,
    pub stability_proxy: f64,
    pub regime: Regime,
}

impl UPDEState {
    #[must_use]
    pub fn mean_r(&self) -> f64 {
        if self.layers.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.layers.iter().map(|l| l.r).sum();
        sum / self.layers.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_serde() {
        let json = serde_json::to_string(&Channel::S).unwrap();
        let c: Channel = serde_json::from_str(&json).unwrap();
        assert_eq!(c, Channel::S);
    }

    #[test]
    fn phase_state_fields() {
        let ps = PhaseState {
            theta: 1.0,
            omega: 2.0,
            amplitude: 0.5,
            quality: 0.9,
            channel: Channel::P,
            node_id: "n0".into(),
        };
        assert_eq!(ps.theta, 1.0);
        assert_eq!(ps.channel, Channel::P);
    }

    #[test]
    fn layer_state_serde() {
        let ls = LayerState { r: 0.8, psi: 1.5 };
        let json = serde_json::to_string(&ls).unwrap();
        let ls2: LayerState = serde_json::from_str(&json).unwrap();
        assert!((ls2.r - 0.8).abs() < 1e-12);
    }

    #[test]
    fn upde_state_mean_r() {
        let state = UPDEState {
            layers: vec![
                LayerState { r: 0.4, psi: 0.0 },
                LayerState { r: 0.8, psi: 0.0 },
            ],
            cross_layer_alignment: vec![],
            stability_proxy: 0.5,
            regime: Regime::Nominal,
        };
        assert!((state.mean_r() - 0.6).abs() < 1e-12);
    }

    #[test]
    fn upde_state_mean_r_empty() {
        let state = UPDEState {
            layers: vec![],
            cross_layer_alignment: vec![],
            stability_proxy: 0.0,
            regime: Regime::Nominal,
        };
        assert_eq!(state.mean_r(), 0.0);
    }
}
