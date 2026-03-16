// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Action types

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Regime {
    Nominal,
    Degraded,
    Critical,
    Recovery,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Knob {
    K,
    Alpha,
    Zeta,
    Psi,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlAction {
    pub knob: Knob,
    pub scope: String,
    pub value: f64,
    pub ttl_s: f64,
    pub justification: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regime_serde() {
        let json = serde_json::to_string(&Regime::Critical).unwrap();
        let r: Regime = serde_json::from_str(&json).unwrap();
        assert_eq!(r, Regime::Critical);
    }

    #[test]
    fn knob_serde() {
        for knob in [Knob::K, Knob::Alpha, Knob::Zeta, Knob::Psi] {
            let json = serde_json::to_string(&knob).unwrap();
            let k2: Knob = serde_json::from_str(&json).unwrap();
            assert_eq!(k2, knob);
        }
    }

    #[test]
    fn control_action_fields() {
        let a = ControlAction {
            knob: Knob::K,
            scope: "global".into(),
            value: 0.05,
            ttl_s: 10.0,
            justification: "boost coupling".into(),
        };
        assert_eq!(a.knob, Knob::K);
        assert_eq!(a.scope, "global");
        assert_eq!(a.value, 0.05);
    }

    #[test]
    fn regime_all_variants() {
        let variants = [
            Regime::Nominal,
            Regime::Degraded,
            Regime::Critical,
            Regime::Recovery,
        ];
        for v in variants {
            assert!(!format!("{v:?}").is_empty());
        }
    }
}
