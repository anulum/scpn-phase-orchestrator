// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Types crate root

#![deny(unsafe_code)]
//! Shared types: state, actions, errors, configuration.

pub mod action;
pub mod config;
pub mod error;
pub mod state;

pub use action::{ControlAction, Knob, Regime};
pub use config::{CouplingConfig, IntegrationConfig, Method};
pub use error::{SpoError, SpoResult};
pub use state::{Channel, LayerState, PhaseState, UPDEState};
