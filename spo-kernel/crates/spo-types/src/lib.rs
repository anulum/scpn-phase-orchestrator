// SCPN Phase Orchestrator — Shared Types
// (C) 1998-2026 Miroslav Sotek. All rights reserved.

pub mod action;
pub mod config;
pub mod error;
pub mod state;

pub use action::{ControlAction, Knob, Regime};
pub use config::{CouplingConfig, IntegrationConfig, Method};
pub use error::{SpoError, SpoResult};
pub use state::{Channel, LayerState, PhaseState, UPDEState};
