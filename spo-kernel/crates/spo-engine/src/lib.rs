#![deny(unsafe_code)]
// SCPN Phase Orchestrator — Engine
// (C) 1998-2026 Miroslav Sotek. All rights reserved.

pub mod coupling;
pub mod imprint;
pub mod lags;
pub mod order_params;
pub mod upde;

pub use coupling::{CouplingBuilder, CouplingState};
pub use imprint::ImprintModel;
pub use order_params::{compute_layer_coherence, compute_order_parameter, compute_plv};
pub use upde::UPDEStepper;
