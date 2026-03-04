#![deny(unsafe_code)]
//! UPDE integration, coupling, order parameters, lags, imprint, Stuart-Landau, PAC.
// (C) 1998-2026 Miroslav Sotek. All rights reserved.

pub mod coupling;
pub mod imprint;
pub mod lags;
pub mod order_params;
pub mod pac;
pub mod stuart_landau;
pub mod upde;

pub use coupling::{CouplingBuilder, CouplingState};
pub use imprint::ImprintModel;
pub use lags::LagModel;
pub use order_params::{compute_layer_coherence, compute_order_parameter, compute_plv};
pub use pac::{modulation_index, pac_matrix};
pub use stuart_landau::StuartLandauStepper;
pub use upde::UPDEStepper;
