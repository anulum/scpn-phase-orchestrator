// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Engine crate root

#![deny(unsafe_code)]
//! UPDE integration, coupling, order parameters, lags, imprint, Stuart-Landau, PAC.

pub mod coupling;
pub mod dp_tableau;
pub mod imprint;
pub mod lags;
pub mod lif_ensemble;
pub mod order_params;
pub mod pac;
pub mod stuart_landau;
pub mod upde;

pub use coupling::{CouplingBuilder, CouplingState};
pub use imprint::ImprintModel;
pub use lags::LagModel;
pub use lif_ensemble::{LIFEnsemble, LIFParams};
pub use order_params::{compute_layer_coherence, compute_order_parameter, compute_plv};
pub use pac::{modulation_index, pac_matrix};
pub use stuart_landau::StuartLandauStepper;
pub use upde::UPDEStepper;
pub mod entropy_prod;
pub mod plasticity;
pub mod sheaf_upde;
pub mod sparse_upde;
pub mod winding;
