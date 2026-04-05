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
pub mod ssgf_costs;
pub mod stuart_landau;
pub mod swarmalator;
pub mod upde;

pub use coupling::{CouplingBuilder, CouplingState};
pub use imprint::ImprintModel;
pub use lags::LagModel;
pub use lif_ensemble::{LIFEnsemble, LIFParams};
pub use order_params::{compute_layer_coherence, compute_order_parameter, compute_plv};
pub use pac::{modulation_index, pac_matrix};
pub use stuart_landau::StuartLandauStepper;
pub use upde::UPDEStepper;
pub mod basin_stability;
pub mod bifurcation;
pub mod carrier;
pub mod chimera;
pub mod connectome;
pub mod coupling_est;
pub mod delay;
pub mod dimension;
pub mod ei_balance;
pub mod embedding;
pub mod entropy_prod;
pub mod envelope;
pub mod ethical;
pub mod evs;
pub mod free_energy;
pub mod freq_id;
pub mod geometric;
pub mod hodge;
pub mod hypergraph;
pub mod inertial;
pub mod itpc;
pub mod lyapunov;
pub mod market;
pub mod npe;
pub mod phase_extract;
pub mod pid;
pub mod plasticity;
pub mod poincare;
pub mod prior;
pub mod psychedelic;
pub mod recurrence;
pub mod reduction;
pub mod sheaf_upde;
pub mod simplicial;
pub mod sindy;
pub mod sleep_staging;
pub mod sparse_upde;
pub mod spectral;
pub mod splitting;
pub mod te_adaptive;
pub mod transfer_entropy;
pub mod winding;
