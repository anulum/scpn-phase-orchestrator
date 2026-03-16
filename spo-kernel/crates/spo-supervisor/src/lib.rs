// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Supervisor crate root

#![deny(unsafe_code)]
//! Regime FSM, boundary monitoring, coherence tracking, policy.

pub mod boundaries;
pub mod coherence;
pub mod policy;
pub mod projector;
pub mod regime;

pub use boundaries::{BoundaryDef, BoundaryObserver, BoundaryState, Severity};
pub use coherence::CoherenceMonitor;
pub use policy::SupervisorPolicy;
pub use projector::ActionProjector;
pub use regime::RegimeManager;
