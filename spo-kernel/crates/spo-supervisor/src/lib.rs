#![deny(unsafe_code)]
//! Regime FSM, boundary monitoring, coherence tracking, policy.
// (C) 1998-2026 Miroslav Sotek. All rights reserved.

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
