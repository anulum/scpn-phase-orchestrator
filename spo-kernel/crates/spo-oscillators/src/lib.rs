#![deny(unsafe_code)]
//! Phase extraction: physical (Hilbert), informational (event timing), symbolic (state-space).
// (C) 1998-2026 Miroslav Sotek. All rights reserved.

pub mod informational;
pub mod physical;
pub mod quality;
pub mod symbolic;

pub use informational::event_phase;
pub use physical::extract_from_analytic;
pub use quality::PhaseQualityScorer;
pub use symbolic::{graph_walk_phase, ring_phase, transition_quality};
