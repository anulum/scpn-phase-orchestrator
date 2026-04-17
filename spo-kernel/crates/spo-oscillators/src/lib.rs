// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Oscillators crate root

#![deny(unsafe_code)]
//! Phase extraction: physical (Hilbert), informational (event timing), symbolic (state-space).

pub mod informational;
pub mod physical;
pub mod quality;
pub mod symbolic;

pub use informational::event_phase;
pub use physical::extract_from_analytic;
pub use quality::PhaseQualityScorer;
pub use symbolic::{graph_walk_phase, ring_phase, transition_quality};
