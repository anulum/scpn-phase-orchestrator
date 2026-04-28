// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Supervisor event types

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventKind {
    BoundaryBreach,
    RThreshold,
    RegimeTransition,
    Manual,
    PetriTransition,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegimeEvent {
    pub kind: EventKind,
    pub step: u64,
    pub detail: String,
}

impl RegimeEvent {
    #[must_use]
    pub fn new(kind: EventKind, step: u64, detail: String) -> Self {
        Self { kind, step, detail }
    }
}

/// Bounded event history. Callbacks are handled on the Python side;
/// the Rust EventBus is a pure append-only ring buffer for audit replay.
#[derive(Debug)]
pub struct EventBus {
    history: VecDeque<RegimeEvent>,
    max_history: usize,
}

impl EventBus {
    #[must_use]
    pub fn new(max_history: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(max_history),
            max_history,
        }
    }

    pub fn post(&mut self, event: RegimeEvent) {
        if self.history.len() == self.max_history {
            self.history.pop_front();
        }
        self.history.push_back(event);
    }

    #[must_use]
    pub fn history(&self) -> &VecDeque<RegimeEvent> {
        &self.history
    }

    #[must_use]
    pub fn count(&self) -> usize {
        self.history.len()
    }

    pub fn clear(&mut self) {
        self.history.clear();
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new(200)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_creation() {
        let e = RegimeEvent::new(EventKind::RegimeTransition, 5, "a->b".into());
        assert_eq!(e.kind, EventKind::RegimeTransition);
        assert_eq!(e.step, 5);
        assert_eq!(e.detail, "a->b");
    }

    #[test]
    fn bus_post_and_history() {
        let mut bus = EventBus::new(10);
        bus.post(RegimeEvent::new(EventKind::Manual, 1, String::new()));
        bus.post(RegimeEvent::new(EventKind::Manual, 2, String::new()));
        assert_eq!(bus.count(), 2);
    }

    #[test]
    fn bus_bounded_history() {
        let mut bus = EventBus::new(3);
        for i in 0..5 {
            bus.post(RegimeEvent::new(EventKind::Manual, i, String::new()));
        }
        assert_eq!(bus.count(), 3);
        assert_eq!(bus.history().front().expect("not empty").step, 2);
    }

    #[test]
    fn bus_clear() {
        let mut bus = EventBus::default();
        bus.post(RegimeEvent::new(EventKind::Manual, 1, String::new()));
        bus.clear();
        assert_eq!(bus.count(), 0);
    }

    #[test]
    fn event_serde_roundtrip() {
        let e = RegimeEvent::new(EventKind::BoundaryBreach, 42, "test".into());
        let json = serde_json::to_string(&e).expect("serialize");
        let e2: RegimeEvent = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(e, e2);
    }
}
