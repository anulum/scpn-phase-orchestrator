// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Petri net engine (Rust port)

//! Classical Petri net with guard-gated transitions.
//!
//! Parity target: `scpn_phase_orchestrator.supervisor.petri_net`.

use std::collections::HashMap;

/// Arc connecting a place to/from a transition.
#[derive(Debug, Clone)]
pub struct Arc {
    pub place: String,
    pub weight: u32,
}

/// Guard condition: metric op threshold.
#[derive(Debug, Clone)]
pub struct Guard {
    pub metric: String,
    pub op: GuardOp,
    pub threshold: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardOp {
    Gt,
    Ge,
    Lt,
    Le,
    Eq,
}

impl Guard {
    #[must_use]
    pub fn evaluate(&self, ctx: &HashMap<String, f64>) -> bool {
        let Some(&val) = ctx.get(&self.metric) else {
            return false;
        };
        if !val.is_finite() || !self.threshold.is_finite() {
            return false;
        }
        match self.op {
            GuardOp::Gt => val > self.threshold,
            GuardOp::Ge => val >= self.threshold,
            GuardOp::Lt => val < self.threshold,
            GuardOp::Le => val <= self.threshold,
            GuardOp::Eq => (val - self.threshold).abs() < 1e-12,
        }
    }
}

/// Named transition with input/output arcs and optional guard.
#[derive(Debug, Clone)]
pub struct Transition {
    pub name: String,
    pub inputs: Vec<Arc>,
    pub outputs: Vec<Arc>,
    pub guard: Option<Guard>,
}

/// Token distribution across places.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Marking {
    tokens: HashMap<String, u32>,
}

impl Marking {
    #[must_use]
    pub fn get(&self, place: &str) -> u32 {
        self.tokens.get(place).copied().unwrap_or(0)
    }

    /// # Errors
    /// Returns error string if count would go negative (should not happen
    /// with correctly constructed nets).
    pub fn set(&mut self, place: &str, count: u32) {
        if count == 0 {
            self.tokens.remove(place);
        } else {
            self.tokens.insert(place.to_string(), count);
        }
    }

    #[must_use]
    pub fn active_places(&self) -> Vec<&str> {
        self.tokens
            .iter()
            .filter(|(_, &n)| n > 0)
            .map(|(p, _)| p.as_str())
            .collect()
    }
}

/// Classical Petri net.  `step()` fires at most one enabled transition
/// per call (first-match priority order).
pub struct PetriNet {
    place_names: Vec<String>,
    transitions: Vec<Transition>,
}

impl PetriNet {
    /// # Errors
    /// Returns error if any arc references an unknown place.
    pub fn new(places: Vec<String>, transitions: Vec<Transition>) -> Result<Self, String> {
        for place in &places {
            if place.is_empty() {
                return Err("place names must not be empty".into());
            }
        }
        for t in &transitions {
            if t.name.is_empty() {
                return Err("transition names must not be empty".into());
            }
            for arc in t.inputs.iter().chain(t.outputs.iter()) {
                if arc.weight == 0 {
                    return Err(format!(
                        "transition {:?} has zero-weight arc for place {:?}",
                        t.name, arc.place
                    ));
                }
                if !places.contains(&arc.place) {
                    return Err(format!(
                        "transition {:?} references unknown place {:?}",
                        t.name, arc.place
                    ));
                }
            }
        }
        Ok(Self {
            place_names: places,
            transitions,
        })
    }

    #[must_use]
    pub fn place_names(&self) -> &[String] {
        &self.place_names
    }

    #[must_use]
    pub fn transitions(&self) -> &[Transition] {
        &self.transitions
    }

    #[must_use]
    pub fn transition_names(&self) -> Vec<String> {
        self.transitions
            .iter()
            .map(|transition| transition.name.clone())
            .collect()
    }

    #[must_use]
    pub fn enabled(&self, marking: &Marking, ctx: &HashMap<String, f64>) -> Vec<usize> {
        self.transitions
            .iter()
            .enumerate()
            .filter(|(_, t)| {
                if let Some(g) = &t.guard {
                    if !g.evaluate(ctx) {
                        return false;
                    }
                }
                t.inputs.iter().all(|a| marking.get(&a.place) >= a.weight)
            })
            .map(|(i, _)| i)
            .collect()
    }

    #[must_use]
    pub fn fire(&self, marking: &Marking, transition_idx: usize) -> Marking {
        let t = &self.transitions[transition_idx];
        let mut new = marking.clone();
        for arc in &t.inputs {
            let cur = new.get(&arc.place);
            new.set(&arc.place, cur.saturating_sub(arc.weight));
        }
        for arc in &t.outputs {
            new.set(&arc.place, new.get(&arc.place) + arc.weight);
        }
        new
    }

    /// Fire the first enabled transition. Returns (new_marking, fired_index).
    #[must_use]
    pub fn step(&self, marking: &Marking, ctx: &HashMap<String, f64>) -> (Marking, Option<usize>) {
        for (i, t) in self.transitions.iter().enumerate() {
            if let Some(g) = &t.guard {
                if !g.evaluate(ctx) {
                    continue;
                }
            }
            if t.inputs.iter().all(|a| marking.get(&a.place) >= a.weight) {
                return (self.fire(marking, i), Some(i));
            }
        }
        (marking.clone(), None)
    }
}

/// Parse guard string like "stability_proxy > 0.6".
///
/// # Errors
/// Returns error if format is invalid.
pub fn parse_guard(text: &str) -> Result<Guard, String> {
    let parts: Vec<&str> = text.split_whitespace().collect();
    if parts.len() != 3 {
        return Err(format!("guard must be 'metric op threshold', got {text:?}"));
    }
    let op = match parts[1] {
        ">" => GuardOp::Gt,
        ">=" => GuardOp::Ge,
        "<" => GuardOp::Lt,
        "<=" => GuardOp::Le,
        "==" => GuardOp::Eq,
        other => return Err(format!("unknown guard operator: {other:?}")),
    };
    let threshold: f64 = parts[2]
        .parse()
        .map_err(|e| format!("invalid threshold: {e}"))?;
    if !threshold.is_finite() {
        return Err(format!("guard threshold must be finite, got {threshold}"));
    }
    Ok(Guard {
        metric: parts[0].to_string(),
        op,
        threshold,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_net() -> PetriNet {
        PetriNet::new(
            vec!["idle".into(), "active".into(), "done".into()],
            vec![
                Transition {
                    name: "start".into(),
                    inputs: vec![Arc {
                        place: "idle".into(),
                        weight: 1,
                    }],
                    outputs: vec![Arc {
                        place: "active".into(),
                        weight: 1,
                    }],
                    guard: None,
                },
                Transition {
                    name: "finish".into(),
                    inputs: vec![Arc {
                        place: "active".into(),
                        weight: 1,
                    }],
                    outputs: vec![Arc {
                        place: "done".into(),
                        weight: 1,
                    }],
                    guard: None,
                },
            ],
        )
        .expect("valid net")
    }

    #[test]
    fn step_fires_first_enabled() {
        let net = simple_net();
        let mut m = Marking::default();
        m.set("idle", 1);
        let (m2, fired) = net.step(&m, &HashMap::new());
        assert_eq!(fired, Some(0));
        assert_eq!(m2.get("idle"), 0);
        assert_eq!(m2.get("active"), 1);
    }

    #[test]
    fn step_no_enabled_returns_same() {
        let net = simple_net();
        let m = Marking::default();
        let (m2, fired) = net.step(&m, &HashMap::new());
        assert_eq!(fired, None);
        assert_eq!(m, m2);
    }

    #[test]
    fn guard_blocks_transition() {
        let net = PetriNet::new(
            vec!["a".into(), "b".into()],
            vec![Transition {
                name: "guarded".into(),
                inputs: vec![Arc {
                    place: "a".into(),
                    weight: 1,
                }],
                outputs: vec![Arc {
                    place: "b".into(),
                    weight: 1,
                }],
                guard: Some(Guard {
                    metric: "x".into(),
                    op: GuardOp::Gt,
                    threshold: 0.5,
                }),
            }],
        )
        .expect("valid");

        let mut m = Marking::default();
        m.set("a", 1);

        let ctx_low: HashMap<String, f64> = [("x".into(), 0.3)].into();
        let (_, fired) = net.step(&m, &ctx_low);
        assert_eq!(fired, None);

        let ctx_high: HashMap<String, f64> = [("x".into(), 0.8)].into();
        let (m2, fired) = net.step(&m, &ctx_high);
        assert_eq!(fired, Some(0));
        assert_eq!(m2.get("b"), 1);
    }

    #[test]
    fn unknown_place_rejected() {
        let result = PetriNet::new(
            vec!["a".into()],
            vec![Transition {
                name: "bad".into(),
                inputs: vec![Arc {
                    place: "nonexistent".into(),
                    weight: 1,
                }],
                outputs: vec![],
                guard: None,
            }],
        );
        assert!(result.is_err());
    }

    #[test]
    fn empty_place_rejected() {
        let result = PetriNet::new(vec!["".into()], vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn empty_transition_rejected() {
        let result = PetriNet::new(
            vec!["a".into()],
            vec![Transition {
                name: String::new(),
                inputs: vec![],
                outputs: vec![],
                guard: None,
            }],
        );
        assert!(result.is_err());
    }

    #[test]
    fn zero_weight_arc_rejected() {
        let result = PetriNet::new(
            vec!["a".into()],
            vec![Transition {
                name: "bad".into(),
                inputs: vec![Arc {
                    place: "a".into(),
                    weight: 0,
                }],
                outputs: vec![],
                guard: None,
            }],
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_guard_valid() {
        let g = parse_guard("stability_proxy > 0.6").expect("valid");
        assert_eq!(g.metric, "stability_proxy");
        assert_eq!(g.op, GuardOp::Gt);
        assert!((g.threshold - 0.6).abs() < 1e-12);
    }

    #[test]
    fn parse_guard_all_ops() {
        for (text, expected_op) in [
            ("x > 1", GuardOp::Gt),
            ("x >= 1", GuardOp::Ge),
            ("x < 1", GuardOp::Lt),
            ("x <= 1", GuardOp::Le),
            ("x == 1", GuardOp::Eq),
        ] {
            let g = parse_guard(text).expect("valid");
            assert_eq!(g.op, expected_op);
        }
    }

    #[test]
    fn parse_guard_invalid_format() {
        assert!(parse_guard("bad").is_err());
        assert!(parse_guard("x > > 1").is_err());
    }

    #[test]
    fn parse_guard_rejects_non_finite_threshold() {
        assert!(parse_guard("x > NaN").is_err());
        assert!(parse_guard("x > inf").is_err());
    }

    #[test]
    fn guard_evaluate_missing_metric() {
        let g = Guard {
            metric: "missing".into(),
            op: GuardOp::Gt,
            threshold: 0.5,
        };
        assert!(!g.evaluate(&HashMap::new()));
    }

    #[test]
    fn guard_rejects_non_finite_context_metric() {
        let g = Guard {
            metric: "x".into(),
            op: GuardOp::Gt,
            threshold: 0.5,
        };
        let ctx: HashMap<String, f64> = [("x".into(), f64::INFINITY)].into();
        assert!(!g.evaluate(&ctx));
    }

    #[test]
    fn transition_names_preserve_priority_order() {
        let net = simple_net();
        assert_eq!(
            net.transition_names(),
            vec!["start".to_string(), "finish".to_string()]
        );
    }

    #[test]
    fn sequential_firing() {
        let net = simple_net();
        let mut m = Marking::default();
        m.set("idle", 1);

        let (m, _) = net.step(&m, &HashMap::new());
        assert_eq!(m.get("active"), 1);

        let (m, _) = net.step(&m, &HashMap::new());
        assert_eq!(m.get("done"), 1);
        assert_eq!(m.get("active"), 0);
    }

    #[test]
    fn enabled_returns_all_fireable() {
        let net = PetriNet::new(
            vec!["a".into(), "b".into(), "c".into()],
            vec![
                Transition {
                    name: "t1".into(),
                    inputs: vec![Arc {
                        place: "a".into(),
                        weight: 1,
                    }],
                    outputs: vec![Arc {
                        place: "b".into(),
                        weight: 1,
                    }],
                    guard: None,
                },
                Transition {
                    name: "t2".into(),
                    inputs: vec![Arc {
                        place: "a".into(),
                        weight: 1,
                    }],
                    outputs: vec![Arc {
                        place: "c".into(),
                        weight: 1,
                    }],
                    guard: None,
                },
            ],
        )
        .expect("valid");

        let mut m = Marking::default();
        m.set("a", 2);
        let en = net.enabled(&m, &HashMap::new());
        assert_eq!(en.len(), 2);
    }
}
