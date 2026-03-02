// SCPN Phase Orchestrator — Boundary Observer

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    Soft,
    Hard,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryDef {
    pub name: String,
    pub variable: String,
    pub lower: Option<f64>,
    pub upper: Option<f64>,
    pub severity: Severity,
}

#[derive(Debug, Clone, Default)]
pub struct BoundaryState {
    pub violations: Vec<String>,
    pub soft_violations: Vec<String>,
    pub hard_violations: Vec<String>,
}

pub struct BoundaryObserver;

impl BoundaryObserver {
    #[must_use]
    pub fn observe(defs: &[BoundaryDef], values: &HashMap<String, f64>) -> BoundaryState {
        let mut state = BoundaryState::default();

        for bdef in defs {
            let val = match values.get(&bdef.variable) {
                Some(&v) => v,
                None => continue,
            };

            let mut violated = false;
            if let Some(lo) = bdef.lower {
                if val < lo {
                    violated = true;
                }
            }
            if let Some(hi) = bdef.upper {
                if val > hi {
                    violated = true;
                }
            }

            if !violated {
                continue;
            }

            let msg = format!(
                "{}: {}={:.4} outside [{:?}, {:?}]",
                bdef.name, bdef.variable, val, bdef.lower, bdef.upper
            );
            state.violations.push(msg.clone());
            match bdef.severity {
                Severity::Soft => state.soft_violations.push(msg),
                Severity::Hard => state.hard_violations.push(msg),
            }
        }

        state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r_boundary() -> BoundaryDef {
        BoundaryDef {
            name: "R_min".into(),
            variable: "R".into(),
            lower: Some(0.3),
            upper: None,
            severity: Severity::Hard,
        }
    }

    fn temp_boundary() -> BoundaryDef {
        BoundaryDef {
            name: "temp_max".into(),
            variable: "temp".into(),
            lower: None,
            upper: Some(100.0),
            severity: Severity::Soft,
        }
    }

    #[test]
    fn no_violations() {
        let defs = vec![r_boundary()];
        let mut values = HashMap::new();
        values.insert("R".into(), 0.8);
        let state = BoundaryObserver::observe(&defs, &values);
        assert!(state.violations.is_empty());
    }

    #[test]
    fn hard_violation() {
        let defs = vec![r_boundary()];
        let mut values = HashMap::new();
        values.insert("R".into(), 0.1);
        let state = BoundaryObserver::observe(&defs, &values);
        assert_eq!(state.hard_violations.len(), 1);
        assert!(state.soft_violations.is_empty());
    }

    #[test]
    fn soft_violation() {
        let defs = vec![temp_boundary()];
        let mut values = HashMap::new();
        values.insert("temp".into(), 120.0);
        let state = BoundaryObserver::observe(&defs, &values);
        assert_eq!(state.soft_violations.len(), 1);
        assert!(state.hard_violations.is_empty());
    }

    #[test]
    fn missing_variable_ignored() {
        let defs = vec![r_boundary()];
        let values = HashMap::new();
        let state = BoundaryObserver::observe(&defs, &values);
        assert!(state.violations.is_empty());
    }

    #[test]
    fn multiple_violations() {
        let defs = vec![r_boundary(), temp_boundary()];
        let mut values = HashMap::new();
        values.insert("R".into(), 0.1);
        values.insert("temp".into(), 150.0);
        let state = BoundaryObserver::observe(&defs, &values);
        assert_eq!(state.violations.len(), 2);
        assert_eq!(state.hard_violations.len(), 1);
        assert_eq!(state.soft_violations.len(), 1);
    }

    #[test]
    fn upper_and_lower_bounds() {
        let def = BoundaryDef {
            name: "range".into(),
            variable: "x".into(),
            lower: Some(0.0),
            upper: Some(1.0),
            severity: Severity::Hard,
        };
        let mut values = HashMap::new();
        values.insert("x".into(), -0.1);
        let state = BoundaryObserver::observe(&[def.clone()], &values);
        assert_eq!(state.violations.len(), 1);

        values.insert("x".into(), 1.5);
        let state = BoundaryObserver::observe(&[def], &values);
        assert_eq!(state.violations.len(), 1);
    }
}
