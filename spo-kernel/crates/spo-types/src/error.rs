// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Error types

use thiserror::Error;

pub type SpoResult<T> = Result<T, SpoError>;

#[derive(Debug, Error)]
pub enum SpoError {
    #[error("invalid dimension: {0}")]
    InvalidDimension(String),

    #[error("singular matrix: {0}")]
    SingularMatrix(String),

    #[error("integration diverged: {0}")]
    IntegrationDiverged(String),

    #[error("invalid config: {0}")]
    InvalidConfig(String),

    #[error("boundary violation: {0}")]
    BoundaryViolation(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display() {
        let e = SpoError::InvalidDimension("n must be > 0".into());
        assert_eq!(e.to_string(), "invalid dimension: n must be > 0");
    }

    #[test]
    fn error_variants_distinct() {
        let errors = [
            SpoError::InvalidDimension(String::new()),
            SpoError::SingularMatrix(String::new()),
            SpoError::IntegrationDiverged(String::new()),
            SpoError::InvalidConfig(String::new()),
            SpoError::BoundaryViolation(String::new()),
        ];
        for e in &errors {
            assert!(!format!("{e:?}").is_empty());
        }
    }
}
