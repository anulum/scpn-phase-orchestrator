// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Error types

use thiserror::Error;

/// Convenient result alias for SPO Rust APIs.
pub type SpoResult<T> = Result<T, SpoError>;

/// Error variants returned by SPO Rust kernels and supervisor crates.
#[derive(Debug, Error)]
pub enum SpoError {
    /// Shape, length, or index-domain mismatch.
    #[error("invalid dimension: {0}")]
    InvalidDimension(String),

    /// Linear solve or matrix factorisation encountered singular structure.
    #[error("singular matrix: {0}")]
    SingularMatrix(String),

    /// Numerical integration produced a non-finite or divergent state.
    #[error("integration diverged: {0}")]
    IntegrationDiverged(String),

    /// Configuration value failed validation.
    #[error("invalid config: {0}")]
    InvalidConfig(String),

    /// Runtime boundary monitor detected a policy violation.
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
