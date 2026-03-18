// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Dormand-Prince RK45 Butcher tableau

//! Shared Dormand-Prince coefficients for adaptive RK45 integration.
//! Used by both `upde::UPDEStepper` and `stuart_landau::StuartLandauStepper`.

pub const A21: f64 = 1.0 / 5.0;
pub const A31: f64 = 3.0 / 40.0;
pub const A32: f64 = 9.0 / 40.0;
pub const A41: f64 = 44.0 / 45.0;
pub const A42: f64 = -56.0 / 15.0;
pub const A43: f64 = 32.0 / 9.0;
pub const A51: f64 = 19372.0 / 6561.0;
pub const A52: f64 = -25360.0 / 2187.0;
pub const A53: f64 = 64448.0 / 6561.0;
pub const A54: f64 = -212.0 / 729.0;
pub const A61: f64 = 9017.0 / 3168.0;
pub const A62: f64 = -355.0 / 33.0;
pub const A63: f64 = 46732.0 / 5247.0;
pub const A64: f64 = 49.0 / 176.0;
pub const A65: f64 = -5103.0 / 18656.0;

// Stage 7 = FSAL evaluation at y5 (same weights as B5)
pub const A71: f64 = 35.0 / 384.0;
pub const A72: f64 = 0.0;
pub const A73: f64 = 500.0 / 1113.0;
pub const A74: f64 = 125.0 / 192.0;
pub const A75: f64 = -2187.0 / 6784.0;
pub const A76: f64 = 11.0 / 84.0;

/// 5th-order weights (solution). k7 weight = 0 (FSAL).
pub const B5: [f64; 7] = [
    35.0 / 384.0,
    0.0,
    500.0 / 1113.0,
    125.0 / 192.0,
    -2187.0 / 6784.0,
    11.0 / 84.0,
    0.0,
];

/// 4th-order weights (error estimate). k7 weight = 1/40.
pub const B4: [f64; 7] = [
    5179.0 / 57600.0,
    0.0,
    7571.0 / 16695.0,
    393.0 / 640.0,
    -92097.0 / 339200.0,
    187.0 / 2100.0,
    1.0 / 40.0,
];
