# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — the ethical cost term refuses bad inputs on every backend

"""``compute_ethical_cost`` must refuse invalid inputs before choosing a backend.

The NumPy path refused NaN phases or couplings while the Rust kernel returned
a NaN or finite cost for them, and both accepted a coupling matrix whose size
did not match the phases, returning different costs.
"""

from __future__ import annotations

import math
from importlib import import_module
from typing import cast

import numpy as np
import pytest

import scpn_phase_orchestrator.ssgf.ethical as ethical
from benchmarks.ethical_cost_reference import reference_cost
from scpn_phase_orchestrator.ssgf.ethical import EthicalKernel, FloatArray
from tests.test_ethical_cost_real_runtime import installed_cost

_K4 = np.full((4, 4), 0.5)
np.fill_diagonal(_K4, 0.0)


@pytest.mark.parametrize(
    ("phases", "knm", "match"),
    [
        (np.array([0.0, math.nan, 2.0, 3.0]), _K4, "phases must contain only finite"),
        (
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.where(_K4 > 0, math.nan, 0.0),
            "knm must contain",
        ),
        (np.zeros(3), _K4, "knm must have shape"),
        (np.zeros((2, 2)), _K4, "one-dimensional"),
    ],
)
def test_invalid_inputs_are_refused(
    phases: FloatArray, knm: FloatArray, match: str
) -> None:
    """Public validation refuses non-finite or mismatched measurements."""
    with pytest.raises(ValueError, match=match):
        ethical.compute_ethical_cost(phases, knm)


@pytest.mark.parametrize("name", ["R_min", "kappa", "max_coupling"])
def test_non_finite_parameters_are_refused(name: str) -> None:
    """Invalid thresholds and penalty weights fail before owner dispatch."""
    with pytest.raises(ValueError, match=f"{name} must be a finite real number"):
        ethical.compute_ethical_cost(np.zeros(4), _K4, **{name: math.nan})


@pytest.mark.native_runtime
def test_backends_agree_on_valid_inputs() -> None:
    """Independent installed owners reproduce the same scalar/eigenvalue oracle."""
    rng = np.random.default_rng(3)
    phases = rng.uniform(0, 2 * np.pi, 6)
    knm = np.abs(rng.normal(size=(6, 6)))
    np.fill_diagonal(knm, 0.0)
    expected = reference_cost(phases, knm, R_min=0.7)
    for owner in ("rust", "python"):
        actual = installed_cost(owner, phases, knm, R_min=0.7)
        np.testing.assert_allclose(actual[:3], expected[:3], rtol=1e-11, atol=1e-11)
        assert actual[3] == expected[3]


@pytest.mark.native_runtime
@pytest.mark.parametrize("use_rust", [False, True])
def test_extreme_finite_coupling_refuses_nonfinite_cost(use_rust: bool) -> None:
    """Actually separate owners refuse a finite but unrepresentable penalty."""
    knm = np.full((4, 4), 1e200)
    np.fill_diagonal(knm, 0.0)
    with pytest.raises(ValueError, match="arithmetic must remain finite"):
        installed_cost("rust" if use_rust else "python", np.zeros(4), knm)


@pytest.mark.native_runtime
def test_direct_rust_ffi_refuses_unrepresentable_cost() -> None:
    """The original compiled FFI refuses overflow in the weighted penalty."""
    kernel = import_module("spo_kernel")
    compute_ethical_cost_rust = cast(
        "EthicalKernel", vars(kernel)["compute_ethical_cost_rust"]
    )
    knm = np.full((2, 2), 1e200)
    np.fill_diagonal(knm, 0.0)
    with pytest.raises(ValueError, match="cost arithmetic must remain finite"):
        compute_ethical_cost_rust(
            np.zeros(2), knm.ravel(), 2, 0.4, 0.3, 0.2, 0.1, 1.0, 0.2, 0.1, 5.0
        )


@pytest.mark.native_runtime
def test_direct_rust_ffi_refuses_nan_before_constraint_clamps() -> None:
    """The original compiled FFI refuses NaN before residual clamping."""
    kernel = import_module("spo_kernel")
    compute_ethical_cost_rust = cast(
        "EthicalKernel", vars(kernel)["compute_ethical_cost_rust"]
    )
    with pytest.raises(ValueError, match="contain only finite values"):
        compute_ethical_cost_rust(
            np.array([0.0, math.nan]),
            np.zeros(4),
            2,
            0.4,
            0.3,
            0.2,
            0.1,
            1.0,
            0.2,
            0.1,
            5.0,
        )
