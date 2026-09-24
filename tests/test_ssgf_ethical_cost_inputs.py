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

import numpy as np
import pytest

import scpn_phase_orchestrator.ssgf.ethical as ethical

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
    phases: np.ndarray, knm: np.ndarray, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        ethical.compute_ethical_cost(phases, knm)


@pytest.mark.parametrize("name", ["R_min", "kappa", "max_coupling"])
def test_non_finite_parameters_are_refused(name: str) -> None:
    with pytest.raises(ValueError, match=f"{name} must be a finite real number"):
        ethical.compute_ethical_cost(np.zeros(4), _K4, **{name: math.nan})


@pytest.mark.skipif(not ethical._HAS_RUST, reason="spo_kernel not built")
def test_backends_agree_on_valid_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(3)
    phases = rng.uniform(0, 2 * np.pi, 6)
    knm = np.abs(rng.normal(size=(6, 6)))
    np.fill_diagonal(knm, 0.0)
    rust = ethical.compute_ethical_cost(phases, knm, R_min=0.7)
    monkeypatch.setattr(ethical, "_HAS_RUST", False)
    numpy_path = ethical.compute_ethical_cost(phases, knm, R_min=0.7)
    assert rust.c15_sec == pytest.approx(numpy_path.c15_sec, abs=1e-12)
    assert rust.constraints_violated == numpy_path.constraints_violated
