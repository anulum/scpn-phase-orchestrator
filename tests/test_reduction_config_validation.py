# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — OA reduction config validation tests

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.reduction import OAState, OttAntonsenReduction


@pytest.mark.parametrize("field", ["omega_0", "delta", "K", "dt"])
@pytest.mark.parametrize("value", [False, "0.1", object()])
def test_oa_reduction_rejects_non_real_constructor_scalars(
    field: str,
    value: Any,
) -> None:
    params: dict[str, Any] = {
        "omega_0": 0.0,
        "delta": 0.1,
        "K": 1.0,
        "dt": 0.01,
    }
    params[field] = value

    with pytest.raises(ValueError, match=f"{field} must be a finite real"):
        OttAntonsenReduction(**params)


@pytest.mark.parametrize("field", ["omega_0", "delta", "K", "dt"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_oa_reduction_rejects_non_finite_constructor_scalars(
    field: str,
    value: float,
) -> None:
    params = {
        "omega_0": 0.0,
        "delta": 0.1,
        "K": 1.0,
        "dt": 0.01,
    }
    params[field] = value

    with pytest.raises(ValueError, match=f"{field} must be a finite real"):
        OttAntonsenReduction(**params)


def test_oa_reduction_normalises_accepted_numpy_scalars() -> None:
    reducer = OttAntonsenReduction(
        omega_0=np.float64(0.2),
        delta=np.float64(0.1),
        K=np.float64(1.0),
        dt=np.float64(0.01),
    )

    assert pytest.approx(0.2) == reducer._omega_0
    assert pytest.approx(0.1) == reducer._delta
    assert pytest.approx(1.0) == reducer._K
    assert pytest.approx(0.01) == reducer._dt


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"z": complex(1.2, 0.0), "R": 0.8, "psi": 0.0, "K_c": 0.2}, "z"),
        ({"z": complex(0.8, 0.0), "R": 1.2, "psi": 0.0, "K_c": 0.2}, "R"),
        ({"z": complex(0.8, 0.0), "R": np.nan, "psi": 0.0, "K_c": 0.2}, "R"),
        ({"z": complex(0.8, 0.0), "R": 0.8, "psi": np.inf, "K_c": 0.2}, "psi"),
        ({"z": complex(0.8, 0.0), "R": 0.8, "psi": 0.0, "K_c": -0.1}, "K_c"),
    ],
)
def test_oa_state_rejects_non_physical_order_parameter_boundary(
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        OAState(**kwargs)


def test_oa_state_accepts_unit_disk_boundary() -> None:
    state = OAState(z=complex(1.0, 0.0), R=1.0, psi=0.0, K_c=0.0)

    assert state.z == complex(1.0, 0.0)
    assert pytest.approx(1.0) == state.R
    assert pytest.approx(0.0) == state.K_c


@pytest.mark.parametrize("z0", [complex(1.01, 0.0), complex(0.8, 0.8)])
def test_oa_reduction_rejects_initial_state_outside_unit_disk(z0: complex) -> None:
    reducer = OttAntonsenReduction(omega_0=0.0, delta=0.1, K=1.0)

    with pytest.raises(ValueError, match="unit disk"):
        reducer.run(z0, n_steps=1)


def test_predict_from_oscillators_rejects_boolean_frequency_aliases() -> None:
    reducer = OttAntonsenReduction(omega_0=0.0, delta=0.1, K=1.0)

    with pytest.raises(ValueError, match="boolean"):
        reducer.predict_from_oscillators(
            np.array([0.9, True, 1.1], dtype=object),
            K=1.0,
        )
